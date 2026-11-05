"""AI Hub 사전 학습: 여러 모양의 라벨 JSON을 읽어 YOLO-seg 데이터셋을 만든다."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from trace_ml.__main__ import main
from trace_ml.pretrain.aihub import parse_points, read_items
from trace_ml.pretrain.build import MapError, build, load_map, scan


def _photo(path: Path, w: int = 800, h: int = 600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((h, w, 3), 128, np.uint8))


def _aihub(root: Path, n: int = 12) -> None:
    """원천·라벨링 폴더가 나뉜 이미지별 JSON(AI Hub 흔한 모양) + labelme 하나 + COCO 하나."""
    for i in range(n):
        _photo(root / "원천데이터" / f"P{i:03d}.jpg")
        doc = {"images": {"file_name": f"P{i:03d}.jpg", "width": 800, "height": 600},
               "annotations": [{"class": "긁힘", "polygon": [[10, 10], [200, 12], [205, 40], [12, 35]]},
                               {"class": "흘러내림", "polygon": [10, 100, 60, 100, 60, 150]},
                               {"class": "찍힘", "bbox": [1, 2, 3, 4]}]}
        (root / "라벨링데이터").mkdir(parents=True, exist_ok=True)
        (root / "라벨링데이터" / f"P{i:03d}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    _photo(root / "원천데이터" / "L000.png")
    labelme = {"imagePath": "..\\원천데이터\\L000.png", "imageWidth": 800, "imageHeight": 600,
               "shapes": [{"label": "dent", "points": [[100, 100], [150, 100], [150, 140]], "shape_type": "polygon"},
                          {"label": "dent", "points": [[1, 1], [5, 5]], "shape_type": "rectangle"}]}
    (root / "라벨링데이터" / "L000.json").write_text(json.dumps(labelme), encoding="utf-8")
    _photo(root / "원천데이터" / "C000.jpg")
    coco = {"images": [{"id": 1, "file_name": "C000.jpg", "width": 800, "height": 600}],
            "categories": [{"id": 7, "name": "crack"}],
            "annotations": [{"image_id": 1, "category_id": 7, "segmentation": [[300, 300, 400, 310, 350, 380]]}]}
    (root / "라벨링데이터" / "coco.json").write_text(json.dumps(coco), encoding="utf-8")


def test_여러_모양의_점_목록을_폴리곤으로_읽는다() -> None:
    assert parse_points([[0, 0], [1, 0], [1, 1]]) == [[(0, 0), (1, 0), (1, 1)]]
    assert parse_points([0, 0, 1, 0, 1, 1]) == [[(0, 0), (1, 0), (1, 1)]]
    assert parse_points([{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}]) == [[(0, 0), (1, 0), (1, 1)]]
    assert len(parse_points([[0, 0, 1, 0, 1, 1], [5, 5, 6, 5, 6, 6]])) == 2  # COCO 여러 조각
    assert parse_points({"x": [0, 1, 1], "y": [0, 0, 1]}) == [[(0, 0), (1, 0), (1, 1)]]


def test_scan은_라벨_이름과_개수_상자만_있는_객체를_보여_준다(tmp_path: Path) -> None:
    _aihub(tmp_path, 3)
    r = scan(read_items([tmp_path]))
    assert r.labels["긁힘"] == 3 and r.labels["흘러내림"] == 3 and r.labels["dent"] == 1 and r.labels["crack"] == 1
    assert r.boxes_only == 4  # 이미지별 bbox 3 + labelme rectangle 1
    assert r.no_image == 0


def test_build는_매핑한_클래스만_남기고_사진을_jpg로_나눠_담는다(tmp_path: Path) -> None:
    _aihub(tmp_path / "src", 12)
    mapping = {"긁힘": "scratch", "흘러내림": None, "dent": "dent", "crack": "crack"}
    r = build(read_items([tmp_path / "src"]), mapping, tmp_path / "out", val=0.3, test=0.0)
    # L000은 dent 상자(폴리곤 없음)가 섞여 있어 통째로 뺀다
    assert sum(r.images.values()) == 13 and r.images["val"] >= 1
    assert r.shapes == {"scratch": 12, "crack": 1}
    assert r.skipped["상자로만 표시된 흠집 있음"] == 1
    assert r.dropped_labels["흘러내림"] == 12
    labels = list((tmp_path / "out" / "labels").rglob("*.txt"))
    first = next(p for p in labels if p.read_text(encoding="utf-8").startswith("0 "))
    nums = [float(v) for v in first.read_text(encoding="utf-8").split()[1:]]
    assert nums[:2] == pytest.approx([10 / 800, 10 / 600], abs=1e-5)
    assert all(p.suffix == ".jpg" for p in (tmp_path / "out" / "images").rglob("*.*"))


def test_map에_우리_클래스가_아닌_값이_있으면_멈춘다(tmp_path: Path) -> None:
    p = tmp_path / "map.yaml"
    p.write_text("긁힘: scratches\n", encoding="utf-8")
    with pytest.raises(MapError):
        load_map(p)


def test_scan_명령은_map_초안을_쓰고_build_명령은_640_타일까지_만든다(tmp_path: Path) -> None:
    _aihub(tmp_path / "src", 12)
    draft = tmp_path / "map.yaml"
    assert main(["pretrain", "scan", str(tmp_path / "src"), "--write-map", str(draft)], env={}) == 0
    assert "긁힘" in draft.read_text(encoding="utf-8")
    draft.write_text("긁힘: scratch\n흘러내림: null\ndent: dent\ncrack: crack\n", encoding="utf-8")
    out = tmp_path / "ds"
    assert main(["pretrain", "build", str(tmp_path / "src"), "--map", str(draft), "--out", str(out),
                 "--val", "0.3"], env={}) == 0
    info = json.loads((out / "build.json").read_text(encoding="utf-8"))
    assert info["tile"] == 640 and info["counts"]["train"]["defect"] > 0
    assert (out / "tile640" / "data.yaml").exists()


def test_같은_out에_다시_빌드하면_지우지_않고_멈춘다(tmp_path: Path) -> None:
    from trace_ml.pretrain.build import build

    dest = tmp_path / "raw"
    (dest / "images" / "train").mkdir(parents=True)
    (dest / "images" / "train" / "old.jpg").write_bytes(b"x")
    with pytest.raises(ValueError, match="이전 빌드 결과"):
        build([], {}, dest)
    assert (dest / "images" / "train" / "old.jpg").exists()


def test_빈_사진_선별은_분할_해시와_독립이라_train에도_들어간다() -> None:
    from trace_ml.pretrain.build import _unit, split_of

    kept = [Path(f"p{i}.jpg") for i in range(2000) if _unit(Path(f"p{i}.jpg"), "empty") < 0.1]
    assert any(split_of(p, 0.1, 0.05) == "train" for p in kept)


def test_선박_도장_COCO_JSON_평평한_segmentation을_읽고_긴_변을_줄인다(tmp_path: Path) -> None:
    lbl, img, out = tmp_path / "lbl", tmp_path / "img", tmp_path / "out"
    lbl.mkdir()
    img.mkdir()
    doc = {"images": [{"id": 1, "file_name": "302_19_a.jpg", "width": 4032, "height": 3024}],
           "categories": [{"id": 302, "name": "스크래치"}],
           "annotations": [{"id": 1, "image_id": 1, "category_id": 302, "iscrowd": 0,
                            "segmentation": [257.5, 1300.9, 844.9, 1380.4, 4031.9, 1239.6, 4031.9, 1177.3]}]}
    (lbl / "302_19_a.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    cv2.imwrite(str(img / "302_19_a.jpg"), np.full((3024, 4032, 3), 128, np.uint8))
    report = build(read_items([lbl, img]), {"스크래치": "scratch"}, out, val=0, test=0, max_side=2048)
    assert report.shapes["scratch"] == 1
    saved = next((out / "images" / "train").glob("*.jpg"))
    assert cv2.imread(str(saved)).shape[:2] == (1536, 2048)
    xs = [float(v) for v in next((out / "labels" / "train").glob("*.txt")).read_text().split()[1::2]]
    assert max(xs) == pytest.approx(1.0, abs=1e-4)


def test_긴_변_제한이_음수면_빌드를_시작하지_않는다(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_side"):
        build([], {}, tmp_path / "out", max_side=-1)


def test_JSON의_JPG와_실제_jpg_대소문자가_달라도_사진을_찾는다(tmp_path: Path) -> None:
    doc = {"images": [{"id": 1, "file_name": "2502_12_a.JPG", "width": 800, "height": 600}],
           "categories": [{"id": 2502, "name": "스크래치"}],
           "annotations": [{"id": 1, "image_id": 1, "category_id": 2502, "segmentation": [10, 10, 200, 12, 205, 40]}]}
    (tmp_path / "a.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    _photo(tmp_path / "img" / "2502_12_a.jpg")
    r = scan(read_items([tmp_path]))
    assert r.no_image == 0 and r.labels["스크래치"] == 1


def test_쓰는_클래스가_상자로만_표시된_사진은_흠집_없는_사진으로_넣지_않는다(tmp_path: Path) -> None:
    doc = {"images": [{"id": 1, "file_name": "b.jpg", "width": 800, "height": 600}],
           "categories": [{"id": 1, "name": "스크래치"}, {"id": 2, "name": "볼트"}],
           "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 30, 40], "segmentation": []},
                           {"id": 2, "image_id": 1, "category_id": 2, "bbox": [5, 5, 9, 9], "segmentation": []}]}
    (tmp_path / "b.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    _photo(tmp_path / "b.jpg")
    assert scan(read_items([tmp_path])).box_labels == {"스크래치": 1, "볼트": 1}
    r = build(read_items([tmp_path]), {"스크래치": "scratch", "볼트": None}, tmp_path / "out", keep_empty=1.0)
    assert sum(r.images.values()) == 0 and r.skipped["상자로만 표시된 흠집 있음"] == 1
    # 버리는 클래스만 상자면 흠집 없는 사진으로 둘 수 있다
    r2 = build(read_items([tmp_path]), {"스크래치": None, "볼트": None}, tmp_path / "out2", keep_empty=1.0)
    assert sum(r2.images.values()) == 1


def test_scan의_map_초안은_없는_폴더도_만들어_쓴다(tmp_path: Path) -> None:
    _aihub(tmp_path / "src", 2)
    draft = tmp_path / "새폴더" / "map.yaml"
    assert main(["pretrain", "scan", str(tmp_path / "src"), "--write-map", str(draft)], env={}) == 0
    assert draft.exists()
