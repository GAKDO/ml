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
    assert sum(r.images.values()) == 14 and r.images["val"] >= 1
    assert r.shapes == {"scratch": 12, "dent": 1, "crack": 1}
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
