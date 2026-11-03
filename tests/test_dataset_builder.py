"""ML1 스토리 1: 제출된 주석으로 기기 단위 데이터셋 버전을 만든다 (FR-DAT-04)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
import yaml
from fakes import NO_DEFECT, SETTINGS, TRI, World, ann, polygon

from trace_ml.__main__ import main
from trace_ml.dataset.annotations import AnnotationErrors, collect
from trace_ml.dataset.builder import BuildError, Commit, build, next_version
from trace_ml.dataset.cli import Deps
from trace_ml.dataset.splits import SplitError, assign, check_change
from trace_ml.registry import Device, VersionExists
from trace_ml.store import AlreadyExists

CLEAN = Commit("a" * 40, False)
NOW = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)


def _build(w: World, **kw):
    return build(w.store, w.registry, SETTINGS, kw.pop("commit", CLEAN), now=NOW, **kw)


def _manifest(w: World, version: str = "v001") -> dict:
    return json.loads(w.s3.objects[("trace-datasets", f"{version}/manifest.json")][0])


def three_devices() -> World:
    w = World()
    w.registry.save_device(Device("REG03", "MacBook Air", "TEST"))
    for code in ("REG01", "REG02", "REG03"):
        w.labeled(code, 3)
    return w


def test_한_기기의_사진은_모두_한_분할에만_있고_manifest의_기기표와_같다():
    w = three_devices()
    result = _build(w)
    m = _manifest(w)
    split_of: dict[str, set[str]] = {}
    for item in m["items"]:
        split_of.setdefault(item["device"], set()).add(item["split"])
    assert all(len(s) == 1 for s in split_of.values())
    assert {d: s.pop().upper() for d, s in split_of.items()} == m["devices"]
    assert m["devices"]["REG03"] == "TEST"  # 사람이 먼저 정한 분할이 이긴다
    placed = [c for codes in result.counts["devices"].values() for c in codes]
    assert sorted(placed) == ["REG01", "REG02", "REG03"]


def test_라벨_파일은_YOLO_분할_형식이고_클래스_번호와_0에서_1_좌표다():
    w = World()
    w.photo("REG01/REG01_TOP_L1_CAM1_01.jpg", ann(1, polygon("dent", [(0, 0), (50, 0), (50, 120)])))
    _build(w)
    body = w.s3.objects[("trace-datasets", "v001/labels/train/REG01_TOP_L1_CAM1_01.txt")][0].decode()
    assert body == "1 0.000000 0.000000 0.500000 0.000000 0.500000 1.000000\n"
    data = yaml.safe_load(w.s3.objects[("trace-datasets", "v001/data.yaml")][0])
    assert data["names"] == {0: "scratch", 1: "dent", 2: "crack", 3: "stain"}
    assert data["train"] == "images/train"


def test_manifest에_사진_해시_빌더_커밋_codes_버전이_있고_기록부에_버전이_남는다():
    w = three_devices()
    _build(w)
    m = _manifest(w)
    item = m["items"][0]
    assert len(item["image"]["sha256"]) == 64 and len(item["label"]["sha256"]) == 64
    assert m["builderCommit"] == "a" * 40 and m["codesVersion"] and m["builtAt"] == "2026-11-05T12:00:00Z"
    row = w.registry.versions()[0]
    assert (row.version, row.manifest_key, row.builder_commit) == ("v001", "v001/manifest.json", "a" * 40)
    assert len(row.counts["manifestSha256"]) == 64
    # manifest는 마지막에 쓴다
    assert w.s3.puts[-1] == ("trace-datasets", "v001/manifest.json")


def test_흠집_없음으로_제출한_사진은_빈_라벨_파일_배경이다():
    w = World()
    w.photo("REG01/REG01_TOP_L1_CAM1_01.jpg", ann(1, NO_DEFECT))
    result = _build(w)
    assert w.s3.objects[("trace-datasets", "v001/labels/train/REG01_TOP_L1_CAM1_01.txt")][0] == b""
    assert result.counts["background"]["train"] == 1


def test_건너뛴_주석이_섞여_있으면_그_사진은_데이터셋에_없다():
    w = World()
    w.labeled("REG01", 1)
    w.photo("REG01/REG01_BOT_L1_CAM1_01.jpg", ann(7, cancelled=True))
    _build(w)
    keys = [i["image"]["key"] for i in _manifest(w)["items"]]
    assert keys == ["REG01/REG01_TOP_L1_CAM1_01.jpg"]


def test_주석이_여럿이면_ground_truth_없으면_가장_최근_것을_쓴다():
    old = ann(1, polygon("scratch", TRI), updated="2026-11-03T00:00:00Z")
    new = ann(2, polygon("dent", TRI), polygon("dent", TRI), updated="2026-11-04T00:00:00Z")
    gt = ann(3, NO_DEFECT, updated="2026-11-01T00:00:00Z", ground_truth=True)
    doc = {"id": 5, "data": {"image": "s3://trace-raw/REG01/REG01_TOP_L1_CAM1_01.jpg"}}
    assert [p.cls for p in collect([{**doc, "annotations": [old, new]}])[0].polygons] == [1, 1]
    assert collect([{**doc, "annotations": [old, new, gt]}])[0].annotation_id == 3


def test_옛_주석_형식_내보내기도_읽는다():
    doc = {"id": 9, "result": [polygon("crack", TRI)], "was_cancelled": False,
           "task": {"id": 4, "data": {"image": "s3://trace-raw/REG01/REG01_SL_L2_CAM1_01.jpg"}}}
    [img] = collect([doc])
    assert (img.task_id, img.annotation_id, img.device, img.polygons[0].cls) == (4, 9, "REG01", 2)


def test_v001이_이미_있으면_v001로_다시_만들기를_거절하고_기존_객체는_그대로다():
    w = three_devices()
    _build(w)
    before = dict(w.s3.objects)
    w.labeled("REG04", 1)
    with pytest.raises(VersionExists):
        _build(w, version="v001")
    assert {k: v for k, v in w.s3.objects.items() if k[0] == "trace-datasets"} == \
        {k: v for k, v in before.items() if k[0] == "trace-datasets"}


def test_버킷에만_남은_버전도_덮어쓰지_않는다():
    w = three_devices()
    w.s3.put_object(Bucket="trace-datasets", Key="v001/manifest.json", Body=b"{}")
    with pytest.raises(AlreadyExists):
        _build(w, version="v001")


def test_조건부_쓰기가_거절되면_이미_있음으로_멈춘다():
    w = World()
    w.store.put_new("trace-datasets", "x", b"1")
    with pytest.raises(AlreadyExists):
        w.store.put_new("trace-datasets", "x", b"2")
    assert w.s3.objects[("trace-datasets", "x")][0] == b"1"


def test_버전을_정하지_않으면_다음_번호로_만든다():
    w = three_devices()
    _build(w)
    assert _build(w).version == "v002"
    assert next_version(["v001", "v009", "x"]) == "v010"
    with pytest.raises(BuildError):
        _build(w, version="2")


def test_빈_제출이_있으면_작업_번호를_알리며_실패하고_아무것도_쓰지_않는다():
    w = World()
    w.labeled("REG01", 2)
    tid = w.photo("REG01/REG01_BOT_L1_CAM1_01.jpg", ann(50))
    both = w.photo("REG01/REG01_SF_L1_CAM1_01.jpg", ann(51, NO_DEFECT, polygon("scratch", TRI)))
    with pytest.raises(AnnotationErrors) as exc:
        _build(w)
    text = " ".join(exc.value.problems)
    assert f"작업 {tid}" in text and f"작업 {both}" in text
    assert w.s3.keys("trace-datasets") == [] and w.registry.versions() == []


@pytest.mark.parametrize("result, image", [
    ([polygon("rust", TRI)], "REG01/REG01_TOP_L1_CAM1_01.jpg"),
    ([polygon("scratch", TRI[:2])], "REG01/REG01_TOP_L1_CAM1_01.jpg"),
    ([NO_DEFECT], "REG01/slate/REG01_SLATE.jpg"),
])
def test_codes_밖_클래스_점_2개_파일명_규칙_위반은_빌드를_멈춘다(result, image):
    doc = {"id": 1, "data": {"image": f"s3://trace-raw/{image}"}, "annotations": [ann(1, *result)]}
    with pytest.raises(AnnotationErrors):
        collect([doc])


def test_주석이_없거나_JSON이_아니면_시작하지_않는다():
    w = World()
    with pytest.raises(BuildError):
        _build(w)
    w.s3.put_object(Bucket="trace-labels", Key="1", Body=b"not json")
    with pytest.raises(AnnotationErrors):
        _build(w)


def test_커밋하지_않은_변경이_있으면_allow_dirty_없이는_멈춘다():
    w = three_devices()
    with pytest.raises(BuildError):
        _build(w, commit=Commit("b" * 40, True))
    _build(w, commit=Commit("b" * 40, True), allow_dirty=True)
    assert _manifest(w)["builderDirty"] is True


def test_자동_배정은_기기_수_기준_70_15_15에_가장_모자란_쪽이고_결정적이다():
    table, new = assign({f"G{i:02d}" for i in range(1, 8)}, {})
    assert list(table.values()).count("TRAIN") == 5
    assert list(table.values()).count("VAL") == 1 and list(table.values()).count("TEST") == 1
    assert assign({f"G{i:02d}" for i in range(1, 8)}, {})[0] == table
    assert all(d.model_name == "미등록" for d in new)


def test_v001에_TEST로_쓰인_기기를_TRAIN으로_바꾸려_하면_거절된다():
    w = three_devices()
    _build(w)
    known = w.registry.devices()
    with pytest.raises(SplitError):
        check_change(Device("REG03", "MacBook Air", "TRAIN"), known, w.registry.used_devices())
    check_change(Device("REG09", "MacBook Air", "TRAIN"), known, w.registry.used_devices())
    check_change(Device("REG03", "MacBook Air M2", "TEST"), known, w.registry.used_devices())


def test_명령줄_device_add_build_list가_돌고_분할_변경_거절은_1로_끝난다(capsys):
    w = World()
    w.labeled("REG01", 2)
    deps = Deps({}, registry=lambda: w.registry, store=lambda: w.store, commit=lambda: CLEAN)
    env = {"MINIO_ENDPOINT": "http://minio.test", "MINIO_ACCESS_KEY": "ak", "MINIO_SECRET_KEY": "sk"}
    deps.env = env
    assert main(["dataset", "device", "add", "REG01", "--split", "VAL", "--model", "MacBook Air"], deps=deps) == 0
    assert main(["dataset", "build"], deps=deps) == 0
    assert main(["dataset", "list"], deps=deps) == 0
    assert main(["dataset", "device", "list"], deps=deps) == 0
    assert main(["dataset", "device", "add", "REG01", "--split", "TRAIN"], deps=deps) == 1
    assert main(["dataset", "device", "add", "REG05"], deps=deps) == 1
    out = capsys.readouterr()
    assert "v001 만듦: train 0 · val 2 · test 0" in out.out
    assert "REG01  VAL    MacBook Air" in out.out
    assert "분할을 바꿀 수 없음" in out.err
    assert "sk" not in out.out + out.err.replace("REG", "")


def test_명령줄_설정이_빠지면_이름만_알리고_2로_끝난다(capsys):
    w = World()
    w.labeled("REG01", 1)
    deps = Deps({}, registry=lambda: w.registry, store=lambda: w.store, commit=lambda: CLEAN)
    assert main(["dataset", "build"], deps=deps) == 2
    assert "MINIO_ENDPOINT" in capsys.readouterr().err


def test_명령줄_build는_val_기기가_없으면_아무것도_쓰지_않고_멈춘다(capsys):
    w = World()
    w.labeled("REG01", 2)
    deps = Deps({"MINIO_ENDPOINT": "http://minio.test", "MINIO_ACCESS_KEY": "ak", "MINIO_SECRET_KEY": "sk"},
                registry=lambda: w.registry, store=lambda: w.store, commit=lambda: CLEAN)
    assert main(["dataset", "device", "add", "REG01", "--split", "TRAIN", "--model", "MacBook Air"], deps=deps) == 0
    assert main(["dataset", "build"], deps=deps) == 1
    assert "val 사진이 없음" in capsys.readouterr().err
    assert w.s3.keys("trace-datasets") == [] and w.registry.versions() == []
    assert main(["dataset", "build", "--allow-no-val"], deps=deps) == 0
