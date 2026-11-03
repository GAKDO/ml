"""부스(BO2)와 같은 640 타일로 잘라 학습하고, Jetson이 쓸 model.json을 함께 남긴다."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import cv2
import numpy as np
import pytest
import yaml
from fakes import SETTINGS, World, ann, polygon
from test_training import FakeTrainer

from trace_ml.dataset.builder import Commit, build
from trace_ml.registry import Device, RegistryError
from trace_ml.training.record import train
from trace_ml.training.tiles import area, clip, tile_label, tiles


def _jpeg(width: int = 1920, height: int = 1080) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((height, width, 3), 128, np.uint8))
    assert ok
    return buf.tobytes()


def _world(scratch: list[tuple[float, float]]) -> World:
    """REG01(TRAIN)·REG02(VAL)·REG03(TEST) 1080p 사진. REG01은 흠집 하나, 나머지는 흠집 없음."""
    w = World()
    for code, split in (("REG01", "TRAIN"), ("REG02", "VAL"), ("REG03", "TEST")):
        w.registry.save_device(Device(code, "MacBook Air", split))
    w.photo("REG01/REG01_TOP_L1_CAM1_01.jpg", ann(1, polygon("scratch", scratch)))
    w.photo("REG01/REG01_BOT_L1_CAM1_01.jpg", ann(2, {"type": "choices", "value": {"choices": ["no_defect"]}}))
    w.photo("REG02/REG02_TOP_L1_CAM1_01.jpg", ann(3, polygon("dent", [(40, 40), (45, 40), (45, 45)])))
    w.photo("REG03/REG03_TOP_L1_CAM1_01.jpg", ann(4, {"type": "choices", "value": {"choices": ["no_defect"]}}))
    for (bucket, key), (_, meta) in list(w.s3.objects.items()):
        if bucket == "trace-raw":
            w.s3.objects[(bucket, key)] = (_jpeg(), meta)
    build(w.store, w.registry, SETTINGS, Commit("c" * 40, False), now=datetime(2026, 11, 5, tzinfo=UTC))
    return w


def _train(w: World, tmp, trainer: FakeTrainer, **kw):
    clock = iter([datetime(2026, 11, 6, 1, 0, tzinfo=UTC), datetime(2026, 11, 6, 3, 0, tzinfo=UTC)])
    return train(w.store, w.registry, SETTINGS, "v001", base_weights="yolo11s-seg.pt", epochs=1, imgsz=1280,
                 cache_dir=tmp, trainer=trainer, now=lambda: next(clock), tile=640, **kw)


def test_타일은_부스와_같은_규칙으로_1080p를_빠짐없이_덮는다():
    got = tiles(1920, 1080)
    assert sorted({t.x for t in got}) == [0, 512, 1024, 1280] and sorted({t.y for t in got}) == [0, 440]
    assert all((t.w, t.h) == (640, 640) for t in got)


def test_폴리곤은_타일_경계에서_잘리고_타일_기준_0_1로_바뀐다():
    square = [(600.0, 100.0), (700.0, 100.0), (700.0, 200.0), (600.0, 200.0)]
    assert area(clip(square, 0, 0, 640, 640)) == pytest.approx(40 * 100)
    lines = tile_label([(0, [(x / 1920, y / 1080) for x, y in square])], 1920, 1080, tiles(1920, 1080)[0], 16)
    vals = [float(v) for v in lines[0].split()[1:]]
    assert lines[0].startswith("0 ") and max(vals[0::2]) == pytest.approx(1.0) and min(vals) >= 0


def test_타일_안에_조금만_걸친_흠집은_버린다():
    sliver = [(639.0, 100.0), (700.0, 100.0), (700.0, 110.0), (639.0, 110.0)]  # 타일 0 안에 1×10px
    assert tile_label([(0, [(x / 1920, y / 1080) for x, y in sliver])], 1920, 1080, tiles(1920, 1080)[0], 16) == []


def test_타일로_학습하면_imgsz는_640이고_train의_빈_타일은_흠집_타일_수만큼만_둔다(tmp_path):
    w = _world([(10, 10), (20, 10), (20, 20), (10, 20)])  # 원본 % 좌표: (192,108)~(384,216) → 타일 1개
    trainer = FakeTrainer()
    _train(w, tmp_path, trainer)
    spec = trainer.specs[0]
    root = yaml.safe_load(spec.data_yaml.read_text(encoding="utf-8"))["path"]
    labels = sorted((tmp_path / "datasets" / "v001-tile640" / "labels" / "train").glob("*.txt"))
    assert spec.imgsz == 640 and root.endswith("v001-tile640")
    assert sum(1 for p in labels if p.read_text(encoding="utf-8")) == 1 and len(labels) == 2
    assert len(list((tmp_path / "datasets" / "v001-tile640" / "images" / "test").glob("*.jpg"))) == 8


def test_model_json은_부스가_읽는_modelVersion과_codes_yaml_클래스를_담는다(tmp_path):
    w = _world([(10, 10), (20, 10), (20, 20), (10, 20)])
    result = _train(w, tmp_path, FakeTrainer())
    info = json.loads(w.s3.objects[("trace-models", result.prefix + "model.json")][0])
    assert info["modelVersion"] == "yolo11s-seg-v001-trt-fp16"
    assert info["classes"] == ["scratch", "dent", "crack", "stain"]
    assert (info["imgsz"], info["tileSize"], info["overlap"]) == (640, 640, 0.2)
    run = json.loads(w.s3.objects[("trace-models", result.prefix + "run.json")][0])
    assert run["tiling"]["counts"]["train"]["defect"] == 1


def test_val_사진이_없으면_학습하지_않는다(tmp_path):
    w = World()
    w.registry.save_device(Device("REG01", "MacBook Air", "TRAIN"))
    w.photo("REG01/REG01_TOP_L1_CAM1_01.jpg", ann(1, polygon("scratch", [(10, 10), (20, 10), (20, 20)])))
    w.s3.objects[("trace-raw", "REG01/REG01_TOP_L1_CAM1_01.jpg")] = (_jpeg(), {})
    build(w.store, w.registry, SETTINGS, Commit("c" * 40, False), now=datetime(2026, 11, 5, tzinfo=UTC))
    trainer = FakeTrainer()
    with pytest.raises(RegistryError, match="val"):
        _train(w, tmp_path, trainer)
    assert trainer.specs == []


def test_타일_크기나_빈_타일_비율이_음수면_자르지_않는다(tmp_path, capsys):
    with pytest.raises(ValueError):
        tiles(1920, 1080, size=0)
    w = _world([(10, 10), (20, 10), (20, 20), (10, 20)])
    with pytest.raises(ValueError):
        _train(w, tmp_path, FakeTrainer(), background_ratio=-1)
    from trace_ml.__main__ import main
    from trace_ml.dataset.cli import Deps

    deps = Deps({}, registry=lambda: w.registry, store=lambda: w.store, trainer=FakeTrainer())
    assert main(["train", "v001", "--tile", "-1"], deps=deps) == 2
