"""사진 없이 학습 파이프라인 리허설: 합성 사진 → 640 타일 → 학습(가짜) → ONNX·model.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trace_ml.__main__ import main
from trace_ml.codes import DEFECT_CLASSES
from trace_ml.training.rehearse import rehearse, synth_photo, write_raw_dataset
from trace_ml.training.yolo import TrainOutput, TrainSpec

METRICS = {"val": {"mask": {"map50": 0.5, "map": 0.3}}}


class FakeTrainer:
    def __init__(self) -> None:
        self.specs: list[TrainSpec] = []

    def __call__(self, spec: TrainSpec) -> TrainOutput:
        self.specs.append(spec)
        out = spec.workdir / spec.name
        out.mkdir(parents=True)
        (out / "best.pt").write_bytes(b"weights")
        (out / "best.onnx").write_bytes(b"onnx")
        return TrainOutput(out / "best.pt", out / "best.onnx", METRICS)


def test_합성_사진의_라벨_폴리곤은_그린_흠집_위에_있고_흠집_없는_사진은_빈_라벨이다() -> None:
    image, lines = synth_photo(3, 1280, 720)
    assert image.shape == (720, 1280, 3)
    assert lines
    for line in lines:
        cls, *nums = line.split()
        assert int(cls) in (DEFECT_CLASSES.index("scratch"), DEFECT_CLASSES.index("dent"))
        xy = np.array(nums, dtype=float).reshape(-1, 2)
        assert len(xy) >= 3 and ((xy >= 0) & (xy <= 1)).all()
    clean, clean_lines = synth_photo(3, 1280, 720, clean=True)
    assert clean_lines == []
    assert not np.array_equal(clean, image)


def test_합성_데이터셋은_train_val_test_모두에_사진이_있다(tmp_path: Path) -> None:
    counts = write_raw_dataset(tmp_path, 10, 1280, 720)
    assert counts == {"train": 7, "val": 2, "test": 1}
    for split in counts:
        assert len(list((tmp_path / "images" / split).glob("*.jpg"))) == counts[split]


def test_리허설은_실제_학습과_같은_640_타일로_학습하고_부스용_testOnly_model_json을_남긴다(tmp_path: Path) -> None:
    trainer = FakeTrainer()
    checked = []
    result = rehearse(tmp_path / "out", photos=10, epochs=2, base_weights="yolo11n-seg.pt", width=1280, height=720,
                      trainer=trainer, onnx_checker=lambda p, s, n: checked.append((s, n)) or "통과")
    spec = trainer.specs[0]
    assert spec.imgsz == 640 and spec.epochs == 2 and spec.has_test
    assert "tile640" in str(spec.data_yaml)
    assert checked == [(640, 4)]
    assert result.counts["train"]["defect"] > 0 and result.counts["val"]["tiles"] > 0
    info = json.loads(result.model_json.read_text(encoding="utf-8"))
    assert info["testOnly"] is True  # 부스가 이 모델 결과를 서버에 올리지 않게
    assert info["classes"] == list(DEFECT_CLASSES)
    assert info["modelVersion"] == "yolo11n-seg-rehearsal-trt-fp16"
    assert result.onnx.read_bytes() == b"onnx"
    assert (result.out_dir / "sample_defect.jpg").exists() and (result.out_dir / "sample_clean.jpg").exists()


def test_리허설_명령은_MinIO_DB_환경_변수_없이_돈다(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                         capsys: pytest.CaptureFixture[str]) -> None:
    import trace_ml.training.rehearse as mod

    real = mod.rehearse
    monkeypatch.setattr(mod, "rehearse", lambda out, **kw: real(out, width=1280, height=720, trainer=FakeTrainer(),
                                                               onnx_checker=lambda *a: "통과", **kw))
    code = main(["rehearse", "--out", str(tmp_path / "o"), "--photos", "10", "--epochs", "1"], env={})
    assert code == 0
    assert "통과" in capsys.readouterr().out


def test_리허설_설정이_잘못되면_학습하지_않는다(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        rehearse(tmp_path, photos=2, trainer=FakeTrainer())
