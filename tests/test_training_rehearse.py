"""이 파일이 확인하는 것: 실제 사진 없이 학습 과정 전체를 미리 돌려 보는 "리허설".
합성 사진 → 640px 조각(타일)으로 자르기 → 학습(가짜) → 부스용 모델 파일(ONNX·model.json)까지 이어지는지 본다.
진짜 GPU 학습은 하지 않으므로 CI에서도 돈다.

사진 없이 학습 파이프라인 리허설: 합성 사진 → 640 타일 → 학습(가짜) → ONNX·model.json.
"""

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


# 테스트 도구: 실제 YOLO 학습 대신 가짜 모델 파일만 만들고, 어떤 설정으로 불렸는지 기록한다.
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


# [상황] 흠집을 그린 합성 사진과, 흠집 없는 깨끗한 합성 사진을 만든다.
# [기대] 흠집 사진의 라벨은 흠집(scratch)·찌그러짐(dent) 종류이고, 좌표가 사진 안(0~1)에 있다.
#        깨끗한 사진은 라벨이 비어 있다.
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


# [상황] 사진 10·12·40장으로 합성 데이터셋을 만든다.
# [기대] 학습·검증·시험 칸에 모두 사진이 들어가고(10장이면 7·2·1), 검증 칸에는
#        흠집 사진과 깨끗한 사진이 둘 다 있다.
def test_합성_데이터셋은_train_val_test_모두에_사진이_있다(tmp_path: Path) -> None:
    for photos in (10, 12, 40):
        counts = write_raw_dataset(tmp_path / str(photos), photos, 640, 360)
        assert sum(counts.values()) == photos and counts["val"] >= 2 and counts["test"] >= 1
    assert write_raw_dataset(tmp_path / "t", 10, 640, 360) == {"train": 7, "val": 2, "test": 1}
    # val에는 흠집 사진과 깨끗한 사진이 모두 있다(부스 bench 샘플 두 장)
    vals = [p.read_text(encoding="utf-8").strip() for p in (tmp_path / "t" / "labels" / "val").glob("*.txt")]
    assert any(vals) and not all(vals)


# [상황] 합성 사진 10장으로 리허설을 끝까지 돌린다.
# [기대] 실제 학습과 같은 640px 조각으로 학습하고 모델 파일을 검사한다.
#        부스용 model.json에는 "시험 전용(testOnly)" 표시가 붙는다(부스가 이 결과를 서버에 올리지 않게).
#        흠집·깨끗한 예시 사진도 남는다.
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


# [상황] 저장소·DB 설정 없이 "trace-ml rehearse" 명령을 실행한다.
# [기대] 정상 종료하고 모델 검사 "통과"를 출력한다(집 PC에서 바로 돌려 볼 수 있음).
def test_리허설_명령은_MinIO_DB_환경_변수_없이_돈다(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                         capsys: pytest.CaptureFixture[str]) -> None:
    import trace_ml.training.rehearse as mod

    real = mod.rehearse
    monkeypatch.setattr(mod, "rehearse", lambda out, **kw: real(out, width=1280, height=720, trainer=FakeTrainer(),
                                                               onnx_checker=lambda *a: "통과", **kw))
    code = main(["rehearse", "--out", str(tmp_path / "o"), "--photos", "10", "--epochs", "1"], env={})
    assert code == 0
    assert "통과" in capsys.readouterr().out


# [상황] 사진 수를 너무 적게(9장) 준다.
# [기대] 설정 오류로 학습을 시작하지 않는다.
def test_리허설_설정이_잘못되면_학습하지_않는다(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        rehearse(tmp_path, photos=9, trainer=FakeTrainer())


# [상황] 진짜 모델 파일이 들어 있는 다른 폴더를 리허설 출력 폴더로 준다.
# [기대] 거절하고 그 폴더의 파일을 지우지 않는다.
def test_리허설은_자기가_만들지_않은_폴더를_지우지_않는다(tmp_path: Path) -> None:
    other = tmp_path / "cache"
    (other / "model").mkdir(parents=True)
    (other / "model" / "best.pt").write_bytes(b"real")
    with pytest.raises(ValueError):
        rehearse(other, photos=10, width=640, height=360, trainer=FakeTrainer(), onnx_checker=lambda *a: "통과")
    assert (other / "model" / "best.pt").read_bytes() == b"real"


# [상황] 같은 출력 폴더로 리허설을 두 번 돌린다.
# [기대] 앞 결과가 남아 있어도 막히지 않고 새로 만든다.
def test_리허설_폴더는_다시_돌리면_새로_만든다(tmp_path: Path) -> None:
    out = tmp_path / "r"
    kw: dict = {"photos": 10, "width": 640, "height": 360, "onnx_checker": lambda *a: "통과"}
    rehearse(out, trainer=FakeTrainer(), **kw)
    rehearse(out, trainer=FakeTrainer(), **kw)  # runs/rehearsal이 남아 있어도 막히지 않는다
    assert (out / "model" / "model.json").exists()
