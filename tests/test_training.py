"""ML1 스토리 2: 학습 결과를 데이터셋 버전과 함께 저장한다 (FR-DAT-05)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from fakes import SETTINGS, World

from trace_ml.__main__ import main
from trace_ml.dataset.builder import Commit, build
from trace_ml.dataset.cli import Deps
from trace_ml.registry import Device, RegistryError
from trace_ml.training.fetch import FetchError
from trace_ml.training.record import train
from trace_ml.training.yolo import TrainOutput, TrainSpec

METRICS = {"epochs": 1, "imgsz": 640, "val": {"box": {"map50": 0.5, "map": 0.3}, "mask": {"map50": 0.4, "map": 0.2}},
           "test": {"box": {"map50": 0.45, "map": 0.25}, "mask": {"map50": 0.35, "map": 0.15}}}


class FakeTrainer:
    """Ultralytics 대신: 받은 폴더를 확인하고 가짜 가중치·ONNX를 만든다."""

    def __init__(self) -> None:
        self.specs: list[TrainSpec] = []

    def __call__(self, spec: TrainSpec) -> TrainOutput:
        self.specs.append(spec)
        out = spec.workdir / spec.name
        out.mkdir(parents=True)
        (out / "best.pt").write_bytes(b"weights")
        (out / "best.onnx").write_bytes(b"onnx")
        return TrainOutput(out / "best.pt", out / "best.onnx", METRICS)


def built() -> World:
    w = World()
    w.registry.save_device(Device("REG03", "MacBook Air", "TEST"))
    for code in ("REG01", "REG02", "REG03"):
        w.labeled(code, 2)
    build(w.store, w.registry, SETTINGS, Commit("c" * 40, False), now=datetime(2026, 11, 5, tzinfo=UTC))
    return w


def _train(w: World, tmp: Path, trainer=None, version: str = "v001"):
    clock = iter([datetime(2026, 11, 6, 1, 0, tzinfo=UTC), datetime(2026, 11, 6, 3, 0, tzinfo=UTC)])
    return train(w.store, w.registry, SETTINGS, version, base_weights="yolo11s-seg.pt", epochs=1, imgsz=640,
                 cache_dir=tmp, trainer=trainer or FakeTrainer(), now=lambda: next(clock))


def test_v001로_학습하면_trace_models에_가중치_ONNX_지표_run_json이_있고_버전과_manifest_해시가_적힌다(tmp_path):
    w = built()
    result = _train(w, tmp_path)
    assert result.prefix == "v001/20261106T010000Z/"
    names = {k.rsplit("/", 1)[-1] for k in w.s3.keys("trace-models")}
    assert names == {"best.pt", "best.onnx", "metrics.json", "run.json", "model.json"}
    run = json.loads(w.s3.objects[("trace-models", result.prefix + "run.json")][0])
    manifest_sha = w.registry.versions()[0].counts["manifestSha256"]
    assert (run["datasetVersion"], run["manifestSha256"], run["builderCommit"]) == ("v001", manifest_sha, "c" * 40)
    meta = w.s3.objects[("trace-models", result.prefix + "best.onnx")][1]
    assert meta == {"trace-dataset-version": "v001", "trace-manifest-sha256": manifest_sha}
    # 재학습 파이프라인(retrain promote --model-dir)이 쓸 수 있게 실행 폴더에도 같은 ONNX·model.json을 둔다
    local = tmp_path / "runs" / "20261106T010000Z" / "registered"
    assert (local / "best.onnx").read_bytes() == b"onnx"
    assert (local / "model.json").read_bytes() == w.s3.objects[("trace-models", result.prefix + "model.json")][0]


def test_받은_폴더는_YOLO_분할_구조이고_data_yaml이_그_폴더를_가리킨다(tmp_path):
    w = built()
    trainer = FakeTrainer()
    _train(w, tmp_path, trainer)
    spec = trainer.specs[0]
    data = yaml.safe_load(spec.data_yaml.read_text(encoding="utf-8"))
    root = Path(data["path"])
    assert len(list((root / "images" / "train").glob("*.jpg"))) == len(list((root / "labels" / "train").glob("*.txt")))
    assert list((root / "images" / "test").glob("REG03_*.jpg"))
    assert spec.has_test and spec.base_weights == "yolo11s-seg.pt"


def test_학습이_끝나면_models_list에서_모델마다_데이터셋_버전이_보인다(tmp_path, capsys):
    w = built()
    deps = Deps({"MINIO_ENDPOINT": "http://minio.test", "MINIO_ACCESS_KEY": "ak", "MINIO_SECRET_KEY": "sk"},
                registry=lambda: w.registry, store=lambda: w.store, trainer=FakeTrainer())
    assert main(["train", "v001", "--epochs", "1", "--tile", "0", "--cache", str(tmp_path)], deps=deps) == 0
    assert main(["models", "list"], deps=deps) == 0
    out = capsys.readouterr().out
    assert "v001  v001/" in out and "best.pt" in out and "mask mAP50 0.35" in out
    assert w.registry.runs()[0].dataset_version == "v001"


def test_trace_raw_사진_하나가_바뀌면_학습_전에_멈추고_바뀐_키를_알린다(tmp_path):
    w = built()
    w.s3.objects[("trace-raw", "REG01/REG01_TOP_L1_CAM1_01.jpg")] = (b"changed", {})
    trainer = FakeTrainer()
    with pytest.raises(FetchError) as exc:
        _train(w, tmp_path, trainer)
    assert any("REG01/REG01_TOP_L1_CAM1_01.jpg" in p for p in exc.value.problems)
    assert trainer.specs == [] and w.s3.keys("trace-models") == []


def test_manifest가_기록과_다르면_멈춘다(tmp_path):
    w = built()
    w.s3.objects[("trace-datasets", "v001/manifest.json")] = (b'{"items": []}', {})
    with pytest.raises(FetchError):
        _train(w, tmp_path)


def test_없는_버전으로_학습하면_실패한다(tmp_path, capsys):
    w = built()
    with pytest.raises(RegistryError):
        _train(w, tmp_path, version="v009")
    deps = Deps({"MINIO_ENDPOINT": "x", "MINIO_ACCESS_KEY": "a", "MINIO_SECRET_KEY": "s"},
                registry=lambda: w.registry, store=lambda: w.store, trainer=FakeTrainer())
    assert main(["train", "v009", "--cache", str(tmp_path)], deps=deps) == 1


def test_이미_받은_파일은_해시가_같으면_다시_받지_않는다(tmp_path):
    w = built()
    _train(w, tmp_path)
    gets = []
    original = w.s3.get_object
    w.s3.get_object = lambda **kw: gets.append(kw["Key"]) or original(**kw)
    clock = iter([datetime(2026, 11, 7, tzinfo=UTC)] * 2)
    train(w.store, w.registry, SETTINGS, "v001", base_weights="yolo11s-seg.pt", epochs=1, imgsz=640,
          cache_dir=tmp_path, trainer=FakeTrainer(), now=lambda: next(clock))
    assert gets == ["v001/manifest.json"]
    assert len(w.registry.runs()) == 2


def test_따로_학습한_폴더를_기록하면_그_가중치가_올라가고_지표는_이_버전으로_다시_잰다(tmp_path):
    from trace_ml.training.yolo import existing_run

    w = built()
    run_dir = tmp_path / "runs-offline" / "v001-aug"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best.pt").write_bytes(b"offline-weights")
    (run_dir / "weights" / "best.onnx").write_bytes(b"offline-onnx")
    (run_dir / "args.yaml").write_text("degrees: 15\nmixup: 0.15\n", encoding="utf-8")
    seen = []

    def validate(weights, spec):
        seen.append((weights, spec.data_yaml))
        return {"val": METRICS["val"]}

    clock = iter([datetime(2026, 11, 8, 10, 0, tzinfo=UTC), datetime(2026, 11, 8, 10, 5, tzinfo=UTC)])
    result = train(w.store, w.registry, SETTINGS, "v001", base_weights="pretrain.pt", epochs=40, imgsz=640,
                   cache_dir=tmp_path / "cache", trainer=existing_run(run_dir, validate=validate),
                   now=lambda: next(clock), model_name="yolo11m-aihub-aug", imported_from=run_dir)
    assert w.s3.objects[("trace-models", result.prefix + "best.pt")][0] == b"offline-weights"
    assert seen and seen[0][0] == run_dir / "weights" / "best.pt"
    run = json.loads(w.s3.objects[("trace-models", result.prefix + "run.json")][0])
    assert run["importedFrom"]["argsYaml"].startswith("degrees: 15") and run["metrics"]["epochs"] == 40
    local = json.loads((run_dir / "registered" / "model.json").read_text(encoding="utf-8"))
    assert local["modelVersion"] == "yolo11m-aihub-aug-v001-20261108T100000Z-trt-fp16"
    assert (run_dir / "registered" / "best.onnx").read_bytes() == b"offline-onnx"


def test_따로_학습한_폴더에_ONNX가_없거나_검수분과_같이_쓰면_기록하지_않는다(tmp_path, capsys):
    from trace_ml.training.yolo import existing_run

    w = built()
    run_dir = tmp_path / "run"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best.pt").write_bytes(b"w")
    with pytest.raises(RuntimeError, match="best.onnx"):
        _train(w, tmp_path / "c", existing_run(run_dir, validate=lambda *_: {}))
    assert not w.registry.runs()
    deps = Deps({"MINIO_ENDPOINT": "http://minio.test", "MINIO_ACCESS_KEY": "ak", "MINIO_SECRET_KEY": "sk"},
                registry=lambda: w.registry, store=lambda: w.store, trainer=FakeTrainer())
    assert main(["train", "v001", "--from-run", str(run_dir), "--extra-train", str(tmp_path),
                 "--cache", str(tmp_path / "c")], deps=deps) == 2
    assert main(["train", "v001", "--from-run", str(tmp_path / "none"), "--cache", str(tmp_path / "c")],
                deps=deps) == 2
    assert main(["train", "v001", "--from-run", str(run_dir), "--tile", "0", "--cache", str(tmp_path / "c")],
                deps=deps) == 2
    assert "--from-run" in capsys.readouterr().err
