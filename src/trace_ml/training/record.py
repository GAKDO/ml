"""train: 버전을 받아 학습하고, 결과를 trace-models와 ml.training_run에 데이터셋 버전과 함께 남긴다."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..registry import Registry, RegistryError, TrainingRun
from ..settings import MinioSettings
from ..store import Store
from .fetch import fetch
from .yolo import TrainOutput, TrainSpec, train_yolo


@dataclass(frozen=True)
class RunResult:
    run_name: str
    prefix: str
    run: TrainingRun


def train(store: Store, registry: Registry, settings: MinioSettings, version: str, *, base_weights: str,
          epochs: int, imgsz: int, cache_dir: Path, trainer: Callable[[TrainSpec], TrainOutput] = train_yolo,
          now: Callable[[], datetime] = lambda: datetime.now(UTC)) -> RunResult:
    found = [v for v in registry.versions() if v.version == version]
    if not found:
        raise RegistryError(f"데이터셋 버전 {version}이 없음: trace-ml dataset list로 확인")
    ds = found[0]
    fetched = fetch(store, settings, ds, cache_dir / "datasets")
    started = now().replace(microsecond=0)
    run_name = started.strftime("%Y%m%dT%H%M%SZ")
    has_test = bool(fetched.manifest["counts"]["images"].get("test"))
    out = trainer(TrainSpec(fetched.data_yaml, base_weights, epochs, imgsz, cache_dir / "runs", run_name, has_test))

    prefix = f"{version}/{run_name}/"
    # 모델 파일만 따로 받아 가도 어느 데이터로 학습했는지 알 수 있게 객체 메타데이터에도 적는다
    meta = {"trace-dataset-version": version, "trace-manifest-sha256": fetched.manifest_sha256}
    run_doc = {
        "schema": "trace.training-run.v1", "datasetVersion": version, "manifestKey": ds.manifest_key,
        "manifestSha256": fetched.manifest_sha256, "builderCommit": ds.builder_commit,
        "baseWeights": base_weights, "startedAt": started.isoformat().replace("+00:00", "Z"),
        "finishedAt": now().replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "metrics": out.metrics, "files": {"weights": "best.pt", "onnx": "best.onnx"},
    }
    files = [
        ("best.pt", out.weights.read_bytes(), "application/octet-stream"),
        ("best.onnx", out.onnx.read_bytes(), "application/octet-stream"),
        ("metrics.json", json.dumps(out.metrics, indent=2).encode(), "application/json"),
        ("run.json", json.dumps(run_doc, ensure_ascii=False, indent=2).encode(), "application/json"),
    ]
    for name, body, ctype in files:
        store.put_new(settings.models_bucket, prefix + name, body, ctype, meta)
    run = TrainingRun(version, base_weights, out.metrics, prefix + "best.pt", prefix + "best.onnx")
    registry.add_run(run)
    return RunResult(run_name, prefix, run)
