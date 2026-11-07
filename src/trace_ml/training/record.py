"""train: 버전을 받아 학습하고, 결과를 trace-models와 ml.training_run에 데이터셋 버전과 함께 남긴다."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..registry import Registry, RegistryError, TrainingRun
from ..settings import MinioSettings
from ..store import Store
from .fetch import fetch
from .tiles import tile_dataset
from .yolo import TrainOutput, TrainSpec, train_yolo


@dataclass(frozen=True)
class RunResult:
    run_name: str
    prefix: str
    run: TrainingRun


# 부스·inspect-worker가 결과에 그대로 싣는 값(계약 최대 64자). worker가 "+contour-v1"(11자)을 붙이므로 그만큼 남긴다
_MODEL_VERSION_MAX = 64 - len("+contour-v1")


def model_version(name: str, version: str, run_name: str) -> str:
    """같은 이름·데이터셋으로 다시 학습해도 실행마다 다른 값이 되도록 실행 시각을 넣는다. 길면 이름을 줄인다."""
    tail = f"-{version}-{run_name}-trt-fp16"
    return name[:max(1, _MODEL_VERSION_MAX - len(tail))] + tail


def train(store: Store, registry: Registry, settings: MinioSettings, version: str, *, base_weights: str,
          epochs: int, imgsz: int, cache_dir: Path, trainer: Callable[[TrainSpec], TrainOutput] = train_yolo,
          now: Callable[[], datetime] = lambda: datetime.now(UTC), tile: int = 0, overlap: float = 0.2,
          background_ratio: float = 1.0, batch: int = -1, extra: dict[str, Any] | None = None,
          model_name: str | None = None) -> RunResult:
    """tile > 0이면 부스(BO2)처럼 원본을 tile px 타일로 잘라 학습하고, imgsz·ONNX도 tile 크기로 맞춘다."""
    found = [v for v in registry.versions() if v.version == version]
    if not found:
        raise RegistryError(f"데이터셋 버전 {version}이 없음: trace-ml dataset list로 확인")
    ds = found[0]
    fetched = fetch(store, settings, ds, cache_dir / "datasets")
    data_yaml, tiling = fetched.data_yaml, None
    has_test = bool(fetched.manifest["counts"]["images"].get("test"))
    if tile:
        names = dict(enumerate(fetched.manifest["classes"]))
        tiled = tile_dataset(fetched.data_yaml.parent, cache_dir / "datasets" / f"{version}-tile{tile}", names,
                             size=tile, overlap=overlap, background_ratio=background_ratio)
        data_yaml, imgsz = tiled.data_yaml, tile
        has_test = bool(tiled.counts["test"]["tiles"])
        tiling = {"size": tile, "overlap": overlap, "backgroundRatio": background_ratio, "counts": tiled.counts}
        if not tiled.counts["val"]["tiles"]:
            raise RegistryError(f"{version}에 val 사진이 없음: dataset device add <기기> --split VAL 뒤 새 버전을 만드세요")
    started = now().replace(microsecond=0)
    run_name = started.strftime("%Y%m%dT%H%M%SZ")
    out = trainer(TrainSpec(data_yaml, base_weights, epochs, imgsz, cache_dir / "runs", run_name, has_test, batch,
                              dict(extra or {})))

    prefix = f"{version}/{run_name}/"
    # 모델 파일만 따로 받아 가도 어느 데이터로 학습했는지 알 수 있게 객체 메타데이터에도 적는다
    meta = {"trace-dataset-version": version, "trace-manifest-sha256": fetched.manifest_sha256}
    run_doc = {
        "schema": "trace.training-run.v1", "datasetVersion": version, "manifestKey": ds.manifest_key,
        "manifestSha256": fetched.manifest_sha256, "builderCommit": ds.builder_commit,
        "baseWeights": base_weights, "startedAt": started.isoformat().replace("+00:00", "Z"),
        "finishedAt": now().replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "imgsz": imgsz, "tiling": tiling, "trainArgs": dict(extra or {}), "metrics": out.metrics,
        "files": {"weights": "best.pt", "onnx": "best.onnx", "modelInfo": "model.json"},
    }
    # Jetson(BO2)이 엔진 옆에 그대로 두는 파일. 엔진은 이 ONNX로 jetson2에서 FP16으로 만든다
    model_info = {"modelVersion": model_version(model_name or Path(base_weights).stem, version, run_name),
                  "classes": list(fetched.manifest["classes"]), "datasetVersion": version, "run": run_name,
                  "imgsz": imgsz, "tileSize": tile or None, "overlap": overlap if tile else None}
    files = [
        ("best.pt", out.weights.read_bytes(), "application/octet-stream"),
        ("best.onnx", out.onnx.read_bytes(), "application/octet-stream"),
        ("metrics.json", json.dumps(out.metrics, indent=2).encode(), "application/json"),
        ("run.json", json.dumps(run_doc, ensure_ascii=False, indent=2).encode(), "application/json"),
        ("model.json", json.dumps(model_info, ensure_ascii=False, indent=2).encode(), "application/json"),
    ]
    for name, body, ctype in files:
        store.put_new(settings.models_bucket, prefix + name, body, ctype, meta)
    run = TrainingRun(version, base_weights, out.metrics, prefix + "best.pt", prefix + "best.onnx")
    registry.add_run(run)
    return RunResult(run_name, prefix, run)
