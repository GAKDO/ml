"""train: 버전을 받아 학습하고, 결과를 trace-models와 ml.training_run에 데이터셋 버전과 함께 남긴다."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..dataset.review_export import ReviewExportError, held_out_names, read_summary
from ..labelstudio.config import ConfigError
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


def _imported(run_dir: Path | None) -> dict[str, Any] | None:
    if run_dir is None:
        return None
    args = run_dir / "args.yaml"
    return {"path": str(run_dir), "argsYaml": args.read_text(encoding="utf-8") if args.is_file() else None}


def train(store: Store, registry: Registry, settings: MinioSettings, version: str, *, base_weights: str,
          epochs: int, imgsz: int, cache_dir: Path, trainer: Callable[[TrainSpec], TrainOutput] = train_yolo,
          now: Callable[[], datetime] = lambda: datetime.now(UTC), tile: int = 0, overlap: float = 0.2,
          background_ratio: float = 1.0, batch: int = -1, extra: dict[str, Any] | None = None,
          model_name: str | None = None, extra_train: Path | None = None,
          imported_from: Path | None = None) -> RunResult:
    """tile > 0이면 부스(BO2)처럼 원본을 tile px 타일로 잘라 학습하고, imgsz·ONNX도 tile 크기로 맞춘다.

    extra_train은 trace-ml review export 폴더(사람 검수 결과). train에만 더하고 run.json에 요약 해시를 남긴다.
    imported_from은 따로 학습한 폴더(trainer=existing_run). 그 폴더의 args.yaml을 run.json에 남기고,
    올린 best.onnx·model.json을 폴더의 registered/에도 써서 USB로 바로 옮길 수 있게 한다.
    """
    if extra_train is not None and not tile:
        raise ConfigError("--extra-train은 타일 학습(--tile 640)에서만 씀")
    extra_doc = None
    if extra_train is not None:
        try:
            summary, summary_sha = read_summary(extra_train)
        except ReviewExportError as exc:
            raise ConfigError(str(exc)) from exc
        extra_doc = {"path": str(extra_train), "summarySha256": summary_sha, "photos": summary.get("photos"),
                     "confirmed": summary.get("confirmed"), "dismissed": summary.get("dismissed")}
    found = [v for v in registry.versions() if v.version == version]
    if not found:
        raise RegistryError(f"데이터셋 버전 {version}이 없음: trace-ml dataset list로 확인")
    ds = found[0]
    fetched = fetch(store, settings, ds, cache_dir / "datasets")
    data_yaml, tiling = fetched.data_yaml, None
    has_test = bool(fetched.manifest["counts"]["images"].get("test"))
    excluded: frozenset[str] = frozenset()
    if extra_train is not None:
        if summary.get("classes") != list(fetched.manifest["classes"]):
            # 클래스 번호가 다르면 라벨이 다른 흠집으로 읽힌다
            raise ConfigError(f"검수 내보내기 클래스 {summary.get('classes')}가 {version}과 다름")
        # VAL·TEST 기기 사진이 train에 들어가면 그 기기로 잰 점수가 부풀려진다. 시리얼·자산번호가 기기 코드와 같으면 뺀다
        held = {code.upper() for code, split in (fetched.manifest.get("devices") or {}).items()
                if split in ("VAL", "TEST")}
        try:
            excluded = frozenset(held_out_names(extra_train, held))
        except ReviewExportError as exc:
            raise ConfigError(str(exc)) from exc
        extra_doc["excludedHeldOut"] = sorted(excluded)
    if tile:
        names = dict(enumerate(fetched.manifest["classes"]))
        # 검수분을 섞으면 같은 버전의 타일 폴더와 구분한다(섞지 않은 학습이 이 폴더를 다시 쓰지 않게)
        suffix = f"-rv{extra_doc['summarySha256'][:8]}" if extra_doc else ""
        tiled = tile_dataset(fetched.data_yaml.parent, cache_dir / "datasets" / f"{version}-tile{tile}{suffix}", names,
                             size=tile, overlap=overlap, background_ratio=background_ratio, extra_train=extra_train,
                             extra_exclude=excluded)
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
        "baseWeights": base_weights, "modelName": model_name, "startedAt": started.isoformat().replace("+00:00", "Z"),
        "finishedAt": now().replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "imgsz": imgsz, "tiling": tiling, "trainArgs": dict(extra or {}), "extraTrain": extra_doc,
        "metrics": out.metrics, "importedFrom": _imported(imported_from),
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
    if imported_from is not None:
        local = imported_from / "registered"
        local.mkdir(exist_ok=True)
        for name, body, _ in files:
            if name in ("best.onnx", "model.json"):
                (local / name).write_bytes(body)
    run = TrainingRun(version, base_weights, out.metrics, prefix + "best.pt", prefix + "best.onnx")
    registry.add_run(run)
    return RunResult(run_name, prefix, run)
