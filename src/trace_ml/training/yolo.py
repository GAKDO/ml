"""Ultralytics YOLO11-seg 학습·검증·ONNX 내보내기. 무거운 외부 라이브러리는 이 함수 하나로 감싼다."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainSpec:
    data_yaml: Path
    base_weights: str
    epochs: int
    imgsz: int
    workdir: Path
    name: str
    has_test: bool
    batch: int = -1  # -1: GPU 메모리의 약 60%에 맞춰 자동(4070 Ti 12GB)
    extra: dict[str, Any] = field(default_factory=dict)  # ultralytics train 인자 더하기(patience·cos_lr 등)


@dataclass(frozen=True)
class TrainOutput:
    weights: Path
    onnx: Path
    metrics: dict[str, Any]


def _seg(result: Any) -> dict[str, Any]:
    return {"box": {"map50": float(result.box.map50), "map": float(result.box.map)},
            "mask": {"map50": float(result.seg.map50), "map": float(result.seg.map)}}


def train_yolo(spec: TrainSpec) -> TrainOutput:  # pragma: no cover - GPU·ultralytics가 있는 집 PC에서만
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('ultralytics가 없음: python -m pip install -e ".[train]"') from exc
    model = YOLO(spec.base_weights)
    model.train(data=str(spec.data_yaml), epochs=spec.epochs, imgsz=spec.imgsz,
                batch=spec.batch, project=str(spec.workdir), name=spec.name, exist_ok=False, **spec.extra)
    best = YOLO(str(model.trainer.best))
    metrics: dict[str, Any] = {"epochs": spec.epochs, "imgsz": spec.imgsz,
                               "val": _seg(best.val(data=str(spec.data_yaml), split="val", imgsz=spec.imgsz))}
    if spec.has_test:
        metrics["test"] = _seg(best.val(data=str(spec.data_yaml), split="test", imgsz=spec.imgsz))
    # Jetson TensorRT(JetPack 6)가 읽는 고정 크기 ONNX: 입력 1×3×imgsz×imgsz, opset 17. FP16 변환은 Jetson에서 한다
    onnx = Path(best.export(format="onnx", imgsz=spec.imgsz, opset=17, dynamic=False, simplify=True, half=False))
    return TrainOutput(Path(model.trainer.best), onnx, metrics)


def _validate_yolo(weights: Path, spec: TrainSpec) -> dict[str, Any]:  # pragma: no cover - ultralytics가 있는 집 PC에서만
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('ultralytics가 없음: python -m pip install -e ".[train]"') from exc
    best = YOLO(str(weights))
    metrics: dict[str, Any] = {"val": _seg(best.val(data=str(spec.data_yaml), split="val", imgsz=spec.imgsz))}
    if spec.has_test:
        metrics["test"] = _seg(best.val(data=str(spec.data_yaml), split="test", imgsz=spec.imgsz))
    return metrics


def existing_run(run_dir: Path, validate: Callable[[Path, TrainSpec], dict[str, Any]] = _validate_yolo,
                 ) -> Callable[[TrainSpec], TrainOutput]:
    """저장소가 멈췄을 때 yolo로 따로 학습한 폴더(weights/best.pt·best.onnx)를 학습 대신 쓴다.

    지표는 기록할 데이터셋 버전의 val·test로 다시 잰다. 그래서 다른 버전 데이터로 학습한 가중치도 점수는 이 버전 기준이 된다.
    """
    weights, onnx = run_dir / "weights" / "best.pt", run_dir / "weights" / "best.onnx"

    def trainer(spec: TrainSpec) -> TrainOutput:
        missing = [p.name for p in (weights, onnx) if not p.is_file()]
        if missing:
            raise RuntimeError(f"{run_dir / 'weights'}에 {', '.join(missing)}가 없음"
                               "(best.onnx는 yolo export format=onnx imgsz=640 opset=17로 먼저 만든다)")
        metrics = {"epochs": spec.epochs, "imgsz": spec.imgsz, **validate(weights, spec)}
        return TrainOutput(weights, onnx, metrics)

    return trainer
