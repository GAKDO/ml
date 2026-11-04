"""Ultralytics YOLO11-seg 학습·검증·ONNX 내보내기. 무거운 외부 라이브러리는 이 함수 하나로 감싼다."""

from __future__ import annotations

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
