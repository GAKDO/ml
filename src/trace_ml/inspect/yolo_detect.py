"""폰 사진 흠집 탐지(FR-INS-01). 부스 Jetson과 같은 학습 모델(trace-models의 best.pt·best.onnx)을 쓴다.

11/2 민서 결정: 폰 탐지는 A가 맡고, 부스와 같은 모델·같은 타일링(640 px, 20% 겹침)·같은 중복 제거를 쓴다.
Jetson은 같은 가중치로 만든 TensorRT 엔진을 쓰므로, 가중치 옆 model.json(modelVersion·classes)도 같은 파일을 둔다.
추론 한 번(타일 → 원본 다각형)만 바꿔 끼울 수 있게 두어 GPU 없는 CI에서도 테스트한다.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..codes import DEFECT_CLASSES
from ..labelstudio.config import ConfigError
from .detect import Detector, NullDetector, PhotoDetection

log = logging.getLogger(__name__)

TILE = 640
OVERLAP = 0.2
DEDUPE = 0.5


@dataclass(frozen=True)
class RawDetection:
    class_index: int
    confidence: float
    polygon: list[tuple[float, float]]  # 타일 px


Infer = Callable[[np.ndarray], list[RawDetection]]


@dataclass(frozen=True)
class ModelInfo:
    model_version: str
    classes: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> ModelInfo:
        """가중치 옆 model.json(부스와 같은 형식). modelVersion이 없거나 클래스가 codes.yaml과 다르면 시작하지 않는다."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"모델 정보를 읽지 못함: {path}") from exc
        version = str(data.get("modelVersion") or "").strip()
        if not version:
            raise ConfigError("model.json에 modelVersion이 없음")
        classes = tuple(data.get("classes") or ())
        if sorted(classes) != sorted(DEFECT_CLASSES):
            raise ConfigError(f"모델 클래스 {list(classes)}가 codes.yaml {list(DEFECT_CLASSES)}와 다름")
        return cls(version, classes)


def _starts(length: int, size: int, step: int) -> list[int]:
    if length <= size:
        return [0]
    starts = list(range(0, length - size, step))
    starts.append(length - size)  # 마지막 타일은 끝에 붙인다
    return sorted(set(starts))


def tiles(width: int, height: int, size: int = TILE, overlap: float = OVERLAP) -> list[tuple[int, int, int, int]]:
    """사진 전체를 빠짐없이 덮는 (x, y, w, h) 타일. 부스 detect/tiling.py와 같은 규칙."""
    step = max(1, int(size * (1 - overlap)))
    return [(x, y, min(size, width), min(size, height))
            for y in _starts(height, size, step) for x in _starts(width, size, step)]


def _box(polygon: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _overlap(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """IoU와 작은 상자 대비 겹침 중 큰 값. 타일 끝에서 잘린 조각도 같은 흠집으로 본다(부스 merge.py와 같음)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    smaller = min(area_a, area_b)
    return max(inter / union if union > 0 else 0.0, inter / smaller if smaller > 0 else 0.0)


def dedupe(found: list[PhotoDetection], threshold: float = DEDUPE) -> list[PhotoDetection]:
    kept: list[PhotoDetection] = []
    for f in sorted(found, key=lambda d: d.confidence, reverse=True):
        if all(k.cls != f.cls or _overlap(_box(k.polygon), _box(f.polygon)) < threshold for k in kept):
            kept.append(f)
    return kept


class TiledDetector:
    """원본 사진을 타일로 잘라 추론하고 원본 px 다각형으로 모은다. Detector 프로토콜을 따른다."""

    def __init__(self, infer: Infer, info: ModelInfo) -> None:
        self._infer = infer
        self._info = info
        self.model_version = info.model_version

    def detect(self, image: np.ndarray) -> list[PhotoDetection]:
        height, width = image.shape[:2]
        found: list[PhotoDetection] = []
        for x, y, w, h in tiles(width, height):
            for raw in self._infer(image[y:y + h, x:x + w]):
                if not 0 <= raw.class_index < len(self._info.classes) or len(raw.polygon) < 3:
                    continue
                polygon = [[min(max(float(px) + x, 0.0), float(width)), min(max(float(py) + y, 0.0), float(height))]
                           for px, py in raw.polygon]
                found.append(PhotoDetection(self._info.classes[raw.class_index], polygon,
                                            round(float(raw.confidence), 4)))
        return dedupe(found)


def ultralytics_infer(weights: Path, min_confidence: float = 0.25) -> Infer:
    """best.pt 또는 best.onnx를 ultralytics로 부른다(집 PC `.[train]`). GPU가 있으면 쓴다."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ConfigError("ultralytics가 설치되지 않음(pip install -e '.[train]')") from exc
    model = YOLO(str(weights), task="segment")

    def infer(tile: np.ndarray) -> list[RawDetection]:
        result: Any = model.predict(tile, imgsz=TILE, conf=min_confidence, verbose=False)[0]
        if result.masks is None:
            return []
        return [RawDetection(int(c), float(p), [(float(px), float(py)) for px, py in poly])
                for c, p, poly in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.masks.xy)
                if len(poly) >= 3]

    return infer


def default_infer(weights: Path, env: Mapping[str, str] | None = None) -> Infer:
    """.onnx면 onnxruntime CPU(학교 서버 컨테이너, torch 없음), 그 밖(.pt)은 ultralytics(집 PC GPU)."""
    env = env or {}
    conf = _float(env, "TRACE_DETECT_CONF", 0.25)
    if weights.suffix.lower() == ".onnx":
        from .onnx_seg import onnx_infer

        threads = env.get("TRACE_DETECT_THREADS") or "0"
        if not threads.isdigit():
            raise ConfigError("TRACE_DETECT_THREADS는 0 이상의 정수여야 함")
        return onnx_infer(weights, min_confidence=conf, threads=int(threads))
    return ultralytics_infer(weights, min_confidence=conf)


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}는 숫자여야 함: {raw}") from exc
    if not 0 < value < 1:
        raise ConfigError(f"{name}는 0과 1 사이여야 함: {raw}")
    return value


def find_weights(env: Mapping[str, str]) -> Path | None:
    """가중치 경로. TRACE_DETECT_WEIGHTS(파일)가 먼저, 없으면 TRACE_MODEL_DIR 폴더의 best.onnx → best.pt.

    TRACE_MODEL_DIR은 컨테이너에 붙인 모델 폴더다. 모델을 바꿀 때 이 폴더의 best.onnx·model.json만 바꾸고
    워커를 다시 시작한다. 폴더가 비어 있으면 탐지만 건너뛰고 정합·텍스처는 계속한다(시작을 막지 않음).
    """
    raw = env.get("TRACE_DETECT_WEIGHTS")
    if raw:
        weights = Path(raw).expanduser()
        if not weights.is_file():
            raise ConfigError(f"탐지 가중치 파일이 없음: {weights}")
        return weights
    folder = env.get("TRACE_MODEL_DIR")
    if not folder:
        return None
    # 컨테이너(onnxruntime만 설치)에서는 best.pt를 고르면 시작이 멈추므로 ultralytics가 있을 때만 .pt를 본다
    names = ("best.onnx", "best.pt") if _has_ultralytics() else ("best.onnx",)
    for name in names:
        candidate = Path(folder).expanduser() / name
        if candidate.is_file():
            return candidate
    log.warning("모델 폴더 %s에 %s가 없어 폰 탐지를 건너뜀(detector-pending)", folder, "·".join(names))
    return None


def _has_ultralytics() -> bool:
    return importlib.util.find_spec("ultralytics") is not None


def load_detector(env: Mapping[str, str],
                  make_infer: Callable[[Path], Infer] | None = None) -> Detector:
    """가중치(옆에 model.json)가 있으면 그 모델, 없으면 NullDetector."""
    weights = find_weights(env)
    if weights is None:
        if not env.get("TRACE_MODEL_DIR"):
            log.warning("TRACE_DETECT_WEIGHTS·TRACE_MODEL_DIR가 없어 폰 탐지를 건너뜀(detector-pending)")
        return NullDetector()
    info = ModelInfo.load(weights.parent / "model.json")
    log.info("폰 탐지 모델 %s (%s)", info.model_version, weights.name)
    infer = make_infer(weights) if make_infer else default_infer(weights, env)
    return TiledDetector(infer, info)
