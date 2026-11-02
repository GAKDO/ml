"""폰 사진 흠집 탐지 인터페이스(FR-INS-01). 구현은 yolo_detect.TiledDetector(부스와 같은 모델), 모델이 없으면 NullDetector.

탐지기는 원본 사진 px 다각형을 내고, 워커가 정합 호모그래피로 표준 px로 옮겨 길이·넓이(mm)를 잰다.
부스 탐지(Jetson)와 같은 모델·같은 좌표 변환을 쓰도록 이 경계를 둔다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from ..codes import DEFECT_CLASSES
from .align import to_standard
from .faces import PX_PER_MM


@dataclass(frozen=True)
class PhotoDetection:
    cls: str  # codes.yaml inspection.defect_class
    polygon: list[list[float]]  # 원본 사진 px
    confidence: float


@dataclass(frozen=True)
class StandardDefect:
    cls: str
    polygon: list[list[float]]  # 표준 px
    confidence: float
    length_mm: float | None
    area_mm2: float | None


class Detector(Protocol):
    model_version: str

    def detect(self, image: np.ndarray) -> list[PhotoDetection]: ...


class NullDetector:
    """모델이 정해지기 전 자리. 아무것도 찾지 않는다(로그에 남긴다)."""

    model_version = "detector-pending"

    def detect(self, image: np.ndarray) -> list[PhotoDetection]:
        return []


def measure(polygon_std: list[list[float]]) -> tuple[float | None, float | None]:
    """표준 px 다각형의 길이(외접 회전 사각형의 긴 변)·넓이를 mm·mm²로. 잴 수 없으면 None."""
    pts = np.asarray(polygon_std, dtype=np.float32)
    if len(pts) < 3 or not np.isfinite(pts).all():
        return None, None
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    area_px = abs(float(cv2.contourArea(pts)))
    length = round(max(w, h) / PX_PER_MM, 2)
    area = round(area_px / (PX_PER_MM * PX_PER_MM), 2)
    return length, (area if area > 0 else None)


def to_standard_defect(det: PhotoDetection, homography: np.ndarray) -> StandardDefect:
    if det.cls not in DEFECT_CLASSES:
        raise ValueError(f"모르는 흠집 클래스: {det.cls}")
    if len(det.polygon) < 3:
        raise ValueError("다각형 꼭짓점이 3개 미만")
    polygon = to_standard(homography, det.polygon)
    length, area = measure(polygon)
    confidence = min(1.0, max(0.0, float(det.confidence)))
    return StandardDefect(det.cls, polygon, confidence, length, area)
