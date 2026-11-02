"""윤곽 4점 호모그래피로 사진을 표준 좌표로 펼친다(method CONTOUR)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .contour import Quad, QuadFailure, find_quad

METHOD_CONTOUR = "CONTOUR"  # codes.yaml inspection.alignment_method
CORNER_SKIP = 0.08  # 둥근 모서리: 변 길이의 8% 안쪽 꼭짓점 근처는 오차에서 뺀다


@dataclass(frozen=True)
class Alignment:
    homography: np.ndarray  # 3×3, 사진 px → 표준 px
    error_px: float | None
    rectified: np.ndarray
    corners: np.ndarray  # 사진 px 꼭짓점(왼쪽 위부터 시계 방향)
    photo_size: tuple[int, int]  # 원본 사진 (가로, 세로)
    partial: bool = False  # 부스 반쪽 사진처럼 면의 일부만 보임
    method: str = METHOD_CONTOUR

    def coverage(self) -> np.ndarray:
        """펼친 이미지에서 사진이 실제로 덮은 곳(255). 반쪽 사진을 이어 붙일 때 쓴다."""
        w, h = self.photo_size
        size = (self.rectified.shape[1], self.rectified.shape[0])
        return cv2.warpPerspective(np.full((h, w), 255, np.uint8), self.homography, size, flags=cv2.INTER_NEAREST)


@dataclass(frozen=True)
class AlignFailure:
    reason: str


def target_corners(size: tuple[int, int]) -> np.ndarray:
    """표준 이미지의 네 모서리. 픽셀 중심이 정수인 규칙이라 가장자리는 -0.5와 w-0.5다(OpenCV warp와 같은 규칙)."""
    w, h = size
    return np.array([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]], dtype=np.float32)


def border_error_px(homography: np.ndarray, contour: np.ndarray, size: tuple[int, int]) -> float | None:
    """윤곽 점을 표준 좌표로 옮겨 직사각형 테두리까지의 거리 RMS. 잴 점이 없으면 None."""
    w, h = size[0] - 0.5, size[1] - 0.5
    pts = cv2.perspectiveTransform(contour.reshape(-1, 1, 2).astype(np.float64), homography).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    dist = np.minimum.reduce([np.abs(x + 0.5), np.abs(x - w), np.abs(y + 0.5), np.abs(y - h)])
    mx, my = CORNER_SKIP * min(w, h), CORNER_SKIP * min(w, h)
    near_corner = ((x < mx) | (x > w - mx)) & ((y < my) | (y > h - my))
    dist = dist[~near_corner]
    if dist.size == 0:
        return None
    return float(np.sqrt(np.mean(dist**2)))


def align_quad(image: np.ndarray, quad: Quad, size: tuple[int, int]) -> Alignment:
    homography = cv2.getPerspectiveTransform(quad.corners, target_corners(size))
    homography = homography / homography[2, 2]
    rectified = cv2.warpPerspective(image, homography, size, flags=cv2.INTER_LINEAR)
    return Alignment(homography=homography, error_px=border_error_px(homography, quad.contour, size),
                     rectified=rectified, corners=quad.corners, photo_size=(image.shape[1], image.shape[0]),
                     partial=quad.partial)


def align_photo(image: np.ndarray, size: tuple[int, int]) -> Alignment | AlignFailure:
    quad = find_quad(image, size)
    if isinstance(quad, QuadFailure):
        return AlignFailure(quad.reason)
    return align_quad(image, quad, size)


def to_standard(homography: np.ndarray, polygon: list[list[float]]) -> list[list[float]]:
    """사진 px 다각형 → 표준 px. 부스 탐지(capture-server)도 같은 행렬로 옮긴다."""
    pts = np.asarray(polygon, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, np.asarray(homography, dtype=np.float64)).reshape(-1, 2)
    return [[round(float(x), 2), round(float(y), 2)] for x, y in out]
