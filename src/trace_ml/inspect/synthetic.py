"""합성 사진: 표준 면 이미지를 원근 변형해 배경 위에 놓는다. 정답 호모그래피를 알아 정합 오차를 잴 수 있다.

가장자리는 4배로 그린 뒤 줄여(면적 평균) 실제 카메라처럼 부드럽다. 테스트와 `trace-ml align eval`이 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .align import target_corners

SS = 4  # 가장자리 초과 표본 배율


@dataclass(frozen=True)
class SyntheticPhoto:
    image: np.ndarray
    corners: np.ndarray  # 정답 사진 px 꼭짓점(왼쪽 위부터 시계 방향)
    homography: np.ndarray  # 정답 사진 px → 표준 px


def face_texture(size: tuple[int, int], seed: int = 0) -> np.ndarray:
    """밝은 면 + 무늬(로고·긁힘 비슷한 선). 특징점 비교가 되도록 무늬를 흩뿌린다."""
    w, h = size
    rng = np.random.default_rng(seed)
    tex = np.full((h, w, 3), 175, np.uint8)
    for _ in range(40):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(6, max(7, min(w, h) // 8)))
        shade = int(rng.integers(90, 150))
        if rng.random() < 0.5:
            cv2.circle(tex, (x, y), r, (shade, shade, shade), -1)
        else:
            x2, y2 = int(x + rng.integers(-3 * r, 3 * r)), int(y + rng.integers(-r, r))
            cv2.line(tex, (x, y), (x2, y2), (shade, shade, shade), 2)
    return tex


def render(texture: np.ndarray, corners: np.ndarray, photo_size: tuple[int, int],
           background: int = 30, noise: float = 3.0, seed: int = 0) -> SyntheticPhoto:
    h_t, w_t = texture.shape[:2]
    pw, ph = photo_size
    corners = np.asarray(corners, dtype=np.float32)
    std = target_corners((w_t, h_t))
    # 4배 캔버스에 그린다: 연속 좌표 x(픽셀 중심 정수) → 4배 좌표 SS*x + (SS-1)/2
    big = corners * SS + (SS - 1) / 2
    to_big = cv2.getPerspectiveTransform(std, big)
    canvas = np.full((ph * SS, pw * SS, 3), background, np.uint8)
    warped = cv2.warpPerspective(texture, to_big, (pw * SS, ph * SS), flags=cv2.INTER_LINEAR)
    mask = np.zeros((ph * SS, pw * SS), np.uint8)
    cv2.fillPoly(mask, [np.round(big * 16).astype(np.int32)], 255, lineType=cv2.LINE_8, shift=4)
    canvas[mask > 0] = warped[mask > 0]
    image = cv2.resize(canvas, (pw, ph), interpolation=cv2.INTER_AREA)
    if noise > 0:
        rng = np.random.default_rng(seed)
        image = np.clip(image + rng.normal(0, noise, image.shape), 0, 255).astype(np.uint8)
    truth = cv2.getPerspectiveTransform(corners, std)
    return SyntheticPhoto(image=image, corners=corners, homography=truth / truth[2, 2])


def random_corners(face_px: tuple[int, int], photo_size: tuple[int, int], rng: np.random.Generator,
                   tilt: float = 0.08) -> np.ndarray:
    """사진 가운데 쯤에 면을 두고 꼭짓점을 흔든다(tilt = 면 크기 대비 원근 흔들림)."""
    pw, ph = photo_size
    fw, fh = face_px
    scale = min(pw / fw, ph / fh) * rng.uniform(0.6, 0.8)
    cx, cy = pw / 2 + rng.uniform(-0.05, 0.05) * pw, ph / 2 + rng.uniform(-0.05, 0.05) * ph
    half_w, half_h = fw * scale / 2, fh * scale / 2
    base = np.array([[cx - half_w, cy - half_h], [cx + half_w, cy - half_h],
                     [cx + half_w, cy + half_h], [cx - half_w, cy + half_h]], dtype=np.float32)
    jitter = rng.uniform(-tilt, tilt, (4, 2)) * np.array([half_w, half_h])
    return (base + jitter).astype(np.float32)


def corner_error_px(homography: np.ndarray, truth_corners: np.ndarray, face_px: tuple[int, int]) -> np.ndarray:
    """정답 꼭짓점을 추정 호모그래피로 옮긴 위치와 표준 꼭짓점 사이 거리(표준 px), 꼭짓점 4개."""
    est = cv2.perspectiveTransform(truth_corners.reshape(-1, 1, 2).astype(np.float64), homography).reshape(-1, 2)
    return np.linalg.norm(est - target_corners(face_px), axis=1)


def booth_half_corners(face_px: tuple[int, int], photo_size: tuple[int, int], rng: np.random.Generator,
                       half: str, tilt: float = 3.0) -> np.ndarray:
    """부스 반쪽 사진: 면 폭이 사진 가로의 ~90%, CAM1(front)은 앞끝이 보이고 뒤가 위로 잘림, CAM2(back)는 반대.

    tilt는 꼭짓점 흔들림(사진 px). 카메라가 바닥을 수직으로 보므로 작다.
    """
    pw, ph = photo_size
    fw, fh = face_px
    scale = pw * rng.uniform(0.85, 0.92) / fw
    w, h = fw * scale, fh * scale
    cx = pw / 2 + rng.uniform(-0.02, 0.02) * pw
    if half == "front":
        y1 = ph * rng.uniform(0.8, 0.9)
        y0 = y1 - h
    else:
        y0 = ph * rng.uniform(0.1, 0.2)
        y1 = y0 + h
    base = np.array([[cx - w / 2, y0], [cx + w / 2, y0], [cx + w / 2, y1], [cx - w / 2, y1]], dtype=np.float32)
    return (base + rng.uniform(-tilt, tilt, (4, 2))).astype(np.float32)
