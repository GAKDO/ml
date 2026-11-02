"""사진에서 노트북 면의 윤곽 4점을 찾는다(배경과 면의 밝기 차이 기준).

부스는 어두운 배경 위에 고정 카메라, 폰은 화면 가이드에 맞춰 찍으므로 사진에서 가장 큰 사각 윤곽이 면이다.
간소화판은 사진의 위쪽이 faces.md "이미지 위쪽"이라고 보고, 네 점을 사진 안 위치로 정렬한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MIN_AREA_RATIO = 0.005  # 윤곽이 사진의 0.5% 미만이면 면이 아니라고 본다(옆면은 가늘어 작다)


@dataclass(frozen=True)
class Quad:
    corners: np.ndarray  # (4, 2) float32, 왼쪽 위·오른쪽 위·오른쪽 아래·왼쪽 아래(사진 px). 잘린 면이면 사진 밖일 수 있음
    contour: np.ndarray  # (N, 2) float32, 면의 실제 가장자리 윤곽 점(사진 테두리에 닿은 점 제외, 오차 재기용)
    partial: bool = False  # 한 변이 사진 밖으로 잘려 꼭짓점 2개를 늘려 만든 경우


@dataclass(frozen=True)
class QuadFailure:
    reason: str


def order_corners(pts: np.ndarray) -> np.ndarray:
    """중심 기준 각도로 시계 방향으로 세운 뒤, 가장 위에 있는 변의 시작점(왼쪽 위)부터.

    x+y 최소로 고르면 가는 옆면이 조금만 기울어도 왼쪽 아래가 뽑혀 180°에 가깝게 뒤집힌다.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]  # 이미지 좌표(y 아래)에서 각도 오름차순 = 시계 방향
    mid_y = [(pts[i, 1] + pts[(i + 1) % 4, 1]) / 2 for i in range(4)]
    return np.roll(pts, -int(np.argmin(mid_y)), axis=0)


def _extreme_corners(hull: np.ndarray) -> np.ndarray:
    """사각형 근사가 안 될 때: 대각 방향으로 가장 먼 점 네 개(면이 사진과 거의 나란하다는 가정)."""
    pts = hull.reshape(-1, 2).astype(np.float32)
    s, d = pts.sum(axis=1), pts[:, 0] - pts[:, 1]
    return np.array([pts[np.argmin(s)], pts[np.argmax(d)], pts[np.argmax(s)], pts[np.argmin(d)]])


def _mask(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 배경이 면보다 밝은 사진(흰 배경)도 받도록, 테두리에 흰 픽셀이 더 많으면 뒤집는다
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    if border.mean() > 127:
        mask = cv2.bitwise_not(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _intersect(l1: np.ndarray, l2: np.ndarray) -> np.ndarray | None:
    """fitLine 결과 (vx, vy, x0, y0) 두 개의 교점."""
    (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = l1, l2
    det = vx1 * (-vy2) - (-vx2) * vy1
    if abs(det) < 1e-9:
        return None
    t = ((x2 - x1) * (-vy2) - (-vx2) * (y2 - y1)) / det
    return np.array([x1 + t * vx1, y1 + t * vy1], dtype=np.float32)


def _side_lines(corners: np.ndarray, contour: np.ndarray, skip: int | None = None) -> list[np.ndarray] | None:
    """변마다 윤곽 점에 직선(vx, vy, x0, y0)을 맞춘다(꼭짓점 근처 제외). skip 변은 None. 점이 모자라면 None."""
    lines: list[np.ndarray] = []
    center = corners.mean(axis=0)
    for i in range(4):
        if i == skip:
            lines.append(np.zeros(4, np.float32))
            continue
        a, b = corners[i], corners[(i + 1) % 4]
        ab = b - a
        length = float(np.linalg.norm(ab))
        if length < 1:
            return None
        rel = contour - a
        t = rel @ ab / (length * length)
        dist = np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / length
        near = contour[(t > 0.1) & (t < 0.9) & (dist < max(4.0, 0.03 * length))]
        if len(near) < 5:
            return None
        line = cv2.fitLine(near, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(4)
        # 윤곽 점은 면 가장자리 픽셀의 중심이라 실제 가장자리보다 0.5 px 안쪽이다 → 바깥으로 0.5 px 민다
        normal = np.array([-line[1], line[0]], dtype=np.float32)
        if float(normal @ (line[2:] - center)) < 0:
            normal = -normal
        line[2:] += 0.5 * normal
        lines.append(line)
    return lines


def _refine(corners: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """변마다 직선을 맞추고 이웃 직선의 교점을 꼭짓점으로. 둥근 모서리에 강하다."""
    lines = _side_lines(corners, contour)
    if lines is None:
        return corners
    refined = []
    for i in range(4):
        p = _intersect(lines[(i - 1) % 4], lines[i])
        if p is None:
            return corners
        refined.append(p)
    out = np.array(refined, dtype=np.float32)
    # 다듬은 점이 처음 점에서 너무 멀면(직선 맞추기 실패) 처음 점을 쓴다
    if float(np.max(np.linalg.norm(out - corners, axis=1))) > 0.05 * max(np.ptp(contour, axis=0)):
        return corners
    return out


def _on_frame(pts: np.ndarray, w: int, h: int, margin: float = 2.0) -> np.ndarray:
    x, y = pts[:, 0], pts[:, 1]
    return (x <= margin) | (y <= margin) | (x >= w - 1 - margin) | (y >= h - 1 - margin)


def _extrapolate(corners: np.ndarray, contour: np.ndarray, cut: int, face_px: tuple[int, int]) -> np.ndarray | None:
    """한 변(cut)이 사진 밖으로 잘린 면: 보이는 세 변의 직선으로 참 꼭짓점 2개를 구하고,
    보이는 변 길이로 배율(사진 px / 표준 px)을 잡아 옆 변 방향으로 표준 길이만큼 늘려 잘린 꼭짓점 2개를 만든다.

    부스 카메라는 바닥을 수직으로 내려다봐 원근이 작다는 가정이다(CAM1 앞 반·CAM2 뒤 반, booth README).
    """
    lines = _side_lines(corners, contour, skip=cut)
    if lines is None:
        return None
    i1, i2, i3 = (cut + 1) % 4, (cut + 2) % 4, (cut + 3) % 4
    a = _intersect(lines[i1], lines[i2])  # 꼭짓점 i2 (보이는 변의 시작)
    b = _intersect(lines[i2], lines[i3])  # 꼭짓점 i3 (보이는 변의 끝)
    if a is None or b is None:
        return None
    w_std, h_std = face_px
    visible_std, side_std = (w_std, h_std) if cut in (0, 2) else (h_std, w_std)
    scale = float(np.linalg.norm(b - a)) / visible_std
    out = corners.copy()
    out[i2], out[i3] = a, b
    for line, anchor, target in ((lines[i1], a, i1), (lines[i3], b, cut)):
        direction = line[:2] / max(1e-9, float(np.linalg.norm(line[:2])))
        if float(direction @ (corners[target] - anchor)) < 0:
            direction = -direction
        out[target] = anchor + direction * side_std * scale
    return out


def _near_sides(points: np.ndarray, corners: np.ndarray, skip: int) -> np.ndarray:
    """보이는 세 변 가까이의 점만 남긴다. 사진 테두리에 닿은 무늬(로고·어두운 흠)가 만든 홈은 가장자리가 아니다."""
    keep = np.zeros(len(points), bool)
    for i in range(4):
        if i == skip:
            continue
        a, b = corners[i], corners[(i + 1) % 4]
        ab = b - a
        length = float(np.linalg.norm(ab))
        rel = points - a
        dist = np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / max(length, 1e-9)
        keep |= dist < max(6.0, 0.02 * length)
    return points[keep]


def find_quad(image: np.ndarray, face_px: tuple[int, int] | None = None) -> Quad | QuadFailure:
    """면의 네 꼭짓점. face_px(표준 크기)를 주면 한 변이 사진 밖으로 잘린 면(부스 반쪽 사진)도 받는다."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    contours, _ = cv2.findContours(_mask(gray), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return QuadFailure("윤곽 4점을 찾지 못함(윤곽 없음)")
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < MIN_AREA_RATIO * w * h:
        return QuadFailure("윤곽 4점을 찾지 못함(면이 너무 작음)")
    hull = cv2.convexHull(biggest)
    # 옆면처럼 가는 면도 네 점이 남도록 허용 오차를 짧은 변 기준으로 잡는다
    short_side = min(cv2.minAreaRect(hull)[1])
    approx = None
    for eps in (0.02, 0.04, 0.07, 0.1, 0.15):
        candidate = cv2.approxPolyDP(hull, max(1.5, eps * short_side), True)
        if len(candidate) == 4:
            approx = candidate
            break
    if approx is None and hasattr(cv2, "approxPolyN"):  # OpenCV 4.11+: 넓이를 가장 적게 늘리는 4각형
        approx = cv2.approxPolyN(hull, 4, ensure_convex=True).reshape(4, 1, 2)
    if approx is None:
        approx = _extreme_corners(hull).reshape(4, 1, 2)
        if len({tuple(p) for p in approx.reshape(4, 2).tolist()}) < 4:
            return QuadFailure("윤곽 4점을 찾지 못함(사각형이 아님)")
    if not cv2.isContourConvex(approx.astype(np.float32)):
        return QuadFailure("윤곽 4점을 찾지 못함(볼록하지 않음)")
    contour = biggest.reshape(-1, 2).astype(np.float32)
    corners = order_corners(approx.reshape(4, 2))
    frame = _on_frame(contour, w, h)
    if not frame.any():
        return Quad(corners=_refine(corners, contour), contour=contour)
    # 사진 테두리에 닿은 면: 두 끝점이 모두 테두리 위인 변이 하나여야 한다
    corner_on_frame = _on_frame(corners, w, h, margin=4.0)
    cut_sides = [i for i in range(4) if corner_on_frame[i] and corner_on_frame[(i + 1) % 4]]
    if len(cut_sides) != 1 or int(corner_on_frame.sum()) != 2:
        return QuadFailure("면이 사진 밖으로 나감(두 변 이상 잘림)")
    if face_px is None:
        return QuadFailure("면이 사진 밖으로 나감(잘린 면을 늘릴 표준 크기 없음)")
    visible = contour[~frame]
    full = _extrapolate(corners, visible, cut_sides[0], face_px)
    if full is None:
        return QuadFailure("잘린 면의 보이는 변을 맞추지 못함")
    return Quad(corners=full, contour=_near_sides(visible, full, cut_sides[0]), partial=True)
