"""정합 오차 재기(경진대회 수치·수용 기준 "폰과 부스 사이 정합 오차를 기록한다").

- synthetic: 정답 호모그래피를 아는 합성 사진으로 꼭짓점 오차(표준 px·mm)
- pair: 같은 면을 찍은 실제 두 사진(부스·폰)을 각각 펼친 뒤, 펼친 이미지끼리 특징점 위치 차이(표준 px·mm)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .align import AlignFailure, Alignment, align_photo
from .faces import PX_PER_MM, DeviceDims, face_size_px
from .synthetic import booth_half_corners, corner_error_px, face_texture, random_corners, render


@dataclass(frozen=True)
class Stats:
    count: int
    failed: int
    mean_px: float | None
    p95_px: float | None
    max_px: float | None
    mean_mm: float | None
    p95_mm: float | None
    max_mm: float | None

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def stats(values: list[float], failed: int) -> Stats:
    if not values:
        return Stats(0, failed, None, None, None, None, None, None)
    arr = np.asarray(values, dtype=np.float64)
    mean, p95, mx = float(arr.mean()), float(np.percentile(arr, 95)), float(arr.max())
    r = lambda v: round(v, 3)
    return Stats(len(values), failed, r(mean), r(p95), r(mx), r(mean / PX_PER_MM), r(p95 / PX_PER_MM),
                 r(mx / PX_PER_MM))


def synthetic(face: str, dims: DeviceDims, n: int, photo_size: tuple[int, int] = (1280, 960),
              seed: int = 0, tilt: float = 0.08) -> Stats:
    """사진마다 꼭짓점 4개 중 가장 큰 오차를 모은다."""
    size = face_size_px(face, dims)
    rng = np.random.default_rng(seed)
    texture = face_texture(size, seed)
    errors: list[float] = []
    failed = 0
    for i in range(n):
        photo = render(texture, random_corners(size, photo_size, rng, tilt), photo_size, seed=seed + i)
        result = align_photo(photo.image, size)
        if isinstance(result, AlignFailure):
            failed += 1
            continue
        errors.append(float(corner_error_px(result.homography, photo.corners, size).max()))
    return stats(errors, failed)


def synthetic_booth_half(face: str, dims: DeviceDims, n: int, half: str,
                         photo_size: tuple[int, int] = (1280, 720), seed: int = 0) -> Stats:
    """부스 반쪽 사진(한 변이 잘림, 잘린 꼭짓점은 늘려 만듦)의 꼭짓점 오차. 잘린 꼭짓점이 오차 대부분이다."""
    size = face_size_px(face, dims)
    rng = np.random.default_rng(seed)
    texture = face_texture(size, seed)
    errors: list[float] = []
    failed = 0
    for i in range(n):
        corners = booth_half_corners(size, photo_size, rng, half)
        photo = render(texture, corners, photo_size, seed=seed + i)
        result = align_photo(photo.image, size)
        if isinstance(result, AlignFailure):
            failed += 1
            continue
        errors.append(float(corner_error_px(result.homography, photo.corners, size).max()))
    return stats(errors, failed)


@dataclass(frozen=True)
class PairResult:
    matches: int
    stats: Stats
    errors: tuple[float | None, float | None]  # 두 사진 각각의 errorPx(윤곽 잔차)


def pair(image_a: np.ndarray, image_b: np.ndarray, face: str, dims: DeviceDims,
         min_matches: int = 8) -> PairResult | str:
    """두 사진을 펼쳐 ORB 특징점을 맞추고, 같은 점이 표준 좌표에서 얼마나 떨어졌는지 잰다. 실패하면 사유 문자열."""
    size = face_size_px(face, dims)
    a, b = align_photo(image_a, size), align_photo(image_b, size)
    for name, r in (("첫 사진", a), ("둘째 사진", b)):
        if isinstance(r, AlignFailure):
            return f"{name}: {r.reason}"
    assert isinstance(a, Alignment) and isinstance(b, Alignment)
    gray = [cv2.cvtColor(x.rectified, cv2.COLOR_BGR2GRAY) if x.rectified.ndim == 3 else x.rectified for x in (a, b)]
    orb = cv2.ORB_create(nfeatures=4000)
    (ka, da), (kb, db) = (orb.detectAndCompute(g, None) for g in gray)
    if da is None or db is None:
        return "특징점이 없음(무늬 없는 면)"
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, *rest in knn if rest and m.distance < 0.75 * rest[0].distance]
    if len(good) < min_matches:
        return f"맞는 특징점이 {len(good)}개뿐"
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    # 잘못 맞은 점은 RANSAC 이상치로 뺀다(정합 오차는 맞은 점의 위치 차이)
    _, inliers = cv2.findHomography(pa, pb, cv2.RANSAC, 10.0)
    keep = inliers.ravel().astype(bool) if inliers is not None else np.ones(len(good), bool)
    dist = np.linalg.norm(pa[keep] - pb[keep], axis=1)
    return PairResult(int(keep.sum()), stats([float(d) for d in dist], 0), (a.error_px, b.error_px))
