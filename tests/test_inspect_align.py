"""ML2 정합: 윤곽 4점 호모그래피(스토리 1·4, 경계 상황)."""

from __future__ import annotations

import numpy as np
import pytest
from inspect_helpers import DIMS, PHOTO

from trace_ml.inspect.align import AlignFailure, Alignment, align_photo, to_standard
from trace_ml.inspect.contour import order_corners
from trace_ml.inspect.faces import face_size_mm, face_size_px, load_dims
from trace_ml.inspect.measure import pair, stats, synthetic, synthetic_booth_half
from trace_ml.inspect.synthetic import booth_half_corners, corner_error_px, face_texture, random_corners, render
from trace_ml.labelstudio.config import ConfigError


def _aligned(face: str, seed: int = 0, photo: tuple[int, int] = PHOTO) -> tuple[Alignment, np.ndarray]:
    size = face_size_px(face, DIMS)
    rng = np.random.default_rng(seed)
    shot = render(face_texture(size, seed), random_corners(size, photo, rng), photo, seed=seed)
    result = align_photo(shot.image, size)
    assert isinstance(result, Alignment), result
    return result, shot.corners


def test_원근_변형한_TOP_사진을_펼치면_1216x860이고_꼭짓점_오차가_2px_이하다() -> None:
    result, truth = _aligned("TOP")
    assert result.rectified.shape[:2] == (860, 1216)
    assert corner_error_px(result.homography, truth, (1216, 860)).max() <= 2.0
    assert result.error_px is not None and result.error_px < 2.0
    assert result.method == "CONTOUR" and not result.partial


@pytest.mark.parametrize("face", ["SF", "SL"])
def test_가는_옆면도_네_꼭짓점으로_펼친다(face: str) -> None:
    result, truth = _aligned(face, seed=3)
    assert corner_error_px(result.homography, truth, face_size_px(face, DIMS)).max() <= 3.0


def test_면_크기는_faces_md의_기종_치수와_4px_per_mm다() -> None:
    assert face_size_mm("TOP", DIMS) == (304.1, 215.0)
    assert face_size_px("SF", DIMS) == (1216, 44)
    assert face_size_px("SL", DIMS) == (860, 44)
    assert face_size_px("SCR0", DIMS) == face_size_px("KB", DIMS) == (1216, 860)


def test_윤곽이_없는_사진은_사유와_함께_실패한다() -> None:
    blank = np.full((PHOTO[1], PHOTO[0], 3), 40, np.uint8)
    result = align_photo(blank, face_size_px("TOP", DIMS))
    assert isinstance(result, AlignFailure) and "윤곽" in result.reason


@pytest.mark.parametrize("half", ["front", "back"])
def test_부스_반쪽_사진은_잘린_꼭짓점을_표준_길이로_늘려_펼친다(half: str) -> None:
    size = face_size_px("TOP", DIMS)
    rng = np.random.default_rng(5)
    photo = (1280, 720)
    corners = booth_half_corners(size, photo, rng, half, tilt=1.0)
    shot = render(face_texture(size, 5), corners, photo, seed=5)
    result = align_photo(shot.image, size)
    assert isinstance(result, Alignment) and result.partial
    assert corner_error_px(result.homography, corners, size).max() <= 8.0
    coverage = result.coverage() > 0
    # 앞 반(CAM1)은 펼친 이미지 아래쪽(앞)을, 뒤 반(CAM2)은 위쪽(뒤)을 덮는다(faces.md: 위 = 힌지)
    top_cov, bottom_cov = coverage[:100].mean(), coverage[-100:].mean()
    assert (bottom_cov > top_cov) if half == "front" else (top_cov > bottom_cov)


def test_두_변_이상_잘린_면은_실패한다() -> None:
    size = face_size_px("TOP", DIMS)
    w, h = PHOTO
    corners = np.array([[-50, -40], [w * 0.8, -40], [w * 0.8, h * 0.7], [-50, h * 0.7]], np.float32)
    shot = render(face_texture(size), corners, PHOTO)
    result = align_photo(shot.image, size)
    assert isinstance(result, AlignFailure) and "잘림" in result.reason


def test_꼭짓점은_기울어진_가는_면에서도_왼쪽_위부터_시계_방향이다() -> None:
    # 왼쪽 아래 점의 x+y가 왼쪽 위보다 작은 경우(가는 옆면이 살짝 기움)
    pts = np.array([[1441, 581], [240, 578], [297, 537], [1521, 535]], np.float32)
    ordered = order_corners(pts)
    assert ordered[0].tolist() == [297, 537] and ordered[1].tolist() == [1521, 535]


def test_사진_px_다각형을_호모그래피로_표준_px로_옮긴다() -> None:
    result, truth = _aligned("TOP", seed=1)
    std = to_standard(result.homography, truth.tolist())
    assert np.allclose(std, [[-0.5, -0.5], [1215.5, -0.5], [1215.5, 859.5], [-0.5, 859.5]], atol=2.0)


def test_기종_설정이_표에_없으면_시작하지_않는다() -> None:
    assert load_dims({}).code == "MBA13-M2"
    with pytest.raises(ConfigError):
        load_dims({"TRACE_DEVICE_MODEL": "GRAM16"})


def test_정합_오차_측정은_px와_mm를_함께_낸다() -> None:
    s = synthetic("TOP", DIMS, n=3, photo_size=(800, 600))
    assert s.count == 3 and s.failed == 0 and s.max_px is not None and s.max_mm == round(s.max_px / 4, 3)
    half = synthetic_booth_half("TOP", DIMS, n=2, half="front", photo_size=(960, 540))
    assert half.count == 2
    assert stats([], 2).mean_px is None


def test_같은_면의_두_사진_사이_정합_오차를_특징점으로_잰다() -> None:
    size = face_size_px("TOP", DIMS)
    texture = face_texture(size, 9)
    rng = np.random.default_rng(9)
    a = render(texture, random_corners(size, PHOTO, rng), PHOTO, seed=1).image
    b = render(texture, random_corners(size, PHOTO, rng), PHOTO, seed=2).image
    result = pair(a, b, "TOP", DIMS)
    assert not isinstance(result, str), result
    assert result.matches >= 8 and result.stats.mean_px is not None and result.stats.mean_px < 3.0
