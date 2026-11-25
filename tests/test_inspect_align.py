"""이 파일이 확인하는 것: 흠집 검사(ML2)의 "정합". 비스듬히 찍힌 노트북 면 사진에서 네 꼭짓점을 찾아
정면에서 본 반듯한 이미지(1mm = 4px)로 펼친다. 그래야 출고·반납 사진을 같은 자리끼리 비교할 수 있다.
면 이름: TOP 윗면(덮개), SF 앞 옆면, SL 왼쪽 옆면, KB 키보드, SCR0 화면.

ML2 정합: 윤곽 4점 호모그래피(스토리 1·4, 경계 상황).
"""

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


# 테스트 도구: 정답 꼭짓점을 아는 합성 사진을 만들어 정합하고, 결과와 정답을 함께 돌려준다.
def _aligned(face: str, seed: int = 0, photo: tuple[int, int] = PHOTO) -> tuple[Alignment, np.ndarray]:
    size = face_size_px(face, DIMS)
    rng = np.random.default_rng(seed)
    shot = render(face_texture(size, seed), random_corners(size, photo, rng), photo, seed=seed)
    result = align_photo(shot.image, size)
    assert isinstance(result, Alignment), result
    return result, shot.corners


# [상황] 비스듬히 찍힌 윗면 합성 사진을 정합한다.
# [기대] 1216×860px(304.1×215mm)로 펼쳐지고, 꼭짓점 오차가 2px(0.5mm) 이하다.
def test_원근_변형한_TOP_사진을_펼치면_1216x860이고_꼭짓점_오차가_2px_이하다() -> None:
    result, truth = _aligned("TOP")
    assert result.rectified.shape[:2] == (860, 1216)
    assert corner_error_px(result.homography, truth, (1216, 860)).max() <= 2.0
    assert result.error_px is not None and result.error_px < 2.0
    assert result.method == "CONTOUR" and not result.partial


# [상황] 두께가 얇은 앞·왼쪽 옆면 사진을 정합한다.
# [기대] 가는 띠 모양이어도 꼭짓점 오차 3px 이하로 펼친다.
@pytest.mark.parametrize("face", ["SF", "SL"])
def test_가는_옆면도_네_꼭짓점으로_펼친다(face: str) -> None:
    result, truth = _aligned(face, seed=3)
    assert corner_error_px(result.homography, truth, face_size_px(face, DIMS)).max() <= 3.0


# [상황] 맥북 에어 13(M2) 치수로 면마다 크기를 계산한다.
# [기대] 윗면 304.1×215mm, 앞 옆면 1216×44px 등 기종 치수 × 4px/mm와 같다.
def test_면_크기는_faces_md의_기종_치수와_4px_per_mm다() -> None:
    assert face_size_mm("TOP", DIMS) == (304.1, 215.0)
    assert face_size_px("SF", DIMS) == (1216, 44)
    assert face_size_px("SL", DIMS) == (860, 44)
    assert face_size_px("SCR0", DIMS) == face_size_px("KB", DIMS) == (1216, 860)


# [상황] 노트북이 없는 단색 사진을 정합한다.
# [기대] "윤곽"을 못 찾았다는 사유와 함께 실패로 돌려준다(엉뚱한 결과를 내지 않음).
def test_윤곽이_없는_사진은_사유와_함께_실패한다() -> None:
    blank = np.full((PHOTO[1], PHOTO[0], 3), 40, np.uint8)
    result = align_photo(blank, face_size_px("TOP", DIMS))
    assert isinstance(result, AlignFailure) and "윤곽" in result.reason


# [상황] 부스 카메라 2대가 윗면의 앞 반(CAM1)·뒤 반(CAM2)만 찍은 사진이라 꼭짓점 일부가 잘려 있다.
# [기대] 잘린 쪽을 기종 길이로 늘려 펼치고(부분 정합 표시), 앞 반 사진은 펼친 이미지의 아래(앞)쪽을,
#        뒤 반 사진은 위(힌지)쪽을 덮는다.
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


# [상황] 면이 사진 밖으로 크게 나가 두 변 이상 잘렸다.
# [기대] "잘림" 사유로 실패한다(추측으로 펼치지 않음).
def test_두_변_이상_잘린_면은_실패한다() -> None:
    size = face_size_px("TOP", DIMS)
    w, h = PHOTO
    corners = np.array([[-50, -40], [w * 0.8, -40], [w * 0.8, h * 0.7], [-50, h * 0.7]], np.float32)
    shot = render(face_texture(size), corners, PHOTO)
    result = align_photo(shot.image, size)
    assert isinstance(result, AlignFailure) and "잘림" in result.reason


# [상황] 살짝 기운 가는 옆면에서 찾은 꼭짓점 4개의 순서를 정한다.
# [기대] 왼쪽 위 → 오른쪽 위 → … 시계 방향으로 바르게 정렬된다(뒤집혀 펼쳐지지 않게).
def test_꼭짓점은_기울어진_가는_면에서도_왼쪽_위부터_시계_방향이다() -> None:
    # 왼쪽 아래 점의 x+y가 왼쪽 위보다 작은 경우(가는 옆면이 살짝 기움)
    pts = np.array([[1441, 581], [240, 578], [297, 537], [1521, 535]], np.float32)
    ordered = order_corners(pts)
    assert ordered[0].tolist() == [297, 537] and ordered[1].tolist() == [1521, 535]


# [상황] 사진 위의 면 꼭짓점 좌표를 펼친 이미지 좌표로 옮긴다.
# [기대] 펼친 이미지의 네 모서리에 맞게 옮겨진다(사진에서 찾은 흠집 위치를 같은 방법으로 옮길 수 있음).
def test_사진_px_다각형을_호모그래피로_표준_px로_옮긴다() -> None:
    result, truth = _aligned("TOP", seed=1)
    std = to_standard(result.homography, truth.tolist())
    assert np.allclose(std, [[-0.5, -0.5], [1215.5, -0.5], [1215.5, 859.5], [-0.5, 859.5]], atol=2.0)


# [상황] 기종 설정이 없을 때와, 표에 없는 기종(GRAM16)을 줄 때.
# [기대] 없으면 기본 MBA13-M2를 쓰고, 모르는 기종이면 설정 오류로 시작하지 않는다.
def test_기종_설정이_표에_없으면_시작하지_않는다() -> None:
    assert load_dims({}).code == "MBA13-M2"
    with pytest.raises(ConfigError):
        load_dims({"TRACE_DEVICE_MODEL": "GRAM16"})


# [상황] 합성 사진 여러 장으로 정합 오차를 측정한다(전체 면, 부스 반쪽 면).
# [기대] 장수·실패 수·최대 오차가 px와 mm(px ÷ 4)로 함께 나오고, 측정할 것이 없으면 값이 비어 있다.
def test_정합_오차_측정은_px와_mm를_함께_낸다() -> None:
    s = synthetic("TOP", DIMS, n=3, photo_size=(800, 600))
    assert s.count == 3 and s.failed == 0 and s.max_px is not None and s.max_mm == round(s.max_px / 4, 3)
    half = synthetic_booth_half("TOP", DIMS, n=2, half="front", photo_size=(960, 540))
    assert half.count == 2
    assert stats([], 2).mean_px is None


# [상황] 같은 윗면을 서로 다른 각도로 찍은 사진 두 장(예: 출고 때와 반납 때).
# [기대] 특징점이 8개 이상 맞고, 두 사진 사이 평균 어긋남이 3px 미만이다.
def test_같은_면의_두_사진_사이_정합_오차를_특징점으로_잰다() -> None:
    size = face_size_px("TOP", DIMS)
    texture = face_texture(size, 9)
    rng = np.random.default_rng(9)
    a = render(texture, random_corners(size, PHOTO, rng), PHOTO, seed=1).image
    b = render(texture, random_corners(size, PHOTO, rng), PHOTO, seed=2).image
    result = pair(a, b, "TOP", DIMS)
    assert not isinstance(result, str), result
    assert result.matches >= 8 and result.stats.mean_px is not None and result.stats.mean_px < 3.0
