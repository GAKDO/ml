# 이 파일이 확인하는 것: 흠집 검사(ML2)의 정합 보정(FR-INS-02). 윤곽으로 찾은 네 꼭짓점은 그림자·반사로 어긋날 수 있어서,
# 이전 시점(예: 출고) 사진의 무늬 특징점과 맞춰 위치를 다듬는다.
# 방법: SuperPoint·LightGlue(AI 특징점 모델, ONNX 파일) → 없거나 실패하면 ORB(OpenCV 기본 특징점) → 그래도 안 되면 윤곽 결과 그대로.

"""ML2 정합 보정(FR-INS-02): 윤곽 4점 + 특징점(SuperPoint·LightGlue ONNX 또는 ORB) 보정."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fakes import MemoryS3
from inspect_helpers import DIMS, PHOTO, contract_validator, pid, uploaded_event

from trace_ml.inspect.align import Alignment, align_photo, align_quad
from trace_ml.inspect.contour import Quad
from trace_ml.inspect.faces import face_size_px
from trace_ml.inspect.features import (
    LightGlueMatcher,
    OrbMatcher,
    Refiner,
    align_mode,
    load_refiner,
)
from trace_ml.inspect.synthetic import SyntheticPhoto, corner_error_px, face_texture, random_corners, render
from trace_ml.inspect.worker import WorkerConfig, parse_request, process
from trace_ml.labelstudio.config import ConfigError
from trace_ml.store import Store

SIZE = face_size_px("TOP", DIMS)
TEXTURE = face_texture(SIZE, 0)


# 테스트 도구: 같은 면을 각도만 다르게 찍은 합성 사진.
def _shot(seed: int) -> SyntheticPhoto:
    """같은 면을 원근·회전이 다르게 찍은 합성 사진."""
    return render(TEXTURE, random_corners(SIZE, PHOTO, np.random.default_rng(seed)), PHOTO, seed=seed)


# 테스트 도구: 이전 시점에 펼쳐 둔 기준 이미지.
def _reference() -> np.ndarray:
    """이전 시점의 펼친 이미지(정답 호모그래피로 펼친 다른 사진)."""
    shot = _shot(100)
    return cv2.warpPerspective(shot.image, shot.homography, SIZE, flags=cv2.INTER_LINEAR)


# 테스트 도구: 윤곽 정합 결과. jitter를 주면 그림자·반사로 꼭짓점이 어긋난 상황을 흉내 낸다.
def _contour(shot: SyntheticPhoto, jitter: float = 0.0, seed: int = 0) -> Alignment:
    """윤곽 정합. jitter > 0이면 그림자·반사로 윤곽이 어긋난 경우처럼 꼭짓점을 흔든다."""
    result = align_photo(shot.image, SIZE)
    assert isinstance(result, Alignment), result
    if jitter == 0:
        return result
    moved = result.corners + np.random.default_rng(seed).uniform(-jitter, jitter, (4, 2)).astype(np.float32)
    assert result.contour is not None
    return align_quad(shot.image, Quad(corners=moved, contour=result.contour), SIZE)


def _err(alignment: Alignment, shot: SyntheticPhoto) -> float:
    return float(corner_error_px(alignment.homography, shot.corners, SIZE).max())


# [상황] 정합 방식 설정이 없거나 contour, 또는 모르는 값(sift)이다.
# [기대] 기본은 윤곽만(보정 없음), 모르는 값은 설정 오류.
def test_TRACE_ALIGN_기본은_contour이고_보정기가_없다() -> None:
    assert align_mode({}) == "contour"
    assert load_refiner({}) is None
    assert load_refiner({"TRACE_ALIGN": "contour"}) is None
    with pytest.raises(ConfigError):
        align_mode({"TRACE_ALIGN": "sift"})


# [상황] 윤곽이 잘 맞은 사진을 ORB로 보정한다.
# [기대] 보정 뒤 오차가 윤곽보다 나쁘지 않다. 이벤트에 싣는 방식 값은 CONTOUR(codes.yaml에 ORB 값이 없음).
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_특징점_보정은_윤곽보다_꼭짓점_오차가_작거나_같다(seed: int) -> None:
    shot = _shot(seed)
    contour = _contour(shot)
    refined = Refiner((OrbMatcher(),)).refine(shot.image, contour, _reference())
    assert refined.align_method == "orb"
    assert refined.method == "CONTOUR"  # codes.yaml에 ORB 값이 없어 이벤트에는 CONTOUR
    assert _err(refined, shot) <= _err(contour, shot) + 0.25  # 둘 다 1px 안팎, 보간 잡음만큼 여유
    assert refined.rectified.shape[:2] == (SIZE[1], SIZE[0])


# [상황] 윤곽 꼭짓점이 6px쯤 흔들려 어긋난 사진.
# [기대] 보정 전 오차 5px 초과가 보정 뒤 2px 미만으로 준다.
@pytest.mark.parametrize("seed", [4, 5])
def test_윤곽이_어긋난_사진은_특징점_보정으로_꼭짓점_오차가_크게_준다(seed: int) -> None:
    shot = _shot(seed)
    contour = _contour(shot, jitter=6.0, seed=seed)
    refined = Refiner((OrbMatcher(),)).refine(shot.image, contour, _reference())
    assert _err(contour, shot) > 5.0
    assert _err(refined, shot) < 2.0
    assert refined.error_px is not None


# [상황] 기준 이미지가 무늬 없는 단색이라 맞는 점이 없다.
# [기대] 억지로 맞추지 않고 윤곽 결과를 그대로 쓴다.
def test_인라이어가_부족하면_윤곽_결과를_그대로_둔다() -> None:
    shot = _shot(1)
    contour = _contour(shot)
    flat = np.full((SIZE[1], SIZE[0], 3), 175, np.uint8)  # 무늬 없는 기준: 맞는 점이 없다
    refined = Refiner((OrbMatcher(),)).refine(shot.image, contour, flat)
    assert refined is contour and refined.align_method == "contour"


# [상황] 기준 이미지가 다른 면(무늬가 전혀 다름)이다.
# [기대] 맞는 점이 모자라 윤곽 결과를 쓴다.
def test_다른_면의_기준이면_인라이어가_부족해_윤곽_결과를_쓴다() -> None:
    shot = _shot(1)
    contour = _contour(shot)
    other = face_texture(SIZE, 77)
    refined = Refiner((OrbMatcher(),), min_inliers=40).refine(shot.image, contour, other)
    assert refined is contour


# [상황] 기준 이미지 크기가 면 크기와 다르다.
# [기대] 윤곽 결과를 쓴다.
def test_기준_크기가_다르면_윤곽_결과를_쓴다() -> None:
    shot = _shot(1)
    contour = _contour(shot)
    assert Refiner((OrbMatcher(),)).refine(shot.image, contour, np.zeros((10, 10, 3), np.uint8)) is contour


# [상황] 특징점 모드인데 AI 모델 파일이 없거나 둘 중 하나만 있다.
# [기대] ORB만 쓴다.
def test_모델_파일이_없으면_ORB로_폴백한다(tmp_path: Path) -> None:
    refiner = load_refiner({"TRACE_ALIGN": "features", "TRACE_MODEL_DIR": str(tmp_path)})
    assert refiner is not None and [m.name for m in refiner.matchers] == ["orb"]
    (tmp_path / "superpoint.onnx").write_bytes(b"x")  # 하나만 있어도 ORB
    refiner = load_refiner({"TRACE_ALIGN": "features", "TRACE_MODEL_DIR": str(tmp_path)})
    assert refiner is not None and [m.name for m in refiner.matchers] == ["orb"]


# [상황] 모델 파일이 깨져 열리지 않는다.
# [기대] ORB만 쓴다(워커가 멈추지 않음).
def test_모델_파일이_열리지_않으면_ORB로_폴백한다(tmp_path: Path) -> None:
    for name in ("superpoint.onnx", "lightglue.onnx"):
        (tmp_path / name).write_bytes(b"broken")

    def broken(path: Path) -> Any:
        raise RuntimeError(f"열 수 없음: {path.name}")

    refiner = load_refiner({"TRACE_ALIGN": "features", "TRACE_MODEL_DIR": str(tmp_path)}, make_session=broken)
    assert refiner is not None and [m.name for m in refiner.matchers] == ["orb"]


# ---- SuperPoint·LightGlue ONNX(가짜 세션: 같은 입출력 모양) ----

class _Input:
    def __init__(self, name: str) -> None:
        self.name = name


# 테스트 도구: 가짜 SuperPoint. 모서리 점과 주변 조각을 특징으로 낸다.
class FakeSuperPoint:
    """모서리 점 + 16×16 조각(256차원)을 기술자로 낸다. 입력 (1, 1, H, W) 0~1."""

    def get_inputs(self) -> list[_Input]:
        return [_Input("image")]

    def run(self, _names: Any, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        image = feed["image"]
        assert image.ndim == 4 and image.shape[2] % 8 == 0 and image.shape[3] % 8 == 0
        gray = (image[0, 0] * 255).astype(np.uint8)
        pts = cv2.goodFeaturesToTrack(gray, 1500, 0.01, 6, blockSize=7)
        pts = np.zeros((0, 2), np.float32) if pts is None else pts.reshape(-1, 2)
        pad = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REFLECT).astype(np.float32)
        desc = np.array([pad[int(y):int(y) + 16, int(x):int(x) + 16].ravel() for x, y in np.round(pts)], np.float32)
        desc = desc - desc.mean(axis=1, keepdims=True) if len(desc) else np.zeros((0, 256), np.float32)
        desc /= np.linalg.norm(desc, axis=1, keepdims=True) + 1e-6
        return [pts[None].astype(np.int64), np.ones((1, len(pts)), np.float32), desc[None]]


# 테스트 도구: 가짜 LightGlue. 가깝고 서로 가장 닮은 점끼리 짝을 지어 낸다.
class FakeLightGlue:
    """정규화된 점이 가깝고 기술자가 서로 가장 비슷한 짝을 matches0 (M, 3)[배치, i, j]로 낸다."""

    def __init__(self) -> None:
        self.inputs: dict[str, np.ndarray] = {}

    def get_inputs(self) -> list[_Input]:
        return [_Input(n) for n in ("kpts0", "kpts1", "desc0", "desc1")]

    def run(self, _names: Any, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.inputs = feed
        k0, k1, d0, d1 = feed["kpts0"][0], feed["kpts1"][0], feed["desc0"][0], feed["desc1"][0]
        assert np.abs(k0).max() <= 1.01  # 긴 변 반을 1로 정규화
        sim = d0 @ d1.T
        sim[np.linalg.norm(k0[:, None] - k1[None], axis=2) > 0.05] = -1
        j = sim.argmax(axis=1)
        mutual = sim.argmax(axis=0)[j] == np.arange(len(k0))
        keep = mutual & (sim[np.arange(len(k0)), j] > 0.9)
        i = np.nonzero(keep)[0]
        return [np.stack([np.zeros_like(i), i, j[i]], axis=1), sim[i, j[i]]]


# [상황] 두 모델 파일이 있고, 꼭짓점이 조금 어긋난 사진.
# [기대] SuperPoint·LightGlue를 먼저(ORB는 예비) 쓰고, 오차가 줄며, 이벤트 방식 값은 LIGHTGLUE.
def test_모델_파일이_있으면_SuperPoint_LightGlue로_다듬고_method는_LIGHTGLUE다(tmp_path: Path) -> None:
    for name in ("superpoint.onnx", "lightglue.onnx"):
        (tmp_path / name).write_bytes(b"onnx")
    sessions = {"superpoint.onnx": FakeSuperPoint(), "lightglue.onnx": FakeLightGlue()}
    refiner = load_refiner({"TRACE_ALIGN": "features", "TRACE_MODEL_DIR": str(tmp_path)},
                           make_session=lambda p: sessions[p.name])
    assert refiner is not None and [m.name for m in refiner.matchers] == ["superpoint-lightglue", "orb"]
    shot = _shot(4)
    contour = _contour(shot, jitter=3.0, seed=4)
    refined = refiner.refine(shot.image, contour, _reference())
    assert refined.method == "LIGHTGLUE" and refined.align_method == "superpoint-lightglue"
    assert _err(refined, shot) < min(2.0, _err(contour, shot))


# [상황] LightGlue 실행 중 오류가 난다.
# [기대] 경고를 남기고 ORB로 다듬는다.
def test_LightGlue_추론이_실패하면_ORB로_다듬는다(caplog: pytest.LogCaptureFixture) -> None:
    class Broken(FakeLightGlue):
        def run(self, _names: Any, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
            raise RuntimeError("모양이 맞지 않음")

    refiner = Refiner((LightGlueMatcher(FakeSuperPoint(), Broken()), OrbMatcher()))
    shot = _shot(5)
    with caplog.at_level(logging.WARNING):
        refined = refiner.refine(shot.image, _contour(shot, jitter=6.0, seed=5), _reference())
    assert refined.align_method == "orb" and "superpoint-lightglue" in caplog.text


# ---- 워커: 대여의 기준 이미지 ----

# 테스트 도구: 합성 사진 한 장짜리 촬영 세트 업로드 메시지와 사진 본문을 만든다.
def _set(seed: int, set_id: str) -> tuple[dict[str, Any], Any, dict[str, bytes]]:
    ok, buf = cv2.imencode(".png", _shot(seed).image)
    assert ok
    body = buf.tobytes()
    photo_id = pid(f"p{seed}")
    return uploaded_event([(photo_id, "TOP", body)], set_id=pid(set_id), event_id=pid(f"e{seed}")), photo_id, \
        {photo_id: body}


# [상황] 같은 렌탈의 첫 촬영 세트와 다음 촬영 세트를 특징점 모드로 차례로 검사한다.
# [기대] 첫 세트의 펼친 이미지를 기준으로 저장하고, 다음 세트는 그 기준으로 ORB 보정한다.
#        기준은 덮어쓰지 않고, 결과 메시지는 약속한 형식(trace.inspect.completed)에 맞는다.
def test_features_모드는_첫_시점_텍스처를_기준으로_남기고_다음_시점을_그_기준으로_다듬는다(
        caplog: pytest.LogCaptureFixture) -> None:
    s3 = MemoryS3()
    store, config, refiner = Store(s3), WorkerConfig(dims=DIMS), Refiner((OrbMatcher(),))
    ref_key = ("trace-derived", f"align-ref/{pid('rental1')}/TOP.png")
    first, _, bodies = _set(100, "set1")
    for photo in first["photos"]:
        s3.objects[("trace-captures", photo["objectKey"])] = (bodies[photo["photoId"]], {})
    out1 = process(parse_request(first), store, config, refiner=refiner)
    assert out1["modelVersion"] == "features-v1" and out1["alignments"][0]["method"] == "CONTOUR"
    assert ref_key in s3.objects
    reference = s3.objects[ref_key][0]

    second, _, bodies = _set(4, "set2")
    for photo in second["photos"]:
        s3.objects[("trace-captures", photo["objectKey"])] = (bodies[photo["photoId"]], {})
    with caplog.at_level(logging.INFO, logger="trace_ml.inspect"):
        out2 = process(parse_request(second), store, config, refiner=refiner)
    assert "alignMethod=orb" in caplog.text
    assert s3.objects[ref_key][0] == reference  # 기준은 덮어쓰지 않는다
    assert out2["alignments"][0]["method"] == "CONTOUR"  # ORB 보정은 이벤트에 새 값을 내지 않는다
    contract_validator("trace.inspect.completed.schema.json").validate(out2)


# [상황] 기본(윤곽만) 모드로 검사한다.
# [기대] 모델 버전은 contour-v1이고 기준 이미지를 저장하지 않는다.
def test_contour_모드는_기준_이미지를_만들지_않는다() -> None:
    s3 = MemoryS3()
    event, _, bodies = _set(1, "set1")
    for photo in event["photos"]:
        s3.objects[("trace-captures", photo["objectKey"])] = (bodies[photo["photoId"]], {})
    out = process(parse_request(event), Store(s3), WorkerConfig(dims=DIMS))
    assert out["modelVersion"] == "contour-v1"
    assert not any(key.startswith("align-ref/") for _, key in s3.objects)


# [상황] 저장된 기준 이미지가 깨져 읽을 수 없다.
# [기대] 새 이미지로 덮어쓰지 않는다(첫 시점 기준을 잃지 않게).
def test_기존_기준을_읽지_못해도_새_텍스처로_덮어쓰지_않는다() -> None:
    s3 = MemoryS3()
    ref_key = ("trace-derived", f"align-ref/{pid('rental1')}/TOP.png")
    s3.objects[ref_key] = (b"not-a-png", {})
    event, _, bodies = _set(1, "set1")
    for photo in event["photos"]:
        s3.objects[("trace-captures", photo["objectKey"])] = (bodies[photo["photoId"]], {})
    process(parse_request(event), Store(s3), WorkerConfig(dims=DIMS), refiner=Refiner((OrbMatcher(),)))
    assert s3.objects[ref_key][0] == b"not-a-png"
