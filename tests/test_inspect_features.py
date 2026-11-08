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


def _shot(seed: int) -> SyntheticPhoto:
    """같은 면을 원근·회전이 다르게 찍은 합성 사진."""
    return render(TEXTURE, random_corners(SIZE, PHOTO, np.random.default_rng(seed)), PHOTO, seed=seed)


def _reference() -> np.ndarray:
    """이전 시점의 펼친 이미지(정답 호모그래피로 펼친 다른 사진)."""
    shot = _shot(100)
    return cv2.warpPerspective(shot.image, shot.homography, SIZE, flags=cv2.INTER_LINEAR)


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


def test_TRACE_ALIGN_기본은_contour이고_보정기가_없다() -> None:
    assert align_mode({}) == "contour"
    assert load_refiner({}) is None
    assert load_refiner({"TRACE_ALIGN": "contour"}) is None
    with pytest.raises(ConfigError):
        align_mode({"TRACE_ALIGN": "sift"})


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_특징점_보정은_윤곽보다_꼭짓점_오차가_작거나_같다(seed: int) -> None:
    shot = _shot(seed)
    contour = _contour(shot)
    refined = Refiner((OrbMatcher(),)).refine(shot.image, contour, _reference())
    assert refined.align_method == "orb"
    assert refined.method == "CONTOUR"  # codes.yaml에 ORB 값이 없어 이벤트에는 CONTOUR
    assert _err(refined, shot) <= _err(contour, shot) + 0.25  # 둘 다 1px 안팎, 보간 잡음만큼 여유
    assert refined.rectified.shape[:2] == (SIZE[1], SIZE[0])


@pytest.mark.parametrize("seed", [4, 5])
def test_윤곽이_어긋난_사진은_특징점_보정으로_꼭짓점_오차가_크게_준다(seed: int) -> None:
    shot = _shot(seed)
    contour = _contour(shot, jitter=6.0, seed=seed)
    refined = Refiner((OrbMatcher(),)).refine(shot.image, contour, _reference())
    assert _err(contour, shot) > 5.0
    assert _err(refined, shot) < 2.0
    assert refined.error_px is not None


def test_인라이어가_부족하면_윤곽_결과를_그대로_둔다() -> None:
    shot = _shot(1)
    contour = _contour(shot)
    flat = np.full((SIZE[1], SIZE[0], 3), 175, np.uint8)  # 무늬 없는 기준: 맞는 점이 없다
    refined = Refiner((OrbMatcher(),)).refine(shot.image, contour, flat)
    assert refined is contour and refined.align_method == "contour"


def test_다른_면의_기준이면_인라이어가_부족해_윤곽_결과를_쓴다() -> None:
    shot = _shot(1)
    contour = _contour(shot)
    other = face_texture(SIZE, 77)
    refined = Refiner((OrbMatcher(),), min_inliers=40).refine(shot.image, contour, other)
    assert refined is contour


def test_기준_크기가_다르면_윤곽_결과를_쓴다() -> None:
    shot = _shot(1)
    contour = _contour(shot)
    assert Refiner((OrbMatcher(),)).refine(shot.image, contour, np.zeros((10, 10, 3), np.uint8)) is contour


def test_모델_파일이_없으면_ORB로_폴백한다(tmp_path: Path) -> None:
    refiner = load_refiner({"TRACE_ALIGN": "features", "TRACE_MODEL_DIR": str(tmp_path)})
    assert refiner is not None and [m.name for m in refiner.matchers] == ["orb"]
    (tmp_path / "superpoint.onnx").write_bytes(b"x")  # 하나만 있어도 ORB
    refiner = load_refiner({"TRACE_ALIGN": "features", "TRACE_MODEL_DIR": str(tmp_path)})
    assert refiner is not None and [m.name for m in refiner.matchers] == ["orb"]


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

def _set(seed: int, set_id: str) -> tuple[dict[str, Any], Any, dict[str, bytes]]:
    ok, buf = cv2.imencode(".png", _shot(seed).image)
    assert ok
    body = buf.tobytes()
    photo_id = pid(f"p{seed}")
    return uploaded_event([(photo_id, "TOP", body)], set_id=pid(set_id), event_id=pid(f"e{seed}")), photo_id, \
        {photo_id: body}


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


def test_contour_모드는_기준_이미지를_만들지_않는다() -> None:
    s3 = MemoryS3()
    event, _, bodies = _set(1, "set1")
    for photo in event["photos"]:
        s3.objects[("trace-captures", photo["objectKey"])] = (bodies[photo["photoId"]], {})
    out = process(parse_request(event), Store(s3), WorkerConfig(dims=DIMS))
    assert out["modelVersion"] == "contour-v1"
    assert not any(key.startswith("align-ref/") for _, key in s3.objects)
