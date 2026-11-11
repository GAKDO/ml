"""특징점 정합(선택, FR-INS-02): 윤곽 4점으로 펼친 사진을 같은 기기의 기준(이전 시점) 펼친 이미지에 맞춰 다듬는다.

환경 변수 `TRACE_ALIGN=contour|features`(기본 contour). features일 때만 쓴다.

1. 특징점 매칭: 모델 폴더(`TRACE_MODEL_DIR`)에 `superpoint.onnx`·`lightglue.onnx`가 둘 다 있으면 SuperPoint+LightGlue(ONNX),
   없거나 실패하면 OpenCV ORB. 매칭은 표준 좌표끼리(펼친 이미지 ↔ 기준 이미지)라 남은 차이만 찾는다.
2. RANSAC 호모그래피 → ECC(밝기 상관) 다듬기(¼·½·1 배율). ECC가 실패하면 RANSAC 결과를 쓴다.
3. 인라이어가 모자라거나, 보정이 너무 크거나, 기준과의 상관이 나빠지면 윤곽 결과를 그대로 둔다.

이벤트 `method`는 codes.yaml `inspection.alignment_method` 값만 낸다: SuperPoint+LightGlue로 다듬었으면 LIGHTGLUE,
그 밖(윤곽만·ORB 보정)은 CONTOUR. ORB 보정 여부는 `Alignment.align_method`(내부 값)와 로그로만 남긴다.
torch 없이 onnxruntime·OpenCV만 쓴다(inspect-worker 이미지를 작게 둔다).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from ..labelstudio.config import ConfigError
from .align import METHOD_CONTOUR, Alignment, border_error_px, target_corners

log = logging.getLogger(__name__)

ALIGN_MODES = ("contour", "features")
METHOD_LIGHTGLUE = "LIGHTGLUE"  # codes.yaml inspection.alignment_method
SUPERPOINT_FILE = "superpoint.onnx"
LIGHTGLUE_FILE = "lightglue.onnx"
REFERENCE_PREFIX = "align-ref"  # trace-derived/align-ref/<rentalId>/<면>.png: 대여의 첫 텍스처를 기준으로 둔다

MIN_INLIERS = 15  # 이보다 적으면 특징점 보정을 믿지 않는다
RANSAC_PX = 3.0
MAX_SHIFT_RATIO = 0.05  # 보정이 표준 꼭짓점을 긴 변의 5%보다 많이 옮기면 잘못 맞은 것으로 본다
ECC_SCALES = (0.25, 0.5, 1.0)
ECC_ITERATIONS = 50
NCC_SLACK = 1e-3  # 기준과의 상관이 이만큼보다 더 나빠지면 버린다


def align_mode(env: Mapping[str, str]) -> str:
    raw = (env.get("TRACE_ALIGN") or "contour").strip().lower()
    if raw not in ALIGN_MODES:
        raise ConfigError(f"TRACE_ALIGN은 {'|'.join(ALIGN_MODES)} 중 하나여야 함: {raw}")
    return raw


def reference_key(rental_id: str, face: str) -> str:
    return f"{REFERENCE_PREFIX}/{rental_id}/{face}.png"


class Matcher(Protocol):
    name: str  # 내부 alignMethod
    method: str  # 이벤트 method(codes.yaml 값)

    def match(self, current: np.ndarray, reference: np.ndarray,
              mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """회색 표준 이미지 두 장 → 맞는 점 (N, 2) 두 개(현재 px, 기준 px)."""
        ...


_EMPTY = (np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32))


class OrbMatcher:
    name = "orb"
    method = METHOD_CONTOUR  # codes.yaml에 ORB 값이 없어 이벤트에는 CONTOUR로 낸다

    def __init__(self, features: int = 4000, ratio: float = 0.8) -> None:
        self._features = features
        self._ratio = ratio

    def match(self, current: np.ndarray, reference: np.ndarray,
              mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        orb = cv2.ORB_create(nfeatures=self._features)
        k0, d0 = orb.detectAndCompute(current, mask)
        k1, d1 = orb.detectAndCompute(reference, None)
        if d0 is None or d1 is None or len(k0) < 2 or len(k1) < 2:
            return _EMPTY
        knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d0, d1, k=2)
        good = [m for m, *rest in knn if rest and m.distance < self._ratio * rest[0].distance]
        if not good:
            return _EMPTY
        return (np.float32([k0[m.queryIdx].pt for m in good]), np.float32([k1[m.trainIdx].pt for m in good]))


class Session(Protocol):
    """onnxruntime.InferenceSession에서 쓰는 것만(테스트에서 바꿔 끼운다)."""

    def get_inputs(self) -> list[Any]: ...
    def run(self, output_names: Any, input_feed: dict[str, np.ndarray]) -> list[Any]: ...


class LightGlueMatcher:
    """SuperPoint·LightGlue 분리 ONNX(fabio-sim/LightGlue-ONNX 내보내기 형식).

    - superpoint.onnx: 입력 (1, 1, H, W) 회색 0~1(H·W는 8의 배수) → keypoints (1, N, 2) x·y px, scores, descriptors (1, N, D)
    - lightglue.onnx: kpts0·kpts1(가운데 0, 긴 변 반을 1로 정규화)·desc0·desc1 → matches0 (M, 2 또는 3: 마지막 두 열이 번호)
    """

    name = "superpoint-lightglue"
    method = METHOD_LIGHTGLUE

    def __init__(self, superpoint: Session, lightglue: Session) -> None:
        self._sp = superpoint
        self._lg = lightglue

    def _extract(self, gray: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        h, w = gray.shape[:2]
        ph, pw = math.ceil(h / 8) * 8, math.ceil(w / 8) * 8
        padded = np.zeros((ph, pw), np.float32)
        padded[:h, :w] = gray.astype(np.float32) / 255.0
        out = self._sp.run(None, {self._sp.get_inputs()[0].name: padded[None, None]})
        kpts = np.asarray(out[0], dtype=np.float32).reshape(-1, 2)
        desc = np.asarray(out[2], dtype=np.float32).reshape(len(kpts), -1)
        x, y = kpts[:, 0], kpts[:, 1]
        keep = (x >= 0) & (y >= 0) & (x <= w - 1) & (y <= h - 1)
        if mask is not None:
            xi, yi = np.clip(x.round().astype(int), 0, w - 1), np.clip(y.round().astype(int), 0, h - 1)
            keep &= mask[yi, xi] > 0
        return kpts[keep], desc[keep]

    def match(self, current: np.ndarray, reference: np.ndarray,
              mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        k0, d0 = self._extract(current, mask)
        k1, d1 = self._extract(reference, None)
        if len(k0) == 0 or len(k1) == 0:
            return _EMPTY
        h, w = current.shape[:2]
        shift, scale = np.array([w / 2, h / 2], np.float32), max(w, h) / 2
        values = {"kpts0": ((k0 - shift) / scale)[None], "kpts1": ((k1 - shift) / scale)[None],
                  "desc0": d0[None], "desc1": d1[None]}
        names = [i.name for i in self._lg.get_inputs()]
        if set(names) != set(values):  # 이름이 다르면 위 순서(kpts0, kpts1, desc0, desc1)로 넣는다
            values = dict(zip(names, values.values(), strict=True))
        out = self._lg.run(None, values)
        matches = np.asarray(out[0]).astype(np.int64)
        if matches.size == 0:
            return _EMPTY
        matches = matches.reshape(-1, matches.shape[-1])[:, -2:]
        ok = (matches[:, 0] >= 0) & (matches[:, 0] < len(k0)) & (matches[:, 1] >= 0) & (matches[:, 1] < len(k1))
        matches = matches[ok]
        return k0[matches[:, 0]], k1[matches[:, 1]]


def _ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    x, y = a[mask].astype(np.float64), b[mask].astype(np.float64)
    if x.size < 2:
        return -1.0
    x, y = x - x.mean(), y - y.mean()
    denom = math.sqrt(float((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / denom) if denom > 0 else -1.0


def _ecc(current: np.ndarray, reference: np.ndarray, homography: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """현재 → 기준 호모그래피를 밝기 상관(ECC)으로 다듬는다. 작은 배율부터 키운다."""
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERATIONS, 1e-6)
    h = homography
    for sc in ECC_SCALES:
        s = np.diag([sc, sc, 1.0])
        cur = cv2.resize(current, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc != 1.0 else current
        ref = cv2.resize(reference, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc != 1.0 else reference
        if min(cur.shape[:2]) < 16:
            continue
        m = None  # 반쪽 사진: 현재 이미지에서 사진이 덮은 곳만 본다
        if mask is not None:
            m = cv2.resize(mask, (cur.shape[1], cur.shape[0]), interpolation=cv2.INTER_NEAREST)
        # ECC의 warp는 기준(template) px → 현재(input) px
        warp = np.linalg.inv(s @ h @ np.linalg.inv(s)).astype(np.float32)
        _, warp = cv2.findTransformECC(ref, cur, warp, cv2.MOTION_HOMOGRAPHY, criteria, m, 3)
        h = np.linalg.inv(np.linalg.inv(s) @ warp.astype(np.float64) @ s)
    return h / h[2, 2]


@dataclass(frozen=True)
class Refiner:
    """matchers를 차례로 해 보고 처음 받아들여진 보정을 쓴다. 모두 안 되면 윤곽 결과 그대로."""

    matchers: tuple[Matcher, ...]
    min_inliers: int = MIN_INLIERS

    @property
    def names(self) -> str:
        return "→".join(m.name for m in self.matchers)

    def refine(self, image: np.ndarray, base: Alignment, reference: np.ndarray) -> Alignment:
        if reference.shape[:2] != base.rectified.shape[:2]:
            log.info("기준 이미지 크기가 달라 윤곽 결과를 씀(%s ≠ %s)", reference.shape[:2], base.rectified.shape[:2])
            return base
        cur = _gray(base.rectified)
        ref = _gray(reference)
        mask = base.coverage() if base.partial else None
        for matcher in self.matchers:
            try:
                result = self._try(matcher, image, base, cur, ref, mask)
            except Exception as exc:  # noqa: BLE001 - 모델·OpenCV 오류면 다음 방식(결국 윤곽)으로
                log.warning("특징점 정합(%s) 실패: %s", matcher.name, exc)
                continue
            if isinstance(result, Alignment):
                return result
            log.info("특징점 정합(%s) 버림: %s", matcher.name, result)
        return base

    def _try(self, matcher: Matcher, image: np.ndarray, base: Alignment, cur: np.ndarray, ref: np.ndarray,
             mask: np.ndarray | None) -> Alignment | str:
        p_cur, p_ref = matcher.match(cur, ref, mask)
        if len(p_cur) < self.min_inliers:
            return f"맞는 점 {len(p_cur)}개(인라이어 부족)"
        h_r, inliers = cv2.findHomography(p_cur, p_ref, cv2.RANSAC, RANSAC_PX)
        count = int(inliers.sum()) if inliers is not None else 0
        if h_r is None or count < self.min_inliers:
            return f"인라이어 {count}개(인라이어 부족)"
        h_r = h_r / h_r[2, 2]
        try:
            h_r = _ecc(cur, ref, h_r, mask)
        except cv2.error as exc:
            log.info("ECC 다듬기 실패(RANSAC 결과 사용): %s", str(exc).splitlines()[0] if str(exc) else exc)
        size = (base.rectified.shape[1], base.rectified.shape[0])
        std = target_corners(size).reshape(-1, 1, 2).astype(np.float64)
        shift = float(np.linalg.norm(cv2.perspectiveTransform(std, h_r) - std, axis=2).max())
        if shift > MAX_SHIFT_RATIO * max(size):
            return f"보정이 너무 큼({shift:.1f}px)"
        homography = h_r @ base.homography
        homography = homography / homography[2, 2]
        rectified = cv2.warpPerspective(image, homography, size, flags=cv2.INTER_LINEAR)
        refined = replace(base, homography=homography, rectified=rectified)
        overlap = (ref > 0) & (refined.coverage() > 0) & (base.coverage() > 0)
        before, after = _ncc(cur, ref, overlap), _ncc(_gray(rectified), ref, overlap)
        if after < before - NCC_SLACK:
            return f"기준과의 상관이 나빠짐({before:.4f} → {after:.4f})"
        corners = cv2.perspectiveTransform(std, np.linalg.inv(homography)).reshape(4, 2).astype(np.float32)
        error = base.error_px if base.contour is None else border_error_px(homography, base.contour, size)
        log.info("특징점 정합(%s): 인라이어 %d, 보정 %.2fpx, 상관 %.4f → %.4f", matcher.name, count, shift, before, after)
        return replace(refined, corners=corners, error_px=error, method=matcher.method, align_method=matcher.name)


def _gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


SessionFactory = Callable[[Path], Session]


def _ort_session(threads: int) -> SessionFactory:
    def make(path: Path) -> Session:
        import onnxruntime as ort

        options = ort.SessionOptions()
        if threads > 0:
            options.intra_op_num_threads = threads
        return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])

    return make


def load_refiner(env: Mapping[str, str], make_session: SessionFactory | None = None) -> Refiner | None:
    """TRACE_ALIGN=features면 Refiner, 아니면 None. 모델 파일이 없거나 열리지 않으면 ORB만 쓴다(시작을 막지 않음)."""
    if align_mode(env) != "features":
        return None
    matchers: list[Matcher] = []
    folder = env.get("TRACE_MODEL_DIR")
    files = [Path(folder).expanduser() / n for n in (SUPERPOINT_FILE, LIGHTGLUE_FILE)] if folder else []
    if files and all(f.is_file() for f in files):
        threads = env.get("TRACE_DETECT_THREADS") or "0"
        try:
            make = make_session or _ort_session(int(threads) if threads.isdigit() else 0)
            matchers.append(LightGlueMatcher(make(files[0]), make(files[1])))
        except Exception as exc:  # noqa: BLE001 - onnxruntime 없음·파일 깨짐이면 ORB로
            log.warning("SuperPoint·LightGlue 모델을 열지 못해 ORB로 정합을 다듬음: %s", exc)
    else:
        log.info("모델 폴더에 %s·%s가 없어 ORB로 정합을 다듬음", SUPERPOINT_FILE, LIGHTGLUE_FILE)
    matchers.append(OrbMatcher())
    refiner = Refiner(tuple(matchers))
    log.info("정합: 윤곽 4점 + 특징점(%s)", refiner.names)
    return refiner


__all__ = ["ALIGN_MODES", "LIGHTGLUE_FILE", "METHOD_LIGHTGLUE", "SUPERPOINT_FILE", "LightGlueMatcher", "Matcher",
           "OrbMatcher", "Refiner", "align_mode", "load_refiner", "reference_key"]
