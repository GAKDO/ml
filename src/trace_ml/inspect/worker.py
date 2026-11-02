"""trace.capture.uploaded 하나를 처리해 trace.inspect.completed 본문을 만든다(브로커와 무관한 핵심).

사진마다: 받기 → SHA-256 대조 → 윤곽 4점 정합 → 펼친 이미지 올리기 → (폰이면) 탐지.
면마다 오차가 가장 작은 펼친 이미지를 텍스처로 올린다.
"""

from __future__ import annotations

import base64
import hashlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import cv2
import numpy as np

from ..store import Store, StoreError
from .align import AlignFailure, Alignment, align_photo
from .detect import Detector, NullDetector, to_standard_defect
from .faces import FACES, DeviceDims, face_size_px

log = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
ALIGN_VERSION = "contour-v1"
PUBLIC_ID_LEN = 21


class InvalidMessage(ValueError):
    """형식이 틀린 메시지. 다시 해도 소용없어 바로 DLQ로 보낸다."""


@dataclass(frozen=True)
class PhotoRef:
    photo_id: str
    face: str
    light: str
    camera: str
    object_key: str
    sha256: str


@dataclass(frozen=True)
class InspectRequest:
    event_id: str
    rental_id: str
    checkpoint_id: str
    capture_set_id: str
    need_detection: bool
    photos: tuple[PhotoRef, ...]
    dims: DeviceDims | None = None  # 이벤트 device(기종·치수). 없으면 WorkerConfig.dims(기본 기종)


@dataclass(frozen=True)
class WorkerConfig:
    dims: DeviceDims
    captures_bucket: str = "trace-captures"
    derived_bucket: str = "trace-derived"


class Progress(Protocol):
    def report(self, req: InspectRequest, stage: str, done: int, total: int) -> None: ...


class NoProgress:
    def report(self, req: InspectRequest, stage: str, done: int, total: int) -> None:
        return None


def stable_id(*parts: str) -> str:
    """같은 입력이면 같은 21자 ID(NanoID 문자 집합). 같은 이벤트가 다시 와도 결과 ID가 같게 한다."""
    digest = hashlib.sha256("\x1f".join(parts).encode()).digest()
    return base64.urlsafe_b64encode(digest).decode()[:PUBLIC_ID_LEN]


def _req_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise InvalidMessage(f"{key} 없음")
    return value


def _parse_device(raw: Any) -> DeviceDims:
    """이벤트 device{modelCode, dimensions{widthMm, depthMm, baseHMm, lidHMm}}. null·객체 아님은 형식 오류."""
    if not isinstance(raw, dict) or not isinstance(raw.get("dimensions"), dict):
        raise InvalidMessage("device 형식이 아님")
    code = _req_str(raw, "modelCode")
    dims = raw["dimensions"]
    values: list[float] = []
    # 범위는 contracts events/envelope.schema.json $defs/DeviceModel과 같다(폭·깊이 1000, 높이 100 이하)
    for key, upper in (("widthMm", 1000), ("depthMm", 1000), ("baseHMm", 100), ("lidHMm", 100)):
        value = dims.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= upper:
            raise InvalidMessage(f"device.dimensions.{key}가 올바르지 않음: {value!r}")
        values.append(float(value))
    return DeviceDims(code, width=values[0], depth=values[1], base_h=values[2], lid_h=values[3])


def parse_request(data: Any) -> InspectRequest:
    """필요한 키만 읽는다(모르는 필드는 무시, README '너그러운 소비자')."""
    if not isinstance(data, dict):
        raise InvalidMessage("JSON 객체가 아님")
    if not str(data.get("schemaVersion", "")).startswith("1."):
        raise InvalidMessage(f"모르는 schemaVersion: {data.get('schemaVersion')}")
    raw_photos = data.get("photos")
    if not isinstance(raw_photos, list) or not raw_photos:
        raise InvalidMessage("photos 없음")
    photos = []
    for item in raw_photos:
        if not isinstance(item, dict):
            raise InvalidMessage("photos 항목이 객체가 아님")
        face = _req_str(item, "face")
        if face not in FACES:
            raise InvalidMessage(f"모르는 면: {face}")
        photos.append(PhotoRef(_req_str(item, "photoId"), face, _req_str(item, "light"), _req_str(item, "camera"),
                               _req_str(item, "objectKey"), _req_str(item, "sha256").lower()))
    need = data.get("needDetection")
    if not isinstance(need, bool):
        raise InvalidMessage("needDetection 없음")
    return InspectRequest(_req_str(data, "eventId"), _req_str(data, "rentalId"), _req_str(data, "checkpointId"),
                          _req_str(data, "captureSetId"), need, tuple(photos),
                          _parse_device(data["device"]) if "device" in data else None)  # 키가 없을 때만 기본 기종


def _encode_png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG 만들기 실패")
    return buf.tobytes()


def _matrix(h: np.ndarray) -> list[list[float]]:
    return [[round(float(v), 9) for v in row] for row in h]


@dataclass
class _Done:
    photo: PhotoRef
    alignment: Alignment
    rectified_key: str


def process(req: InspectRequest, store: Store, config: WorkerConfig, detector: Detector | None = None,
            progress: Progress | None = None, now: Callable[[], datetime] | None = None) -> dict[str, Any]:
    """세트 하나를 처리해 trace.inspect.completed 본문을 돌려준다. 저장소 오류는 StoreError로 올린다(다시 시도)."""
    detector = detector or NullDetector()
    progress = progress or NoProgress()
    dims = req.dims or config.dims  # 기종마다 표준 좌표 크기가 다르다(11/2 민서 결정)
    total = len(req.photos)
    done: list[_Done] = []
    failed: list[dict[str, str]] = []
    defects: list[dict[str, Any]] = []
    progress.report(req, "ALIGN", 0, total)
    for index, photo in enumerate(req.photos, start=1):
        body = store.get(config.captures_bucket, photo.object_key)
        if hashlib.sha256(body).hexdigest() != photo.sha256:
            failed.append({"photoId": photo.photo_id, "reason": "해시 불일치: 업로드 때와 다른 사진"})
            progress.report(req, "ALIGN", index, total)
            continue
        image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            failed.append({"photoId": photo.photo_id, "reason": "사진을 읽지 못함(디코딩 실패)"})
            progress.report(req, "ALIGN", index, total)
            continue
        result = align_photo(image, face_size_px(photo.face, dims))
        if isinstance(result, AlignFailure):
            failed.append({"photoId": photo.photo_id, "reason": result.reason})
            progress.report(req, "ALIGN", index, total)
            continue
        key = f"rectified/{photo.photo_id}.png"
        store.put(config.derived_bucket, key, _encode_png(result.rectified), "image/png")
        done.append(_Done(photo, result, key))
        if req.need_detection:
            for n, det in enumerate(detector.detect(image)):
                try:
                    std = to_standard_defect(det, result.homography)
                except ValueError as exc:
                    log.warning("탐지 결과 버림(사진 %s): %s", photo.photo_id, exc)
                    continue
                defects.append({
                    "defectId": stable_id(req.event_id, photo.photo_id, str(n)),
                    "photoId": photo.photo_id, "class": std.cls, "polygon": std.polygon,
                    "confidence": round(std.confidence, 4), "lengthMm": std.length_mm, "areaMm2": std.area_mm2,
                })
        progress.report(req, "ALIGN", index, total)

    texture_keys = _upload_textures(req, done, store, config)
    progress.report(req, "DONE", total, total)
    model_version = ALIGN_VERSION if not req.need_detection else f"{detector.model_version}+{ALIGN_VERSION}"
    clock = now or (lambda: datetime.now(UTC))
    return {
        "schemaVersion": SCHEMA_VERSION,
        "eventId": stable_id("trace.inspect.completed", req.event_id),
        "occurredAt": clock().isoformat(timespec="seconds"),
        "rentalId": req.rental_id,
        "checkpointId": req.checkpoint_id,
        "captureSetId": req.capture_set_id,
        "modelVersion": model_version[:64],
        "alignments": [{
            "photoId": d.photo.photo_id,
            "method": d.alignment.method,
            "homography": _matrix(d.alignment.homography),
            "errorPx": None if d.alignment.error_px is None else round(d.alignment.error_px, 3),
            "rectifiedKey": d.rectified_key,
        } for d in done],
        "defects": defects,
        "textureKeys": texture_keys,
        "failedPhotos": failed,
    }


# 텍스처에 먼저 쓰는 조명: 확산광(L1)·실내(ROOM)가 면 색을 고르게 보여 준다. 스침광은 흠집용이라 뒤로
LIGHT_ORDER = ("L1", "ROOM", "OFF", "L2", "L3", "L4")


def _texture_rank(d: _Done) -> tuple[int, float]:
    light = LIGHT_ORDER.index(d.photo.light) if d.photo.light in LIGHT_ORDER else len(LIGHT_ORDER)
    err = d.alignment.error_px if d.alignment.error_px is not None else float("inf")
    return light, err


def compose_texture(items: list[_Done]) -> np.ndarray:
    """한 면의 펼친 이미지들을 순위대로 겹친다. 앞 순위가 덮지 못한 곳(반쪽 사진의 잘린 쪽)만 다음 것으로 채운다."""
    ordered = sorted(items, key=_texture_rank)
    base = ordered[0].alignment.rectified.copy()
    if not any(d.alignment.partial for d in ordered):
        return base
    covered = ordered[0].alignment.coverage() > 0
    for d in ordered[1:]:
        if covered.all():
            break
        fill = (d.alignment.coverage() > 0) & ~covered
        base[fill] = d.alignment.rectified[fill]
        covered |= fill
    return base


def _upload_textures(req: InspectRequest, done: list[_Done], store: Store, config: WorkerConfig) -> dict[str, str]:
    """면마다 텍스처 하나(3D 트윈). 온전한 사진이면 가장 좋은 펼친 이미지 그대로, 반쪽 사진이면 이어 붙인다."""
    by_face: dict[str, list[_Done]] = {}
    for d in done:
        by_face.setdefault(d.photo.face, []).append(d)
    keys: dict[str, str] = {}
    for face in FACES:
        if face not in by_face:
            continue
        key = f"textures/{req.capture_set_id}/{face}.png"
        store.put(config.derived_bucket, key, _encode_png(compose_texture(by_face[face])), "image/png")
        keys[face] = key
    return keys


__all__ = ["InspectRequest", "InvalidMessage", "NoProgress", "PhotoRef", "Progress", "StoreError", "WorkerConfig",
           "parse_request", "process", "stable_id"]
