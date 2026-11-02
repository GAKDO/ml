"""ML2 테스트 공용: 합성 사진·세트 이벤트·계약 스키마 검사."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import urllib.request
from typing import Any

import cv2
import numpy as np
import pytest

from trace_ml.inspect.faces import DEVICE_MODELS, face_size_px
from trace_ml.inspect.synthetic import face_texture, random_corners, render

DIMS = DEVICE_MODELS["MBA13-M2"]
PHOTO = (960, 720)


def synthetic_jpeg(face: str, seed: int = 0) -> tuple[bytes, np.ndarray]:
    """면 사진(PNG, 무손실) 바이트와 정답 꼭짓점."""
    size = face_size_px(face, DIMS)
    rng = np.random.default_rng(seed)
    photo = render(face_texture(size, seed), random_corners(size, PHOTO, rng), PHOTO, seed=seed)
    ok, buf = cv2.imencode(".png", photo.image)
    assert ok
    return buf.tobytes(), photo.corners


def blank_photo() -> bytes:
    ok, buf = cv2.imencode(".png", np.full((PHOTO[1], PHOTO[0], 3), 40, np.uint8))
    assert ok
    return buf.tobytes()


def pid(n: int | str) -> str:
    """테스트용 21자 ID."""
    return str(n).rjust(21, "x")[-21:].replace(" ", "x")


def uploaded_event(photos: list[tuple[str, str, bytes]], need_detection: bool = False,
                   event_id: str | None = None, set_id: str | None = None) -> dict[str, Any]:
    """photos: (photoId, face, 본문). 본문 SHA-256을 이벤트에 싣는다."""
    set_id = set_id or pid("set1")
    return {
        "schemaVersion": "1.0",
        "eventId": event_id or pid("evt1"),
        "occurredAt": "2026-11-09T14:00:00+09:00",
        "rentalId": pid("rental1"),
        "checkpointId": pid("cp1"),
        "captureSetId": set_id,
        "source": "PHONE" if need_detection else "BOOTH",
        "needDetection": need_detection,
        "photos": [{
            "photoId": photo_id, "face": face, "light": "ROOM" if need_detection else "L1",
            "camera": "PHONE" if need_detection else "CAM1",
            "objectKey": f"captures/{set_id}/{face}-{photo_id}.png",
            "sha256": hashlib.sha256(body).hexdigest(),
        } for photo_id, face, body in photos],
    }


# ---- contracts 스키마 검사 ----

_RAW = "https://raw.githubusercontent.com/GAKDO/contracts/{ref}/{path}"
_FILES = ("events/envelope.schema.json", "events/trace.inspect.completed.schema.json",
          "events/trace.capture.uploaded.schema.json", "schemas/codes.gen.schema.json")


def _load_contract(path: str) -> dict[str, Any]:
    local = os.environ.get("TRACE_CONTRACTS_DIR") or str(pathlib.Path(__file__).resolve().parents[2] / "contracts")
    file = pathlib.Path(local) / path
    if file.is_file():
        return json.loads(file.read_text(encoding="utf-8"))
    ref = os.environ.get("TRACE_CONTRACTS_REF", "develop")
    with urllib.request.urlopen(_RAW.format(ref=ref, path=path), timeout=10) as resp:
        return json.loads(resp.read())


def contract_validator(event_file: str) -> Any:
    """contracts 레포(옆 폴더 또는 GitHub develop)의 이벤트 스키마 검사기. 받을 수 없으면 건너뛴다."""
    jsonschema = pytest.importorskip("jsonschema")
    referencing = pytest.importorskip("referencing")
    try:
        docs = {p: _load_contract(p) for p in _FILES}
    except OSError as exc:
        pytest.skip(f"contracts 스키마를 받을 수 없음: {exc}")
    resources = [(d["$id"], referencing.Resource.from_contents(d)) for d in docs.values() if "$id" in d]
    registry = referencing.Registry().with_resources(resources)
    schema = docs[f"events/{event_file}"]
    return jsonschema.Draft202012Validator(schema, registry=registry)
