"""테스트 도구(테스트 자체는 아님): 흠집 검사(ML2) 테스트가 함께 쓰는 도우미.
  - 합성 사진: 노트북 면 무늬를 비스듬히 찍은 것처럼 만든 사진과 정답 꼭짓점
  - 촬영 업로드 이벤트: 부스·폰이 사진을 올렸다는 메시지(trace.capture.uploaded)를 흉내
  - 계약 검사: contracts 레포의 이벤트 형식으로 우리가 내는 메시지를 검사

ML2 테스트 공용: 합성 사진·세트 이벤트·계약 스키마 검사.
"""

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

# 시험에 쓰는 기종 치수: 맥북 에어 13 M2(가로 304.1 × 세로 215.0 mm, 본체 두께 7.0, 덮개 4.0).
# 면 사진을 펼칠 크기(px)를 이 치수로 정하므로, 합성 사진과 워커가 같은 기종을 써야 결과를 맞춰 볼 수 있다.
DIMS = DEVICE_MODELS["MBA13-M2"]
# 합성 사진 한 장의 크기(가로, 세로 px). 실제 사진보다 작게 해서 GPU 없는 CI에서도 빨리 돈다.
PHOTO = (960, 720)


def synthetic_jpeg(face: str, seed: int = 0) -> tuple[bytes, np.ndarray]:
    """노트북 면(face: TOP·BOT 등)을 비스듬히 찍은 것처럼 만든 합성 사진과 정답 꼭짓점을 돌려준다.

    이름은 jpeg지만 실제로는 PNG(무손실)다. 압축 손실 없이 정합 결과를 정답과 정확히 비교하려고 PNG로 만든다.
    seed가 같으면 늘 같은 무늬·같은 각도의 사진이 나와 테스트 결과가 매번 같다.
    돌려주는 값: (사진 파일 바이트, 사진 속 면의 네 꼭짓점 좌표 = 정합이 찾아야 할 정답)
    """
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
    """테스트용 21자 ID를 만든다. 예) pid("set1") → "xxxxxxxxxxxxxxxxxset1".

    팀 ID 규칙(publicId = NanoID 21자)과 계약 스키마가 21자 ID만 받으므로, 앞을 x로 채워 길이를 맞춘다.
    """
    return str(n).rjust(21, "x")[-21:].replace(" ", "x")


def uploaded_event(photos: list[tuple[str, str, bytes]], need_detection: bool = False,
                   event_id: str | None = None, set_id: str | None = None) -> dict[str, Any]:
    """부스·폰이 사진을 다 올렸을 때 capture-server가 내는 메시지(trace.capture.uploaded)를 흉내 낸다.

    photos: (사진 ID, 면, 사진 바이트) 목록. 사진마다 바이트의 SHA-256(지문)을 실어, 워커가 받은 파일이
            바뀌지 않았는지 확인할 수 있게 한다.
    need_detection: True면 폰 촬영(조명 ROOM·카메라 PHONE)이라 워커가 흠집 탐지까지 해야 한다.
                    False면 부스 촬영(L1·CAM1)이라 부스가 이미 탐지했고 워커는 정합만 한다.
    event_id·set_id: 같은 메시지가 두 번 오는 경우 등을 시험할 때 ID를 직접 정한다.
    """
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
