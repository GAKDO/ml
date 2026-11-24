"""이 파일이 확인하는 것: 흠집 검사 워커(inspect-worker, ML2)의 핵심 처리.
촬영 세트 업로드 메시지 하나를 받아 → 사진마다 면을 반듯하게 펼치고(정합) → 면마다 펼친 그림(텍스처)을 저장하고
→ 결과 메시지(trace.inspect.completed)를 만든다. 웹의 3D 리포트가 이 텍스처를 노트북 모형에 입힌다.
부스 사진은 흠집 탐지를 부스 Jetson이 이미 했으므로 여기서는 정합만, 폰 사진은 여기서 흠집도 찾는다.

ML2 워커 핵심: 세트 → 펼친 이미지·텍스처·trace.inspect.completed (스토리 1·3).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import cv2
import numpy as np
import pytest
from fakes import MemoryS3
from inspect_helpers import DIMS, blank_photo, contract_validator, pid, synthetic_jpeg, uploaded_event

from trace_ml.inspect.detect import NullDetector, PhotoDetection, measure
from trace_ml.inspect.faces import face_size_px
from trace_ml.inspect.synthetic import booth_half_corners, face_texture, render
from trace_ml.inspect.worker import InvalidMessage, WorkerConfig, parse_request, process, stable_id
from trace_ml.store import Store

CONFIG = WorkerConfig(dims=DIMS)
NOW = lambda: datetime(2026, 11, 9, 5, 0, tzinfo=UTC)


# 테스트 도구: 메시지에 적힌 사진들을 가짜 저장소에 넣어 둔다.
def _store_with(event: dict, bodies: dict[str, bytes]) -> tuple[Store, MemoryS3]:
    s3 = MemoryS3()
    for photo in event["photos"]:
        s3.objects[("trace-captures", photo["objectKey"])] = (bodies[photo["photoId"]], {})
    return Store(s3), s3


# 테스트 도구: 부스가 찍은 윗면(TOP)·앞 옆면(SF) 사진 두 장짜리 세트를 만든다.
def _booth_set() -> tuple[dict, dict[str, bytes]]:
    top, _ = synthetic_jpeg("TOP", seed=1)
    sf, _ = synthetic_jpeg("SF", seed=2)
    bodies = {pid("p1"): top, pid("p2"): sf}
    return uploaded_event([(pid("p1"), "TOP", top), (pid("p2"), "SF", sf)]), bodies


# 테스트 도구: 가짜 흠집 탐지기. 정해 둔 흠집을 돌려주고 몇 번 불렸는지 센다.
class FakeDetector:
    model_version = "yolo11s-seg-test"

    def __init__(self, polygon: list[list[float]]) -> None:
        self.polygon = polygon
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[PhotoDetection]:
        self.calls += 1
        return [PhotoDetection("scratch", self.polygon, 0.9), PhotoDetection("unknown", self.polygon, 0.5)]


# [상황] 부스 세트(윗면·앞 옆면)를 처리한다.
# [기대] 사진마다 윤곽(CONTOUR) 정합 결과가 나오고, 면마다 텍스처가 저장되며(윗면 1216×860, 옆면 1216×44),
#        흠집 목록은 비어 있다(부스가 따로 올림). 모델 버전은 contour-v1.
def test_부스_세트는_사진마다_CONTOUR_정합과_면마다_텍스처를_내고_흠집은_비어_있다() -> None:
    event, bodies = _booth_set()
    store, s3 = _store_with(event, bodies)
    out = process(parse_request(event), store, CONFIG, now=NOW)
    assert [a["photoId"] for a in out["alignments"]] == [pid("p1"), pid("p2")]
    assert all(a["method"] == "CONTOUR" and len(a["homography"]) == 3 for a in out["alignments"])
    assert out["defects"] == [] and out["failedPhotos"] == []
    assert out["textureKeys"] == {"TOP": f"textures/{pid('set1')}/TOP.png", "SF": f"textures/{pid('set1')}/SF.png"}
    assert out["modelVersion"] == "contour-v1"
    rect = s3.objects[("trace-derived", f"rectified/{pid('p1')}.png")][0]
    img = cv2.imdecode(np.frombuffer(rect, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (860, 1216)
    tex = s3.objects[("trace-derived", out["textureKeys"]["SF"])][0]
    assert cv2.imdecode(np.frombuffer(tex, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (44, 1216)


# [상황] 처리 결과 메시지를 contracts의 형식으로 검사한다.
# [기대] 오류가 없다(다른 서버가 그대로 받을 수 있음).
def test_결과_이벤트는_contracts_trace_inspect_completed_스키마를_통과한다() -> None:
    validator = contract_validator("trace.inspect.completed.schema.json")
    event, bodies = _booth_set()
    store, _ = _store_with(event, bodies)
    out = process(parse_request(event), store, CONFIG, now=NOW)
    errors = sorted(validator.iter_errors(json.loads(json.dumps(out))), key=str)
    assert not errors, errors


# [상황] 테스트가 쓰는 촬영 업로드 메시지 자체를 contracts 형식으로 검사한다.
# [기대] 오류가 없다(테스트 데이터가 실제와 같은 모양임을 보장).
def test_입력_예시_이벤트가_contracts_스키마를_통과한다() -> None:
    validator = contract_validator("trace.capture.uploaded.schema.json")
    event, _ = _booth_set()
    assert not list(validator.iter_errors(event))


# [상황] 세트 중 바닥면(BOT) 사진에 노트북 윤곽이 없다.
# [기대] 그 사진만 "윤곽 4점을 찾지 못함" 사유로 실패 목록에 들어가고 나머지는 정상 처리된다.
def test_윤곽이_없는_사진은_사유와_함께_failedPhotos에_있고_나머지는_정합된다() -> None:
    top, _ = synthetic_jpeg("TOP", seed=1)
    blank = blank_photo()
    event = uploaded_event([(pid("p1"), "TOP", top), (pid("p2"), "BOT", blank)])
    store, _ = _store_with(event, {pid("p1"): top, pid("p2"): blank})
    out = process(parse_request(event), store, CONFIG, now=NOW)
    assert [a["photoId"] for a in out["alignments"]] == [pid("p1")]
    assert out["failedPhotos"] == [{"photoId": pid("p2"), "reason": "윤곽 4점을 찾지 못함(윤곽 없음)"}]
    assert "BOT" not in out["textureKeys"]


# [상황] 같은 업로드 메시지가 두 번 온다(메시지 큐 재전달).
# [기대] 결과 메시지 ID와 내용이 똑같다(받는 쪽이 중복으로 알아보고 한 번만 반영).
def test_같은_이벤트가_두_번_오면_결과_eventId와_내용이_같다() -> None:
    event, bodies = _booth_set()
    store, _ = _store_with(event, bodies)
    first = process(parse_request(event), store, CONFIG, now=NOW)
    second = process(parse_request(event), store, CONFIG, now=NOW)
    assert first == second
    assert len(first["eventId"]) == 21 and first["eventId"] == stable_id("trace.inspect.completed", pid("evt1"))


# [상황] 저장소의 사진 내용이 메시지에 적힌 해시와 다르다(바뀌거나 망가짐).
# [기대] "해시 불일치" 사유로 그 사진을 실패 처리한다(증거 사진이 바뀐 채 검사되지 않게).
def test_원본_사진_SHA256이_다르면_해시_불일치로_실패한다() -> None:
    event, bodies = _booth_set()
    store, s3 = _store_with(event, bodies)
    key = event["photos"][0]["objectKey"]
    s3.objects[("trace-captures", key)] = (blank_photo(), {})
    out = process(parse_request(event), store, CONFIG, now=NOW)
    assert out["failedPhotos"][0]["photoId"] == pid("p1") and "해시 불일치" in out["failedPhotos"][0]["reason"]


# [상황] 폰으로 찍은 세트를 처리하고, 탐지기가 면 왼쪽 위 근처 긁힘 하나와 모르는 종류 하나를 낸다.
# [기대] 모르는 종류는 버리고, 긁힘은 펼친 이미지 좌표의 다각형과 mm 단위 길이·넓이로 나온다.
#        모델 버전은 "탐지 모델+contour-v1".
def test_폰_세트의_흠집은_표준_px_다각형과_mm_길이_넓이로_나온다() -> None:
    body, corners = synthetic_jpeg("TOP", seed=4)
    event = uploaded_event([(pid("p1"), "TOP", body)], need_detection=True)
    store, _ = _store_with(event, {pid("p1"): body})
    # 사진에서 면의 왼쪽 위 꼭짓점 근처 작은 사각형(사진 px)
    x, y = corners[0] + 20
    detector = FakeDetector([[x, y], [x + 10, y], [x + 10, y + 4], [x, y + 4]])
    out = process(parse_request(event), store, CONFIG, detector=detector, now=NOW)
    assert detector.calls == 1
    assert len(out["defects"]) == 1  # 모르는 클래스는 버린다
    d = out["defects"][0]
    assert d["class"] == "scratch" and d["photoId"] == pid("p1") and len(d["defectId"]) == 21
    xs = [p[0] for p in d["polygon"]]
    assert 0 < min(xs) < 100 and max(xs) < 150  # 표준 px: 왼쪽 위 근처
    # 길이 = 표준 px 긴 변 / 4
    assert d["lengthMm"] == pytest.approx((max(xs) - min(xs)) / 4, abs=0.5) and d["areaMm2"] > 0
    assert out["modelVersion"] == "yolo11s-seg-test+contour-v1"


# [상황] 부스 세트를 처리한다.
# [기대] 흠집 탐지기를 한 번도 부르지 않는다.
def test_부스_세트는_탐지기를_부르지_않는다() -> None:
    event, bodies = _booth_set()
    store, _ = _store_with(event, bodies)
    detector = FakeDetector([[0, 0], [1, 0], [1, 1]])
    process(parse_request(event), store, CONFIG, detector=detector, now=NOW)
    assert detector.calls == 0


# [상황] 탐지 모델이 아직 없는 상태에서 폰 세트를 처리한다.
# [기대] 흠집은 비어 있고 모델 버전에 "detector-pending"(모델 대기)이 적힌다.
def test_모델이_정해지기_전_폰_탐지는_빈_결과와_detector_pending_버전이다() -> None:
    body, _ = synthetic_jpeg("TOP", seed=4)
    event = uploaded_event([(pid("p1"), "TOP", body)], need_detection=True)
    store, _ = _store_with(event, {pid("p1"): body})
    out = process(parse_request(event), store, CONFIG, detector=NullDetector(), now=NOW)
    assert out["defects"] == [] and out["modelVersion"] == "detector-pending+contour-v1"


# [상황] 펼친 이미지에서 흠집 다각형의 길이·넓이를 잰다(1px = 0.25mm).
# [기대] 정해진 값이 나오고, 점이 2개뿐인 다각형은 잴 수 없어 비어 있다.
def test_흠집_길이와_넓이는_1px_0_25mm로_잰다() -> None:
    length, area = measure([[0, 0], [40, 0], [40, 8], [0, 8]])
    assert length == 10.0 and area == 20.0
    assert measure([[0, 0], [1, 1]]) == (None, None)


# [상황] 부스 카메라 2대가 윗면의 앞 반·뒤 반을 각각 찍었다.
# [기대] 두 사진을 이어 붙여 빈 곳 없는 윗면 텍스처 하나를 만들고, 원래 무늬와 거의 같다.
def test_부스_앞_반과_뒤_반_사진을_이어_TOP_텍스처를_만든다() -> None:
    size = face_size_px("TOP", DIMS)
    texture = face_texture(size, 6)
    rng = np.random.default_rng(6)
    photo = (1280, 720)
    bodies = {}
    for name, half in (("front", "front"), ("back", "back")):
        shot = render(texture, booth_half_corners(size, photo, rng, half, tilt=1.0), photo, seed=7)
        bodies[pid(name)] = cv2.imencode(".png", shot.image)[1].tobytes()
    event = uploaded_event([(pid("front"), "TOP", bodies[pid("front")]), (pid("back"), "TOP", bodies[pid("back")])])
    event["photos"][1]["camera"] = "CAM2"
    store, s3 = _store_with(event, bodies)
    out = process(parse_request(event), store, CONFIG, now=NOW)
    assert len(out["alignments"]) == 2
    tex = cv2.imdecode(np.frombuffer(s3.objects[("trace-derived", out["textureKeys"]["TOP"])][0], np.uint8),
                       cv2.IMREAD_GRAYSCALE)
    # 이어 붙이면 면 전체가 채워진다(배경색 30 근처인 빈 곳이 거의 없음)
    assert (tex < 15).mean() < 0.02
    assert np.abs(tex.astype(int) - cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY).astype(int)).mean() < 25


# [상황] 형식이 틀린 업로드 메시지들.
# [기대] "잘못된 메시지(InvalidMessage)"로 분류된다(DLQ로 보내기 위함).
@pytest.mark.parametrize("broken", [
    [], {"schemaVersion": "2.0"}, {"schemaVersion": "1.0", "photos": []},
    {"schemaVersion": "1.0", "photos": [{"photoId": "x", "face": "NOPE"}]},
])
def test_형식이_틀린_메시지는_InvalidMessage다(broken: object) -> None:
    with pytest.raises(InvalidMessage):
        parse_request(broken)


# [상황] 메시지에 약속에 없는 필드가 더 있다.
# [기대] 무시하고 정상으로 읽는다(계약이 늘어나도 워커가 깨지지 않게).
def test_모르는_필드는_무시한다() -> None:
    event, _ = _booth_set()
    event["newField"] = 1
    event["photos"][0]["extra"] = "x"
    assert parse_request(event).capture_set_id == pid("set1")


GRAM = {"modelCode": "GRAM16-2024", "dimensions": {"widthMm": 357.7, "depthMm": 250.0, "baseHMm": 11.0, "lidHMm": 5.0}}


# [상황] 메시지에 기종 치수(다른 노트북)가 실려 있다.
# [기대] 그 치수로 펼친다(윗면 1431×1000, 옆면 1431×64).
def test_이벤트에_device가_있으면_그_기종_치수로_펼친다() -> None:
    event, bodies = _booth_set()
    event["device"] = GRAM
    store, s3 = _store_with(event, bodies)
    out = process(parse_request(event), store, CONFIG, now=NOW)
    rect = s3.objects[("trace-derived", f"rectified/{pid('p1')}.png")][0]
    assert cv2.imdecode(np.frombuffer(rect, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (1000, 1431)
    tex = s3.objects[("trace-derived", out["textureKeys"]["SF"])][0]
    assert cv2.imdecode(np.frombuffer(tex, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (64, 1431)


# [상황] 메시지에 기종 치수가 없다.
# [기대] 기본 기종(맥북 에어 13 M2)으로 펼친다.
def test_device가_없으면_기본_기종으로_펼친다() -> None:
    event, _ = _booth_set()
    assert parse_request(event).dims is None


# [상황] 기종 치수가 실린 메시지를 contracts 형식으로 검사한다.
# [기대] 오류가 없다.
def test_device가_실린_입력_이벤트도_contracts_스키마를_통과한다() -> None:
    validator = contract_validator("trace.capture.uploaded.schema.json")
    event, _ = _booth_set()
    event["device"] = GRAM
    assert not list(validator.iter_errors(event))


# [상황] 기종 치수 형식이 틀렸다(값 누락·음수·글자 등).
# [기대] 잘못된 메시지로 분류된다.
@pytest.mark.parametrize("device", [
    None, "GRAM",
    {"modelCode": "X", "dimensions": {"widthMm": 300, "depthMm": 200, "baseHMm": 500, "lidHMm": 1}}, {"modelCode": "GRAM16-2024"}, {"modelCode": "X", "dimensions": {"widthMm": 0, "depthMm": 1,
                                                                              "baseHMm": 1, "lidHMm": 1}},
    {"modelCode": "X", "dimensions": {"widthMm": "357", "depthMm": 1, "baseHMm": 1, "lidHMm": 1}},
])
def test_device_형식이_틀리면_InvalidMessage다(device: object) -> None:
    event, _ = _booth_set()
    event["device"] = device
    with pytest.raises(InvalidMessage):
        parse_request(event)


# [상황] 0.1mm처럼 아주 작은 치수로 면 크기를 계산한다.
# [기대] 최소 1px로 잡혀 0px 이미지가 생기지 않는다.
def test_아주_얇은_치수도_0px가_되지_않는다() -> None:
    from trace_ml.inspect.faces import DeviceDims

    assert face_size_px("SF", DeviceDims("X", width=0.1, depth=0.1, base_h=0.05, lid_h=0.05)) == (1, 1)


# 테스트 도구: 머리말만 거대한 크기로 적힌 가짜 PNG(열면 메모리를 다 쓰는 공격용 사진 흉내).
def _bomb_png(width: int, height: int) -> bytes:
    """헤더만 큰 PNG(IHDR만 맞고 본문은 비어 있음). 디코딩하면 안 된다."""
    import struct
    import zlib
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    chunk = b"IHDR" + ihdr
    return (b"\x89PNG\r\n\x1a\n" + struct.pack(">I", len(ihdr)) + chunk
            + struct.pack(">I", zlib.crc32(chunk)) + b"\x00\x00\x00\x00IEND\xaeB`\x82")


# [상황] 머리말에 터무니없이 큰 크기가 적힌 사진이 섞여 있다.
# [기대] 열어 보지도 않고 "너무 큼" 사유로 실패 처리한다(서버 메모리 고갈 방지).
def test_헤더상_너무_큰_사진은_디코딩하지_않고_failedPhotos에_넣는다(monkeypatch: pytest.MonkeyPatch) -> None:
    from trace_ml.inspect import worker as worker_mod
    big = _bomb_png(32000, 32000)
    top, _ = synthetic_jpeg("TOP", seed=1)
    event = uploaded_event([(pid("p1"), "TOP", big), (pid("p2"), "TOP", top)])
    store, _ = _store_with(event, {pid("p1"): big, pid("p2"): top})
    decoded: list[int] = []
    real = worker_mod.cv2.imdecode
    monkeypatch.setattr(worker_mod.cv2, "imdecode", lambda buf, flag: decoded.append(len(buf)) or real(buf, flag))
    out = process(parse_request(event), store, CONFIG, now=NOW)
    assert [f["photoId"] for f in out["failedPhotos"]] == [pid("p1")]
    assert "너무 큼" in out["failedPhotos"][0]["reason"]
    assert decoded == [len(top)]  # 큰 사진은 imdecode에 가지 않았다


# [상황] JPEG·PNG·GIF·잘린 JPEG의 머리말에서 크기를 읽는다.
# [기대] JPEG·PNG는 크기를 읽고, 그 밖은 "모름(None)"이다.
def test_JPEG_PNG_헤더에서_크기를_읽고_다른_형식은_None이다() -> None:
    from trace_ml.inspect.worker import image_size
    ok, jpg = cv2.imencode(".jpg", np.zeros((30, 50, 3), np.uint8))
    assert ok
    assert image_size(jpg.tobytes()) == (50, 30)
    ok, png = cv2.imencode(".png", np.zeros((7, 9, 3), np.uint8))
    assert image_size(png.tobytes()) == (9, 7)
    assert image_size(b"GIF89a....") is None
    assert image_size(b"\xff\xd8\xff") is None
