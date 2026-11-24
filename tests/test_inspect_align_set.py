"""이 파일이 확인하는 것: "trace-ml align set" 명령. 촬영 세트(한 번 찍은 사진 묶음)의 사진마다
면을 반듯하게 펼치는 정합이 됐는지, 안 됐으면 왜인지 적는다(11/9 옆면 판정 불가 원인 조사용).

trace-ml align set: 세트 사진마다 정합 결과와 실패 사유(11/9 옆면 판정 불가 조사).
"""

from __future__ import annotations

import cv2
import numpy as np
from fakes import MemoryS3
from inspect_helpers import DIMS, blank_photo, pid, synthetic_jpeg

from trace_ml.inspect.cli import align_set
from trace_ml.store import Store


# 테스트 도구: 정합이 실패하는 상황을 재현한 사진(옆면 띠 아래 밝은 책상이 사진 테두리까지 닿음).
def _desk_photo() -> bytes:
    """어두운 배경판 아래 밝은 책상이 사진 테두리 세 변에 닿고, 그 위에 옆면 띠가 놓인 사진(정합 실패 재현)."""
    img = np.full((720, 1280, 3), 30, np.uint8)
    img[430:] = 200
    img[400:430, 240:1040] = 190
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


# [상황] 한 세트에 윗면(TOP), 앞 옆면(SF, 책상 때문에 실패), 왼쪽 옆면(SL, 빈 사진)이 있고
#        다른 세트 사진도 섞여 있다.
# [기대] 이 세트 사진 3장만 결과가 나오고, 윗면은 성공, 앞 옆면은 "면이 사진 밖으로 나감" 사유로 실패,
#        빈 사진도 실패. 확인용 마스크·펼친 이미지 파일이 남는다.
def test_세트_사진마다_정합_결과와_실패_사유를_적고_마스크를_남긴다(tmp_path) -> None:
    s3 = MemoryS3()
    store = Store(s3)
    set_id = pid(7)
    top, _ = synthetic_jpeg("TOP")
    store.put("trace-captures", f"captures/{set_id}/TOP-L1-CAM1.png", top)
    store.put("trace-captures", f"captures/{set_id}/SF-L1-CAM1.png", _desk_photo())
    store.put("trace-captures", f"captures/{set_id}/SL-L1-CAM1.png", blank_photo())
    store.put("trace-captures", f"captures/{pid(8)}/SF-L1-CAM1.png", top)  # 다른 세트는 읽지 않는다

    rows = align_set(store, set_id, DIMS, out_dir=str(tmp_path))

    by = {r["photo"]: r for r in rows}
    assert set(by) == {"TOP-L1-CAM1.png", "SF-L1-CAM1.png", "SL-L1-CAM1.png"}
    assert by["TOP-L1-CAM1.png"]["ok"] is True
    assert by["SF-L1-CAM1.png"] == {"photo": "SF-L1-CAM1.png", "face": "SF", "light": "L1", "camera": "CAM1", "method": "CONTOUR",
                                    "ok": False, "reason": "면이 사진 밖으로 나감(두 변 이상 잘림)"}
    assert by["SL-L1-CAM1.png"]["ok"] is False
    assert (tmp_path / "SF-L1-CAM1_png-mask.png").exists()
    assert (tmp_path / "TOP-L1-CAM1_png-rect.png").exists()


# [상황] 윗면·앞 옆면 사진이 있는 세트에서 앞 옆면(SF)만 고른다.
# [기대] 앞 옆면 결과만 나온다.
def test_면을_고르면_그_면_사진만_돌린다() -> None:
    store = Store(MemoryS3())
    set_id = pid(9)
    top, _ = synthetic_jpeg("TOP")
    store.put("trace-captures", f"captures/{set_id}/TOP-L1-CAM1.png", top)
    store.put("trace-captures", f"captures/{set_id}/SF-L1-CAM1.png", _desk_photo())

    rows = align_set(store, set_id, DIMS, faces={"SF"})

    assert [r["face"] for r in rows] == ["SF"]
