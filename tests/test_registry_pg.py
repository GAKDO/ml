"""이 파일이 확인하는 것: 데이터셋 버전·기기 배정·학습 기록을 진짜 PostgreSQL에 저장하는 부분(ml 스키마).
기기 배정: 촬영한 노트북(기기 ID)을 학습(TRAIN)·검증(VAL)·시험(TEST) 중 어디에 쓸지 정한 기록.
DB가 있는 CI에서만 돌고(TRACE_ML_TEST_DB=1), 없으면 건너뛴다.

PgRegistry를 실제 PostgreSQL에서 확인한다. TRACE_ML_TEST_DB=1과 DB 환경 변수가 있을 때만(CI 서비스).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest

from trace_ml.registry import DatasetVersion, Device, PgRegistry, VersionExists
from trace_ml.settings import load_db

pytestmark = pytest.mark.skipif(not os.environ.get("TRACE_ML_TEST_DB"), reason="PostgreSQL 없음")


# 테스트 도구: 표를 만들고(두 번 돌려도 안전한지 확인) 비운 뒤 저장소를 준다.
@pytest.fixture()
def reg() -> PgRegistry:
    r = PgRegistry(load_db(os.environ))
    r.init_schema()
    r.init_schema()  # 두 번 돌려도 된다
    with r._connect() as conn:
        conn.execute("TRUNCATE ml.training_run, ml.dataset_version, ml.dataset_device")
    return r


# [상황] 데이터셋 v001과 기기 2대 배정을 함께 기록하고, 같은 v001을 새 기기와 다시 기록한다.
# [기대] 처음엔 기기·버전이 함께 저장된다. 두 번째는 "이미 있는 버전"으로 거절되고,
#        그때 함께 넣으려던 기기(G03)도 남지 않는다(전부 아니면 전무).
def test_새_기기와_버전은_한_번에_기록되고_같은_버전은_고유_제약으로_거절된다(reg: PgRegistry):
    v = DatasetVersion("v001", "v001/manifest.json", "a" * 40,
                       {"devices": {"TRAIN": ["G01"], "VAL": [], "TEST": ["G02"]}}, datetime(2026, 11, 5, tzinfo=UTC))
    reg.add_version(v, [Device("G01", "미등록", "TRAIN"), Device("G02", "MacBook Air", "TEST", "silver")])
    assert reg.devices()["G02"] == Device("G02", "MacBook Air", "TEST", "silver")
    assert reg.used_devices() == {"G01", "G02"}
    assert reg.versions()[0].counts["devices"]["TEST"] == ["G02"]
    with pytest.raises(VersionExists):
        reg.add_version(v, [Device("G03", "미등록", "VAL")])
    assert "G03" not in reg.devices()  # 실패한 트랜잭션의 기기 배정은 남지 않는다


# [상황] 기기 G01의 기종을 "미등록"에서 "MacBook Pro"로 고친다.
# [기대] 고친 값으로 바뀐다.
def test_기기_정보는_고쳐_쓸_수_있다(reg: PgRegistry):
    reg.save_device(Device("G01", "미등록", "TRAIN"))
    reg.save_device(Device("G01", "MacBook Pro", "TRAIN", "gray"))
    assert reg.devices()["G01"].model_name == "MacBook Pro"


# [상황] v001로 학습한 기록을 남기고, 없는 버전(v404)으로도 남겨 본다.
# [기대] v001 기록은 모델 파일 위치·생성 시각과 함께 저장되고, 없는 버전은 거절된다.
def test_학습_기록은_데이터셋_버전에_묶이고_없는_버전이면_거절된다(reg: PgRegistry):
    from trace_ml.registry import RegistryError, TrainingRun

    v = DatasetVersion("v001", "v001/manifest.json", "a" * 40, {"devices": {}}, datetime(2026, 11, 5, tzinfo=UTC))
    reg.add_version(v, [])
    reg.add_run(TrainingRun("v001", "yolo11s-seg.pt", {"val": {}}, "v001/r/best.pt", "v001/r/best.onnx"))
    [run] = reg.runs()
    assert (run.dataset_version, run.onnx_key) == ("v001", "v001/r/best.onnx") and run.created_at
    with pytest.raises(RegistryError):
        reg.add_run(TrainingRun("v404", "x", {}, "k", "o"))
