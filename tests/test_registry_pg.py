"""PgRegistry를 실제 PostgreSQL에서 확인한다. TRACE_ML_TEST_DB=1과 DB 환경 변수가 있을 때만(CI 서비스)."""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest

from trace_ml.registry import DatasetVersion, Device, PgRegistry, VersionExists
from trace_ml.settings import load_db

pytestmark = pytest.mark.skipif(not os.environ.get("TRACE_ML_TEST_DB"), reason="PostgreSQL 없음")


@pytest.fixture()
def reg() -> PgRegistry:
    r = PgRegistry(load_db(os.environ))
    r.init_schema()
    r.init_schema()  # 두 번 돌려도 된다
    with r._connect() as conn:
        conn.execute("TRUNCATE ml.training_run, ml.dataset_version, ml.dataset_device")
    return r


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


def test_기기_정보는_고쳐_쓸_수_있다(reg: PgRegistry):
    reg.save_device(Device("G01", "미등록", "TRAIN"))
    reg.save_device(Device("G01", "MacBook Pro", "TRAIN", "gray"))
    assert reg.devices()["G01"].model_name == "MacBook Pro"


def test_학습_기록은_데이터셋_버전에_묶이고_없는_버전이면_거절된다(reg: PgRegistry):
    from trace_ml.registry import RegistryError, TrainingRun

    v = DatasetVersion("v001", "v001/manifest.json", "a" * 40, {"devices": {}}, datetime(2026, 11, 5, tzinfo=UTC))
    reg.add_version(v, [])
    reg.add_run(TrainingRun("v001", "yolo11s-seg.pt", {"val": {}}, "v001/r/best.pt", "v001/r/best.onnx"))
    [run] = reg.runs()
    assert (run.dataset_version, run.onnx_key) == ("v001", "v001/r/best.onnx") and run.created_at
    with pytest.raises(RegistryError):
        reg.add_run(TrainingRun("v404", "x", {}, "k", "o"))
