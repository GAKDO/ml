"""ML1 기반: 설정(비밀 숨김·버킷 겹침 거절)과 저장소 조건부 쓰기."""

from __future__ import annotations

import pytest
from fakes import World

from trace_ml.labelstudio.config import ConfigError
from trace_ml.settings import load_db, load_minio
from trace_ml.store import AlreadyExists, StoreError

ENV = {"MINIO_ENDPOINT": "http://minio.test", "MINIO_ACCESS_KEY": "team-ak", "MINIO_SECRET_KEY": "team-sk"}


def test_trainer_키가_없으면_팀_키를_쓰고_repr에_비밀이_없다():
    s = load_minio(ENV)
    assert (s.access_key, s.datasets_bucket, s.models_bucket) == ("team-ak", "trace-datasets", "trace-models")
    assert "team-sk" not in repr(s)
    assert load_minio({**ENV, "TRAINER_MINIO_ACCESS_KEY": "t-ak", "TRAINER_MINIO_SECRET_KEY": "t-sk"}).access_key == "t-ak"


def test_출력_버킷이_원본_버킷과_같으면_시작하지_않는다():
    with pytest.raises(ConfigError):
        load_minio({**ENV, "TRACE_DATASETS_BUCKET": "trace-raw"})


def test_빠진_환경_변수는_이름만_알린다():
    with pytest.raises(ConfigError, match="MINIO_ENDPOINT"):
        load_minio({})
    with pytest.raises(ConfigError, match="DB_PASSWORD"):
        load_db({"DB_USERNAME": "u"})
    db = load_db({"DB_USERNAME": "u", "DB_PASSWORD": "secret-pw"})
    assert (db.host, db.port, db.dbname) == ("localhost", 5432, "trace") and "secret-pw" not in repr(db)


def test_같은_키는_조건부_쓰기로_덮어쓸_수_없다():
    w = World()
    w.store.put_new("trace-datasets", "v001/a", b"1")
    with pytest.raises(AlreadyExists):
        w.store.put_new("trace-datasets", "v001/a", b"2")
    assert w.s3.objects[("trace-datasets", "v001/a")][0] == b"1"


def test_목록은_여러_페이지를_이어_받고_없는_키는_없음으로_알린다():
    w = World()
    for i in range(5):
        w.store.put_new("b", f"p/{i}", b"x")
    assert list(w.store.keys("b", "p/")) == [f"p/{i}" for i in range(5)]
    assert w.store.exists("b", "p/1") and not w.store.exists("b", "p/9")
    with pytest.raises(StoreError):
        w.store.get("b", "p/9")
