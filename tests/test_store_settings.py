"""이 파일이 확인하는 것: 데이터셋 빌더(ML1)의 기본 설정과 저장소 동작.
  - 설정: 환경 변수에서 MinIO·DB 접속 정보를 읽고, 비밀 값이 로그에 찍히지 않으며, 잘못된 버킷 설정은 거절
  - 저장소: 이미 있는 파일은 덮어쓰지 않음(조건부 쓰기), 목록을 끝까지 받음
테스트의 키·비밀번호 값은 모두 가짜다.

ML1 기반: 설정(비밀 숨김·버킷 겹침 거절)과 저장소 조건부 쓰기.
"""

from __future__ import annotations

import pytest
from fakes import World

from trace_ml.labelstudio.config import ConfigError
from trace_ml.settings import load_db, load_minio
from trace_ml.store import AlreadyExists, StoreError

ENV = {"MINIO_ENDPOINT": "http://minio.test", "MINIO_ACCESS_KEY": "team-ak", "MINIO_SECRET_KEY": "team-sk"}


# [상황] 학습 전용 키 없이 팀 공용 키만 설정한다. 다음엔 학습 전용 키도 준다.
# [기대] 공용 키와 기본 버킷 이름을 쓰고, 설정을 출력해도 비밀 키가 보이지 않는다.
#        학습 전용 키가 있으면 그걸 쓴다.
def test_trainer_키가_없으면_팀_키를_쓰고_repr에_비밀이_없다():
    s = load_minio(ENV)
    assert (s.access_key, s.datasets_bucket, s.models_bucket) == ("team-ak", "trace-datasets", "trace-models")
    assert "team-sk" not in repr(s)
    assert load_minio({**ENV, "TRAINER_MINIO_ACCESS_KEY": "t-ak", "TRAINER_MINIO_SECRET_KEY": "t-sk"}).access_key == "t-ak"


# [상황] 데이터셋을 쓸 저장소를 원본 사진 저장소(trace-raw)로 잘못 설정한다.
# [기대] 설정 오류로 시작하지 않는다(원본 위에 덮어쓰는 사고 방지).
def test_출력_버킷이_원본_버킷과_같으면_시작하지_않는다():
    with pytest.raises(ConfigError):
        load_minio({**ENV, "TRACE_DATASETS_BUCKET": "trace-raw"})


# [상황] MinIO 주소나 DB 비밀번호 같은 필수 설정이 빠져 있다.
# [기대] 어떤 이름이 빠졌는지만 알려 준다. 다 있으면 기본값(localhost:5432/trace)이 채워지고
#        출력에 비밀번호가 보이지 않는다.
def test_빠진_환경_변수는_이름만_알린다():
    with pytest.raises(ConfigError, match="MINIO_ENDPOINT"):
        load_minio({})
    with pytest.raises(ConfigError, match="DB_PASSWORD"):
        load_db({"DB_USERNAME": "u"})
    db = load_db({"DB_USERNAME": "u", "DB_PASSWORD": "secret-pw"})
    assert (db.host, db.port, db.dbname) == ("localhost", 5432, "trace") and "secret-pw" not in repr(db)


# [상황] 데이터셋 저장소의 같은 위치에 파일을 두 번 올린다.
# [기대] 두 번째는 "이미 있음"으로 거절되고 처음 내용이 그대로다(만든 데이터셋은 바뀌지 않음).
def test_같은_키는_조건부_쓰기로_덮어쓸_수_없다():
    w = World()
    w.store.put_new("trace-datasets", "v001/a", b"1")
    with pytest.raises(AlreadyExists):
        w.store.put_new("trace-datasets", "v001/a", b"2")
    assert w.s3.objects[("trace-datasets", "v001/a")][0] == b"1"


# [상황] 파일 5개를 올리고 목록·존재 여부를 묻고, 없는 파일을 읽는다.
# [기대] 여러 쪽으로 나뉘어도 5개 모두 순서대로 받는다. 없는 파일은 "없음"·저장소 오류로 알린다.
def test_목록은_여러_페이지를_이어_받고_없는_키는_없음으로_알린다():
    w = World()
    for i in range(5):
        w.store.put_new("b", f"p/{i}", b"x")
    assert list(w.store.keys("b", "p/")) == [f"p/{i}" for i in range(5)]
    assert w.store.exists("b", "p/1") and not w.store.exists("b", "p/9")
    with pytest.raises(StoreError):
        w.store.get("b", "p/9")
