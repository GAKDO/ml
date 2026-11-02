"""데이터셋 빌더·학습(ML1) 설정. 이름은 gitops infra.env·trace-secrets와 같다."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .labelstudio.config import ConfigError


@dataclass(frozen=True)
class MinioSettings:
    endpoint: str
    raw_bucket: str = "trace-raw"
    labels_bucket: str = "trace-labels"
    datasets_bucket: str = "trace-datasets"
    models_bucket: str = "trace-models"
    # 비밀 값은 repr에 나오지 않게 한다
    access_key: str = field(default="", repr=False)
    secret_key: str = field(default="", repr=False)


@dataclass(frozen=True)
class DbSettings:
    host: str = "localhost"
    port: int = 5432
    dbname: str = "trace"
    user: str = ""
    password: str = field(default="", repr=False)


def _get(env: Mapping[str, str], *names: str) -> str:
    return next((env[n] for n in names if env.get(n)), "")


def load_minio(env: Mapping[str, str]) -> MinioSettings:
    """trainer 역할 키가 없으면 팀 키를 쓴다. 빠진 이름만 알리고 값은 보여 주지 않는다."""
    values = {
        "MINIO_ENDPOINT": _get(env, "MINIO_ENDPOINT"),
        "TRAINER_MINIO_ACCESS_KEY": _get(env, "TRAINER_MINIO_ACCESS_KEY", "MINIO_ACCESS_KEY"),
        "TRAINER_MINIO_SECRET_KEY": _get(env, "TRAINER_MINIO_SECRET_KEY", "MINIO_SECRET_KEY"),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ConfigError("환경 변수가 없음: " + ", ".join(missing))
    s = MinioSettings(
        endpoint=values["MINIO_ENDPOINT"],
        raw_bucket=_get(env, "TRACE_RAW_BUCKET") or "trace-raw",
        labels_bucket=_get(env, "TRACE_LABELS_BUCKET") or "trace-labels",
        datasets_bucket=_get(env, "TRACE_DATASETS_BUCKET") or "trace-datasets",
        models_bucket=_get(env, "TRACE_MODELS_BUCKET") or "trace-models",
        access_key=values["TRAINER_MINIO_ACCESS_KEY"],
        secret_key=values["TRAINER_MINIO_SECRET_KEY"],
    )
    # 원본·주석 버킷에 쓰는 일이 없도록 출력 버킷이 입력 버킷과 겹치면 시작하지 않는다
    if {s.datasets_bucket, s.models_bucket} & {s.raw_bucket, s.labels_bucket}:
        raise ConfigError("데이터셋·모델 버킷이 원본·주석 버킷과 같음: 원본을 덮어쓸 수 있어 시작하지 않음")
    return s


def load_db(env: Mapping[str, str]) -> DbSettings:
    missing = [n for n in ("DB_USERNAME", "DB_PASSWORD") if not env.get(n)]
    if missing:
        raise ConfigError("환경 변수가 없음: " + ", ".join(missing))
    port = _get(env, "POSTGRES_PORT") or "5432"
    if not port.isdigit():
        raise ConfigError("POSTGRES_PORT는 숫자여야 함")
    return DbSettings(
        host=_get(env, "POSTGRES_HOST") or "localhost",
        port=int(port),
        dbname=_get(env, "POSTGRES_DB") or "trace",
        user=env["DB_USERNAME"],
        password=env["DB_PASSWORD"],
    )
