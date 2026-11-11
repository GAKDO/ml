"""기록부: PostgreSQL `ml` 스키마(dataset_device·dataset_version·training_run).

흐름 코드는 Registry 프로토콜만 보고, 테스트는 메모리 구현을 끼운다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from importlib import resources
from typing import Any, Protocol

from .settings import DbSettings

SPLITS: tuple[str, ...] = ("TRAIN", "VAL", "TEST")


class RegistryError(Exception):
    """기록부 호출 실패. 메시지에 비밀번호는 넣지 않는다."""


class VersionExists(RegistryError):
    """같은 데이터셋 버전이 이미 있다(덮어쓰기 금지)."""


@dataclass(frozen=True)
class Device:
    code: str
    model_name: str
    split: str
    color: str | None = None


@dataclass(frozen=True)
class DatasetVersion:
    version: str
    manifest_key: str
    builder_commit: str
    counts: dict[str, Any]
    built_at: datetime


@dataclass(frozen=True)
class TrainingRun:
    dataset_version: str
    base_weights: str
    metrics: dict[str, Any]
    model_key: str
    onnx_key: str
    created_at: datetime | None = None


class Registry(Protocol):
    def devices(self) -> dict[str, Device]: ...
    def save_device(self, device: Device) -> None: ...
    def used_devices(self) -> set[str]: ...
    def versions(self) -> list[DatasetVersion]: ...
    def add_version(self, version: DatasetVersion, new_devices: list[Device]) -> None: ...
    def add_run(self, run: TrainingRun) -> None: ...
    def runs(self) -> list[TrainingRun]: ...


def used_in(versions: list[DatasetVersion]) -> set[str]:
    """버전 counts.devices({분할: [기기]})에 한 번이라도 나온 기기."""
    return {code for v in versions for codes in v.counts.get("devices", {}).values() for code in codes}


def schema_sql() -> str:
    return resources.files("trace_ml").joinpath("sql/V1__ml.sql").read_text(encoding="utf-8")


class PgRegistry:
    """psycopg 3. 호출마다 연결을 열고 닫는다(집 PC에서 가끔 도는 명령이라 풀이 필요 없다)."""

    def __init__(self, settings: DbSettings) -> None:
        self._s = settings

    def _connect(self) -> Any:
        import psycopg

        try:
            return psycopg.connect(host=self._s.host, port=self._s.port, dbname=self._s.dbname,
                                   user=self._s.user, password=self._s.password, connect_timeout=10)
        except psycopg.Error as exc:
            raise RegistryError(
                f"PostgreSQL에 닿지 않음({self._s.host}:{self._s.port}/{self._s.dbname}): {type(exc).__name__}."
                " 집 PC라면 kubectl -n trace port-forward svc/postgres 5432:5432"
            ) from exc

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(schema_sql())

    def devices(self) -> dict[str, Device]:
        with self._connect() as conn:
            rows = conn.execute("SELECT device_code, model_name, split, color FROM ml.dataset_device").fetchall()
        return {r[0]: Device(r[0], r[1], r[2], r[3]) for r in rows}

    def save_device(self, device: Device) -> None:
        with self._connect() as conn:
            self._upsert(conn, device)

    @staticmethod
    def _upsert(conn: Any, d: Device) -> None:
        conn.execute(
            "INSERT INTO ml.dataset_device (device_code, model_name, color, split) VALUES (%s, %s, %s, %s)"
            " ON CONFLICT (device_code) DO UPDATE SET model_name = EXCLUDED.model_name,"
            " color = EXCLUDED.color, split = EXCLUDED.split, updated_at = now()",
            (d.code, d.model_name, d.color, d.split),
        )

    def used_devices(self) -> set[str]:
        return used_in(self.versions())

    def versions(self) -> list[DatasetVersion]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT version, manifest_key, builder_commit, counts, built_at FROM ml.dataset_version ORDER BY version"
            ).fetchall()
        return [DatasetVersion(r[0], r[1], r[2], r[3], r[4]) for r in rows]

    def add_version(self, version: DatasetVersion, new_devices: list[Device]) -> None:
        """새로 배정한 기기와 버전 행을 한 트랜잭션으로 쓴다. 버전이 있으면 VersionExists."""
        import psycopg

        try:
            with self._connect() as conn, conn.transaction():
                for d in new_devices:
                    self._upsert(conn, d)
                conn.execute(
                    "INSERT INTO ml.dataset_version (version, manifest_key, builder_commit, counts, built_at)"
                    " VALUES (%s, %s, %s, %s::jsonb, %s)",
                    (version.version, version.manifest_key, version.builder_commit,
                     json.dumps(version.counts, ensure_ascii=False), version.built_at),
                )
        except psycopg.errors.UniqueViolation as exc:
            raise VersionExists(f"데이터셋 버전 {version.version}이 이미 있음") from exc

    def add_run(self, run: TrainingRun) -> None:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO ml.training_run (dataset_version_id, base_weights, metrics, model_key, onnx_key)"
                " SELECT id, %s, %s::jsonb, %s, %s FROM ml.dataset_version WHERE version = %s",
                (run.base_weights, json.dumps(run.metrics), run.model_key, run.onnx_key, run.dataset_version),
            )
            if cur.rowcount != 1:
                raise RegistryError(f"데이터셋 버전 {run.dataset_version}이 없어 학습 기록을 남기지 못함")

    def runs(self) -> list[TrainingRun]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT v.version, r.base_weights, r.metrics, r.model_key, r.onnx_key, r.created_at"
                " FROM ml.training_run r JOIN ml.dataset_version v ON v.id = r.dataset_version_id"
                " ORDER BY r.created_at, r.id"
            ).fetchall()
        return [TrainingRun(*r) for r in rows]
