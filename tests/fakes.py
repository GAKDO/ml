"""메모리 MinIO·메모리 기록부·Label Studio 내보내기 만들기."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Any

from trace_ml.registry import DatasetVersion, Device, RegistryError, TrainingRun, VersionExists, used_in
from trace_ml.settings import MinioSettings
from trace_ml.store import Store

SETTINGS = MinioSettings(endpoint="http://minio.test", access_key="ak", secret_key="sk")


class FakeClientError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class MemoryS3:
    """boto3 S3 클라이언트에서 쓰는 메서드만. 페이지 크기를 작게 해 이어 받기도 시험한다."""

    def __init__(self, page_size: int = 2) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, dict[str, str]]] = {}
        self.page_size = page_size
        self.puts: list[tuple[str, str]] = []

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, IfNoneMatch: str | None = None,
                   Metadata: dict[str, str] | None = None, **_: Any) -> dict[str, Any]:
        if IfNoneMatch == "*" and (Bucket, Key) in self.objects:
            raise FakeClientError("PreconditionFailed")
        self.objects[(Bucket, Key)] = (bytes(Body), dict(Metadata or {}))
        self.puts.append((Bucket, Key))
        return {}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        if (Bucket, Key) not in self.objects:
            raise FakeClientError("NoSuchKey")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)][0])}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        if (Bucket, Key) not in self.objects:
            raise FakeClientError("404")
        return {"Metadata": self.objects[(Bucket, Key)][1]}

    def list_objects_v2(self, *, Bucket: str, Prefix: str = "", ContinuationToken: str | None = None,
                        **_: Any) -> dict[str, Any]:
        keys = sorted(k for b, k in self.objects if b == Bucket and k.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = keys[start:start + self.page_size]
        more = start + self.page_size < len(keys)
        out: dict[str, Any] = {"Contents": [{"Key": k} for k in page], "IsTruncated": more}
        if more:
            out["NextContinuationToken"] = str(start + self.page_size)
        return out

    def keys(self, bucket: str) -> list[str]:
        return sorted(k for b, k in self.objects if b == bucket)


@dataclass
class MemoryRegistry:
    device_rows: dict[str, Device] = field(default_factory=dict)
    version_rows: list[DatasetVersion] = field(default_factory=list)
    run_rows: list[TrainingRun] = field(default_factory=list)

    def devices(self) -> dict[str, Device]:
        return dict(self.device_rows)

    def save_device(self, device: Device) -> None:
        self.device_rows[device.code] = device

    def used_devices(self) -> set[str]:
        return used_in(self.version_rows)

    def versions(self) -> list[DatasetVersion]:
        return sorted(self.version_rows, key=lambda v: v.version)

    def add_version(self, version: DatasetVersion, new_devices: list[Device]) -> None:
        if any(v.version == version.version for v in self.version_rows):
            raise VersionExists(version.version)
        for d in new_devices:
            self.device_rows[d.code] = d
        self.version_rows.append(version)

    def add_run(self, run: TrainingRun) -> None:
        if not any(v.version == run.dataset_version for v in self.version_rows):
            raise RegistryError(run.dataset_version)
        self.run_rows.append(run)

    def runs(self) -> list[TrainingRun]:
        return list(self.run_rows)


def polygon(cls: str, points: list[tuple[float, float]]) -> dict[str, Any]:
    return {"type": "polygonlabels", "from_name": "defect", "to_name": "image",
            "value": {"points": [list(p) for p in points], "polygonlabels": [cls], "closed": True}}


NO_DEFECT = {"type": "choices", "from_name": "no_defect", "to_name": "image", "value": {"choices": ["no_defect"]}}
TRI = [(10.0, 10.0), (20.0, 10.0), (15.0, 20.0)]


def task_doc(task_id: int, key: str, *annotations: dict[str, Any]) -> dict[str, Any]:
    """Label Studio 1.19 작업 형식 내보내기."""
    return {"id": task_id, "data": {"image": f"s3://trace-raw/{key}"}, "annotations": list(annotations)}


def ann(ann_id: int, *result: dict[str, Any], cancelled: bool = False, updated: str = "2026-11-03T00:00:00Z",
        ground_truth: bool = False) -> dict[str, Any]:
    return {"id": ann_id, "result": list(result), "was_cancelled": cancelled, "updated_at": updated,
            "ground_truth": ground_truth}


class World:
    """trace-raw 사진과 trace-labels 내보내기를 함께 만든다."""

    def __init__(self) -> None:
        self.s3 = MemoryS3()
        self.store = Store(self.s3)
        self.registry = MemoryRegistry()
        self._next = 1

    def photo(self, key: str, *annotations: dict[str, Any]) -> int:
        tid = self._next
        self._next += 1
        self.s3.put_object(Bucket="trace-raw", Key=key, Body=f"jpeg:{key}".encode())
        self.s3.put_object(Bucket="trace-labels", Key=str(tid),
                           Body=json.dumps(task_doc(tid, key, *annotations)).encode())
        return tid

    def labeled(self, device: str, n: int, face: str = "TOP") -> None:
        for i in range(n):
            self.photo(f"{device}/{device}_{face}_L1_CAM1_{i + 1:02d}.jpg", ann(1000 + self._next, polygon("scratch", TRI)))
