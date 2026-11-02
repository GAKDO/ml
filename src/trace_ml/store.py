"""MinIO(S3) 래퍼. 클라이언트는 주입받아 테스트에서 메모리 구현으로 바꿔 끼운다."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

from .settings import MinioSettings


class StoreError(Exception):
    """저장소 호출 실패. 메시지에 키 값은 넣지 않는다."""


class AlreadyExists(StoreError):
    """조건부 쓰기(If-None-Match: *)가 거절됨 = 같은 키가 이미 있음."""


class S3Client(Protocol):
    def get_object(self, **kwargs: Any) -> dict[str, Any]: ...
    def put_object(self, **kwargs: Any) -> dict[str, Any]: ...
    def head_object(self, **kwargs: Any) -> dict[str, Any]: ...
    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]: ...


def _code(exc: Exception) -> str:
    return str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))


class Store:
    def __init__(self, client: S3Client) -> None:
        self._s3 = client

    def get(self, bucket: str, key: str) -> bytes:
        try:
            return self._s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        except Exception as exc:  # botocore 예외 종류가 많아 코드로 가른다
            if _code(exc) in ("NoSuchKey", "404"):
                raise StoreError(f"없음: {bucket}/{key}") from exc
            raise StoreError(f"{bucket}/{key} 읽기 실패: {_code(exc) or type(exc).__name__}") from exc

    def exists(self, bucket: str, key: str) -> bool:
        try:
            self._s3.head_object(Bucket=bucket, Key=key)
            return True
        except Exception as exc:
            if _code(exc) in ("404", "NoSuchKey", "NotFound"):
                return False
            raise StoreError(f"{bucket}/{key} 확인 실패: {_code(exc) or type(exc).__name__}") from exc

    def keys(self, bucket: str, prefix: str = "") -> Iterator[str]:
        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            try:
                page = self._s3.list_objects_v2(**kwargs)
            except Exception as exc:
                raise StoreError(f"{bucket} 목록 실패: {_code(exc) or type(exc).__name__}") from exc
            for item in page.get("Contents", []):
                yield item["Key"]
            if not page.get("IsTruncated"):
                return
            token = page.get("NextContinuationToken")

    def put_new(self, bucket: str, key: str, body: bytes, content_type: str = "application/octet-stream",
                metadata: dict[str, str] | None = None) -> None:
        """없을 때만 쓴다. 있으면 AlreadyExists. 덮어쓰기 금지(FR-DAT-04)의 마지막 방어선이다."""
        try:
            self._s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type,
                                Metadata=metadata or {}, IfNoneMatch="*")
        except Exception as exc:
            if _code(exc) in ("PreconditionFailed", "412", "ConditionalRequestConflict"):
                raise AlreadyExists(f"이미 있음: {bucket}/{key}") from exc
            raise StoreError(f"{bucket}/{key} 쓰기 실패: {_code(exc) or type(exc).__name__}") from exc


def make_store(settings: MinioSettings) -> Store:
    import boto3  # 무거워서 쓸 때만 가져온다

    client = boto3.client(
        "s3",
        endpoint_url=settings.endpoint,
        aws_access_key_id=settings.access_key,
        aws_secret_access_key=settings.secret_key,
        region_name="us-east-1",
    )
    return Store(client)
