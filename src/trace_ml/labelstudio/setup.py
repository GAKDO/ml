"""setup(없을 때만 만듦)과 sync(소스 가져오기 → 주석 내보내기) 흐름."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .client import LabelStudio, LabelStudioError
from .config import KEY_REGEX, Settings, label_config, same_config

SOURCE = "s3"
TARGET = "export/s3"


@dataclass
class SetupResult:
    project_id: int
    created: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class SyncResult:
    project_id: int
    tasks_before: int
    tasks_after: int

    @property
    def added(self) -> int:
        return self.tasks_after - self.tasks_before


def source_payload(settings: Settings, project_id: int) -> dict[str, Any]:
    """trace-raw 읽기 전용 소스. presign=False가 프록시 모드(브라우저가 MinIO에 직접 닿지 않음)."""
    return {
        "project": project_id,
        "title": settings.raw_bucket,
        "bucket": settings.raw_bucket,
        "prefix": "",
        "regex_filter": KEY_REGEX,
        "use_blob_urls": True,
        "recursive_scan": True,
        "presign": False,
        "s3_endpoint": settings.minio_endpoint,
        "aws_access_key_id": settings.minio_access_key,
        "aws_secret_access_key": settings.minio_secret_key,
    }


def target_payload(settings: Settings, project_id: int) -> dict[str, Any]:
    """trace-labels 주석 타깃. Label Studio가 저장소 객체를 지우지 못하게 한다."""
    return {
        "project": project_id,
        "title": settings.labels_bucket,
        "bucket": settings.labels_bucket,
        "prefix": "",
        "can_delete_objects": False,
        "s3_endpoint": settings.minio_endpoint,
        "aws_access_key_id": settings.minio_access_key,
        "aws_secret_access_key": settings.minio_secret_key,
    }


def find_project(ls: LabelStudio, title: str) -> dict[str, Any] | None:
    found = ls.projects_titled(title)
    if len(found) > 1:
        ids = ", ".join(str(p["id"]) for p in found)
        raise LabelStudioError(f"'{title}' 프로젝트가 {len(found)}개(id {ids}): 하나만 남기고 다시 실행")
    return found[0] if found else None


def _by_bucket(ls: LabelStudio, kind: str, project_id: int, bucket: str) -> dict[str, Any] | None:
    return next((s for s in ls.storages(kind, project_id) if s.get("bucket") == bucket), None)


def setup(ls: LabelStudio, settings: Settings) -> SetupResult:
    project = find_project(ls, settings.project_title)
    if project is None:
        project = ls.create_project(settings.project_title, label_config())
        result = SetupResult(project["id"], created=["프로젝트"])
    else:
        result = SetupResult(project["id"])
        current = ls.project(project["id"]).get("label_config", "")
        if not same_config(current, label_config()):
            result.warnings.append("라벨 설정이 이 도구와 다름: 기존 주석을 지키려고 덮어쓰지 않음 (화면에서 확인)")

    source = _by_bucket(ls, SOURCE, result.project_id, settings.raw_bucket)
    if source is None:
        ls.create_storage(SOURCE, source_payload(settings, result.project_id))
        result.created.append(f"소스 {settings.raw_bucket}")
    else:
        if source.get("presign"):
            result.warnings.append(f"소스 {settings.raw_bucket}이 사전 서명 URL을 씀: 프록시 모드로 바꾸세요")
        if source.get("regex_filter") != KEY_REGEX:
            result.warnings.append(f"소스 {settings.raw_bucket}의 키 필터가 다름: SLATE 등이 섞일 수 있음")

    target = _by_bucket(ls, TARGET, result.project_id, settings.labels_bucket)
    if target is None:
        ls.create_storage(TARGET, target_payload(settings, result.project_id))
        result.created.append(f"타깃 {settings.labels_bucket}")
    elif target.get("can_delete_objects"):
        result.warnings.append(f"타깃 {settings.labels_bucket}에서 객체 삭제가 켜져 있음: 끄세요")
    return result


def sync(ls: LabelStudio, settings: Settings, **wait: Any) -> SyncResult:
    project = find_project(ls, settings.project_title)
    if project is None:
        raise LabelStudioError(f"'{settings.project_title}' 프로젝트가 없음: 먼저 setup 실행")
    pid = project["id"]
    source = _by_bucket(ls, SOURCE, pid, settings.raw_bucket)
    target = _by_bucket(ls, TARGET, pid, settings.labels_bucket)
    if source is None or target is None:
        raise LabelStudioError("소스 또는 타깃 스토리지가 없음: 먼저 setup 실행")
    before = ls.task_count(pid)
    ls.sync_storage(SOURCE, source["id"], **wait)
    after = ls.task_count(pid)
    ls.sync_storage(TARGET, target["id"], **wait)
    return SyncResult(pid, before, after)
