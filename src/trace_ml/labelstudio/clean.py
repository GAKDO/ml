"""흠집 없는 기기의 사진에 no_defect 주석을 한꺼번에 제출한다(손 클릭 대신)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .client import LabelStudio, LabelStudioError
from .config import NO_DEFECT, Settings
from .setup import find_project

DEVICE_RE = re.compile(r"^[A-Z]{2,4}[0-9]{2}$")

# 화면에서 no_defect를 체크했을 때와 같은 주석 한 줄
NO_DEFECT_RESULT: list[dict[str, Any]] = [
    {"from_name": NO_DEFECT, "to_name": "image", "type": "choices", "value": {"choices": [NO_DEFECT]}},
]


@dataclass
class CleanResult:
    project_id: int
    # 기기 → (새로 제출, 이미 주석 있어 건너뜀)
    counts: dict[str, tuple[int, int]] = field(default_factory=dict)


def _image_key(task: dict[str, Any]) -> str:
    return str((task.get("data") or {}).get("image") or "")


def _device_of(key: str) -> str | None:
    # s3://trace-raw/MB02/MB02_TOP_L1_CAM1_01.jpg → MB02 (폴더와 파일 앞부분이 같을 때만)
    m = re.search(r"([A-Z]{2,4}[0-9]{2})/\1_[^/]+$", key)
    return m[1] if m else None


def _has_annotation(task: dict[str, Any]) -> bool:
    if int(task.get("total_annotations") or 0) > 0 or int(task.get("cancelled_annotations") or 0) > 0:
        return True
    return bool(task.get("annotations"))


def mark_clean(ls: LabelStudio, settings: Settings, devices: list[str], *, dry_run: bool = False) -> CleanResult:
    """기기마다 아직 주석이 없는 작업에만 no_defect를 제출한다. 이미 그린 사진은 건드리지 않는다."""
    bad = [d for d in devices if not DEVICE_RE.fullmatch(d)]
    if bad:
        raise LabelStudioError(f"기기 ID 형식이 아님: {', '.join(bad)} (예: MB02)")
    project = find_project(ls, settings.project_title)
    if project is None:
        raise LabelStudioError(f"'{settings.project_title}' 프로젝트가 없음: 먼저 setup 실행")
    result = CleanResult(project["id"], {d: (0, 0) for d in devices})
    for task in ls.tasks(project["id"]):
        device = _device_of(_image_key(task))
        if device not in result.counts:
            continue
        made, skipped = result.counts[device]
        if _has_annotation(task):
            result.counts[device] = (made, skipped + 1)
            continue
        if not dry_run:
            ls.create_annotation(int(task["id"]), NO_DEFECT_RESULT)
        result.counts[device] = (made + 1, skipped)
    return result
