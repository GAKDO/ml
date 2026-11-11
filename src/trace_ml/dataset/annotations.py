"""Label Studio 내보내기(trace-labels) → 사진별 YOLO 폴리곤.

타깃 스토리지는 버전·설정에 따라 두 형식으로 쓴다.
- 작업 형식: {"id": 작업, "data": {"image": ...}, "annotations": [...]}
- 주석 형식: {"id": 주석, "result": [...], "task": {"id": 작업, "data": {...}}}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..codes import DEFECT_CLASSES
from ..labelstudio.config import KEY_REGEX, NO_DEFECT

_KEY = re.compile(KEY_REGEX)


@dataclass(frozen=True)
class Polygon:
    cls: int
    points: tuple[tuple[float, float], ...]  # 0~1


@dataclass
class LabeledImage:
    task_id: int
    annotation_id: int
    bucket: str
    key: str
    polygons: list[Polygon] = field(default_factory=list)

    @property
    def device(self) -> str:
        return self.key.split("/", 1)[0]

    @property
    def stem(self) -> str:
        return self.key.rsplit("/", 1)[-1].rsplit(".", 1)[0]

    def yolo_text(self) -> str:
        """한 줄에 폴리곤 하나: `클래스 x1 y1 x2 y2 ...`. 흠집 없는 사진은 빈 파일(배경)."""
        lines = [f"{p.cls} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in p.points) for p in self.polygons]
        return "".join(line + "\n" for line in lines)


class AnnotationErrors(Exception):
    """검증 오류 목록. 하나라도 있으면 빌드는 아무것도 쓰지 않는다."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__(f"주석 오류 {len(problems)}건")
        self.problems = problems


def _tasks(doc: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """(작업, 주석) 쌍으로 펼친다."""
    if "annotations" in doc:
        task = {"id": doc.get("id"), "data": doc.get("data") or {}}
        return [(task, a) for a in doc.get("annotations") or []]
    if "result" in doc:
        task = doc.get("task") or {}
        if not isinstance(task, dict):  # 아주 옛 형식은 작업 id만 있다
            task = {"id": task, "data": doc.get("data") or {}}
        return [({"id": task.get("id"), "data": task.get("data") or {}}, doc)]
    return []


def _recency(a: dict[str, Any]) -> tuple[str, int]:
    return str(a.get("updated_at") or ""), int(a.get("id") or 0)


def _pick(annotations: list[dict[str, Any]]) -> dict[str, Any] | None:
    """건너뛴 주석은 버리고, ground truth가 있으면 그것, 없으면 가장 최근 것.

    가장 최근 주석이 Skip이면 사진을 뺀다. trace-labels는 Label Studio가 객체를 지우지 못하게 해 두어서,
    주석을 지우고 Skip으로 바꾸면 지운 주석의 옛 내보내기 파일이 남는다. 그 옛 파일을 쓰면 안 된다.
    """
    if not annotations:
        return None
    alive = [a for a in annotations if not a.get("was_cancelled")]
    truth = [a for a in alive if a.get("ground_truth")]
    if truth:
        return max(truth, key=_recency)
    latest = max(annotations, key=_recency)
    return None if latest.get("was_cancelled") else latest


def _image_ref(data: dict[str, Any]) -> tuple[str, str] | None:
    url = str(data.get("image") or "")
    m = re.match(r"^s3://([^/]+)/(.+)$", url)
    return (m.group(1), m.group(2)) if m else None


def _parse(task: dict[str, Any], ann: dict[str, Any], problems: list[str]) -> LabeledImage | None:
    tid = task.get("id")
    ref = _image_ref(task["data"])
    if ref is None or not _KEY.match(ref[1]):
        problems.append(f"작업 {tid}: 사진 주소가 파일명 규칙에 맞지 않음({task['data'].get('image')!r})")
        return None
    image = LabeledImage(int(tid), int(ann.get("id") or 0), ref[0], ref[1])
    no_defect = False
    for r in ann.get("result") or []:
        value = r.get("value") or {}
        if r.get("type") == "choices" and NO_DEFECT in (value.get("choices") or []):
            no_defect = True
        elif r.get("type") == "polygonlabels":
            labels = value.get("polygonlabels") or []
            if len(labels) != 1 or labels[0] not in DEFECT_CLASSES:
                problems.append(f"작업 {tid}: 클래스가 codes.yaml 밖({labels})")
                continue
            pts = value.get("points") or []
            if len(pts) < 3:
                problems.append(f"작업 {tid}: 점이 3개 미만인 폴리곤")
                continue
            clamp = tuple((min(max(float(x) / 100, 0.0), 1.0), min(max(float(y) / 100, 0.0), 1.0)) for x, y in pts)
            image.polygons.append(Polygon(DEFECT_CLASSES.index(labels[0]), clamp))
    if no_defect and image.polygons:
        problems.append(f"작업 {tid}: 폴리곤과 '흠집 없음'이 함께 있음")
        return None
    if not no_defect and not image.polygons:
        problems.append(f"작업 {tid}: 폴리곤도 '흠집 없음'도 없는 제출")
        return None
    return image


def collect(docs: list[dict[str, Any]]) -> list[LabeledImage]:
    """내보낸 JSON 문서들 → 사진마다 하나. 오류가 있으면 모두 모아 AnnotationErrors."""
    tasks: dict[Any, dict[str, Any]] = {}
    anns: dict[Any, dict[Any, dict[str, Any]]] = {}
    for doc in docs:
        for task, ann in _tasks(doc):
            if task["data"] or task["id"] not in tasks:
                tasks[task["id"]] = task
            # 두 형식 파일이 함께 있으면 같은 주석이 두 번 나온다: 주석 id로 하나만 남긴다
            anns.setdefault(task["id"], {})[ann.get("id")] = ann
    problems: list[str] = []
    images: list[LabeledImage] = []
    for tid, task in tasks.items():
        chosen = _pick(list(anns[tid].values()))
        if chosen is not None:
            parsed = _parse(task, chosen, problems)
            if parsed is not None:
                images.append(parsed)
    seen: dict[str, int] = {}
    for img in images:
        if img.stem in seen:
            problems.append(f"작업 {seen[img.stem]}·{img.task_id}: 같은 사진({img.key})이 두 작업에 있음")
        seen[img.stem] = img.task_id
    if problems:
        raise AnnotationErrors(sorted(problems))
    return sorted(images, key=lambda i: i.key)
