"""가짜 Label Studio: 메모리 상태로 필요한 API만 흉내 내고 호출을 기록한다."""

from __future__ import annotations

import json as jsonlib
import re
from dataclasses import dataclass, field
from typing import Any

import requests

GOOD_TOKEN = "Token test-key"
BASE = "http://ls.test"


@dataclass
class FakeResponse:
    status_code: int
    payload: Any = None

    @property
    def content(self) -> bytes:
        return b"" if self.payload is None else jsonlib.dumps(self.payload).encode()

    @property
    def text(self) -> str:
        return self.content.decode()

    def json(self) -> Any:
        return self.payload


@dataclass
class FakeLabelStudio:
    objects: dict[str, list[str]] = field(default_factory=dict)  # 버킷 → 키 목록
    projects: list[dict[str, Any]] = field(default_factory=list)
    storages: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: {"s3": [], "export/s3": []})
    calls: list[tuple[str, str]] = field(default_factory=list)
    imported: set[str] = field(default_factory=set)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    unreachable: bool = False
    _next_id: int = 1

    def _id(self) -> int:
        self._next_id += 1
        return self._next_id

    def add_project(self, title: str, label_config: str) -> dict[str, Any]:
        project = {"id": self._id(), "title": title, "label_config": label_config, "task_number": 0}
        self.projects.append(project)
        return project

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        if self.unreachable:
            raise requests.ConnectionError("연결 거부")
        path = url.removeprefix(BASE)
        self.calls.append((method, path))
        if kwargs.get("headers", {}).get("Authorization") != GOOD_TOKEN:
            return FakeResponse(401, {"detail": "Invalid token"})
        body = kwargs.get("json") or {}
        params = kwargs.get("params") or {}

        if path == "/api/projects":
            if method == "POST":
                return FakeResponse(201, self.add_project(body["title"], body["label_config"]))
            return FakeResponse(200, {"results": [p for p in self.projects if p["title"] == params.get("title")]})
        if m := re.fullmatch(r"/api/projects/(\d+)", path):
            return FakeResponse(200, next(p for p in self.projects if p["id"] == int(m[1])))
        if m := re.fullmatch(r"/api/storages/(s3|export/s3)", path):
            kind = m[1]
            if method == "POST":
                storage = {**body, "id": self._id(), "status": "initialized"}
                self.storages[kind].append(storage)
                return FakeResponse(201, storage)
            return FakeResponse(200, [s for s in self.storages[kind] if s["project"] == params.get("project")])
        if m := re.fullmatch(r"/api/storages/(s3|export/s3)/(\d+)(/sync)?", path):
            storage = next(s for s in self.storages[m[1]] if s["id"] == int(m[2]))
            if m[3] and m[1] == "s3":
                self._import(storage)
            if m[3]:
                storage["status"] = "completed"
            return FakeResponse(200, storage)
        if path == "/api/tasks":
            mine = [t for t in self.tasks if t["project"] == params.get("project")]
            size, page = int(params.get("page_size", 100)), int(params.get("page", 1))
            return FakeResponse(200, {"total": len(mine), "tasks": mine[(page - 1) * size:page * size]})
        if m := re.fullmatch(r"/api/tasks/(\d+)/annotations", path):
            task = next(t for t in self.tasks if t["id"] == int(m[1]))
            task["annotations"].append(body)
            task["total_annotations"] = len(task["annotations"])
            return FakeResponse(201, {"id": self._id(), **body})
        return FakeResponse(404, {"detail": f"없는 경로 {path}"})

    def add_task(self, project_id: int, key: str, annotated: bool = False) -> dict[str, Any]:
        task = {"id": self._id(), "project": project_id, "data": {"image": f"s3://trace-raw/{key}"},
                "annotations": [{"result": []}] if annotated else [], "total_annotations": int(annotated)}
        self.tasks.append(task)
        return task

    def _import(self, storage: dict[str, Any]) -> None:
        """키 필터에 맞는 새 객체마다 작업 하나를 만든다(use_blob_urls)."""
        project = next(p for p in self.projects if p["id"] == storage["project"])
        for key in self.objects.get(storage["bucket"], []):
            ref = f"{storage['bucket']}/{key}"
            if ref not in self.imported and re.search(storage.get("regex_filter") or "", key):
                self.imported.add(ref)
                project["task_number"] += 1
