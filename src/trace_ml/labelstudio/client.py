"""Label Studio REST API(1.19) 래퍼. HTTP 세션은 주입받아 테스트에서 바꿔 끼운다."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol

import requests


class LabelStudioError(Exception):
    """Label Studio 호출 실패. 메시지에 토큰 값은 넣지 않는다."""


class HttpSession(Protocol):
    def request(self, method: str, url: str, **kwargs: Any) -> Any: ...


class LabelStudio:
    def __init__(self, url: str, api_key: str, session: HttpSession | None = None, timeout: float = 30) -> None:
        self._base = url.rstrip("/")
        self._headers = {"Authorization": f"Token {api_key}"}
        self._session: HttpSession = session or requests.Session()
        self._timeout = timeout

    def _call(self, method: str, path: str, *, json: dict[str, Any] | None = None,
              params: dict[str, Any] | None = None) -> Any:
        try:
            resp = self._session.request(method, self._base + path, headers=self._headers, json=json,
                                         params=params, timeout=self._timeout)
        except requests.RequestException as exc:
            raise LabelStudioError(f"Label Studio에 닿지 않음({self._base}): {type(exc).__name__}") from exc
        if resp.status_code in (401, 403):
            raise LabelStudioError(f"토큰이 틀렸거나 권한이 없음(HTTP {resp.status_code}): LABEL_STUDIO_API_KEY 확인")
        if resp.status_code >= 400:
            raise LabelStudioError(f"{method} {path} 실패(HTTP {resp.status_code}): {resp.text[:200]}")
        return resp.json() if resp.content else None

    @staticmethod
    def _items(data: Any) -> list[dict[str, Any]]:
        # 목록 API는 버전에 따라 배열이거나 {"results": [...]}다
        return list(data.get("results", [])) if isinstance(data, dict) else list(data or [])

    # 프로젝트
    def projects_titled(self, title: str) -> list[dict[str, Any]]:
        data = self._call("GET", "/api/projects", params={"title": title, "page_size": 1000})
        return [p for p in self._items(data) if p.get("title") == title]

    def project(self, project_id: int) -> dict[str, Any]:
        return self._call("GET", f"/api/projects/{project_id}")

    def create_project(self, title: str, label_config: str) -> dict[str, Any]:
        return self._call("POST", "/api/projects", json={"title": title, "label_config": label_config})

    def task_count(self, project_id: int) -> int:
        return int(self.project(project_id).get("task_number") or 0)

    # 스토리지: kind는 "s3"(소스) 또는 "export/s3"(타깃)
    def storages(self, kind: str, project_id: int) -> list[dict[str, Any]]:
        return self._items(self._call("GET", f"/api/storages/{kind}", params={"project": project_id}))

    def create_storage(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._call("POST", f"/api/storages/{kind}", json=payload)

    def sync_storage(self, kind: str, storage_id: int, *, wait: float = 300,
                     sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
        """동기화를 걸고 끝날 때까지 기다린다(Redis가 있으면 비동기로 돈다)."""
        storage = self._call("POST", f"/api/storages/{kind}/{storage_id}/sync")
        waited = 0.0
        while (storage or {}).get("status") in ("initialized", "queued", "in_progress"):
            if waited >= wait:
                raise LabelStudioError(f"{kind} 스토리지 {storage_id} 동기화가 {wait:.0f}초 안에 끝나지 않음")
            sleep(2)
            waited += 2
            storage = self._call("GET", f"/api/storages/{kind}/{storage_id}")
        if (storage or {}).get("status") == "failed":
            reason = str(storage.get("traceback") or "").strip().splitlines()[-1:] or ["사유 없음"]
            raise LabelStudioError(f"{kind} 스토리지 {storage_id} 동기화 실패: {reason[0]}")
        return storage or {}

    # 작업·주석
    def tasks(self, project_id: int, page_size: int = 500) -> list[dict[str, Any]]:
        """프로젝트의 작업을 모두 읽는다(페이지를 끝까지 넘김)."""
        found: list[dict[str, Any]] = []
        page = 1
        while True:
            # resolve_uri=False: 이미지 값을 프록시 URL이 아닌 원본 s3:// 경로로 받는다(기기 ID를 읽으려고)
            data = self._call("GET", "/api/tasks", params={"project": project_id, "page": page,
                                                           "page_size": page_size, "fields": "all",
                                                           "resolve_uri": False})
            if not isinstance(data, dict):
                return found + list(data or [])
            items = list(data.get("tasks") or data.get("results") or [])
            found.extend(items)
            total = data.get("total")
            if total is None:
                if len(items) < page_size:
                    return found
            elif len(found) >= int(total):
                return found
            elif not items:
                # 서버가 page_size를 더 작게 잘라도 total까지 계속 읽는다. 빈 페이지면 더 진행할 수 없다
                raise LabelStudioError(f"작업 {total}개 중 {len(found)}개만 읽힘: 페이지 {page}가 비어 있음")
            page += 1

    def create_annotation(self, task_id: int, result: list[dict[str, Any]]) -> dict[str, Any]:
        return self._call("POST", f"/api/tasks/{task_id}/annotations", json={"result": result})
