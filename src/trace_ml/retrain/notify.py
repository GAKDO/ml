"""운영자 알림: Discord·Slack 웹훅(TRACE_ALERT_WEBHOOK_URL). 주소는 비밀이라 메시지·로그에 남기지 않는다."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

Post = Callable[[str, dict[str, Any]], int]  # (주소, JSON) → HTTP 상태


def _post(url: str, payload: dict[str, Any]) -> int:  # pragma: no cover - 네트워크
    import requests

    return requests.post(url, json=payload, timeout=10).status_code


def payload(url: str, text: str) -> dict[str, Any]:
    """Slack은 text, Discord는 content 필드를 읽는다."""
    return {"text": text} if "hooks.slack.com" in url else {"content": text}


def notify(env: Mapping[str, str], text: str, post: Post = _post) -> str:
    """보낸 결과를 한 줄로 돌려준다. 실패해도 예외를 내지 않는다(알림 때문에 파이프라인이 멈추지 않게)."""
    url = env.get("TRACE_ALERT_WEBHOOK_URL")
    if not url:
        return "알림 건너뜀: TRACE_ALERT_WEBHOOK_URL 없음"
    try:
        status = post(url, payload(url, text))
    except Exception as exc:  # noqa: BLE001 - 어떤 네트워크 오류든 알림만 실패로 남긴다
        return f"알림 실패: {type(exc).__name__}"
    return "알림 보냄" if 200 <= status < 300 else f"알림 실패: HTTP {status}"
