"""이 파일이 확인하는 것: 흠집 검사 워커(inspect-worker, ML2)가 메시지 큐(RabbitMQ)와 주고받는 방식.
흐름: 부스·폰이 사진을 올렸다는 메시지를 받음 → 정합·흠집 찾기 → 결과 메시지 발행 → 받은 메시지 처리 완료(ack).
형식이 틀리거나 계속 실패하는 메시지는 DLQ(처리 못 한 메시지 보관함)로 보내고, 진행률은 Redis로 알린다.

ML2 브로커: 처리 → 발행 → ack, 형식 오류·계속 실패는 DLQ, 쌓인 작업 처리(스토리 2).

용어 풀이
  - ML2: 백로그(GAKDO/docs speckit/backlog.md)의 기능 번호. 흠집 검사 워커(inspect-worker)를 말한다.
  - 브로커: 서비스끼리 메시지를 주고받게 맡아 두는 메시지 큐 서버. 여기서는 RabbitMQ.
  - ack: "이 메시지는 처리 끝"이라고 브로커에 알리는 것. 알리기 전에 워커가 죽으면 브로커가 다시 보낸다.
  - DLQ(Dead Letter Queue): 처리할 수 없는 메시지를 버리지 않고 따로 모아 두는 큐. 사람이 나중에 살펴본다.
  - 스토리 2: 명세 specs/003-inspect-worker/spec.md의 사용자 스토리 2 "꺼져 있던 동안 쌓인 작업을 처리한다".
"""

from __future__ import annotations

import json
import os
import time
import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from fakes import MemoryS3
from inspect_helpers import DIMS, pid, synthetic_jpeg, uploaded_event

from trace_ml.inspect.runner import (
    DEAD_KEY,
    OUT_KEY,
    QUEUE,
    Consumer,
    RedisProgress,
    load_amqp,
    load_redis,
    load_worker_config,
)
from trace_ml.inspect.worker import parse_request
from trace_ml.labelstudio.config import ConfigError
from trace_ml.store import Store


# 테스트 도구: 가짜 메시지 큐 채널. 완료(ack)·거절(reject)·발행(publish) 호출을 순서대로 기록한다.
class FakeChannel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def basic_ack(self, delivery_tag: int) -> None:
        self.calls.append(("ack", delivery_tag))

    def basic_reject(self, delivery_tag: int, requeue: bool) -> None:
        self.calls.append(("reject", (delivery_tag, requeue)))

    def basic_publish(self, exchange: str, routing_key: str, body: bytes, properties: Any, mandatory: bool) -> None:
        self.calls.append(("publish", (exchange, routing_key, json.loads(body), properties)))


def _event_body() -> bytes:
    body, _ = synthetic_jpeg("TOP", seed=1)
    return json.dumps(uploaded_event([(pid("p1"), "TOP", body)])).encode()


OK_RESULT = {"eventId": pid("out"), "alignments": [], "failedPhotos": [], "defects": []}


# [상황] 올바른 촬영 업로드 메시지를 처리한다.
# [기대] 결과 메시지를 먼저 발행하고 그다음 완료(ack)한다(결과 없이 메시지가 사라지는 일 방지).
#        결과는 지워지지 않는 메시지로, 이벤트 ID를 메시지 ID로 보낸다.
def test_처리가_끝나면_결과를_보낸_뒤에_ack한다() -> None:
    ch = FakeChannel()
    result = Consumer(ch, lambda req: OK_RESULT).on_message(SimpleNamespace(delivery_tag=7), _event_body())
    assert result == "ok"
    assert [c[0] for c in ch.calls] == ["publish", "ack"]
    exchange, key, body, props = ch.calls[0][1]
    assert (exchange, key, body["eventId"]) == ("trace.events", OUT_KEY, pid("out"))
    assert props.message_id == pid("out") and props.delivery_mode == 2 and props.content_type == "application/json"


# [상황] JSON이 아닌 메시지와, 필수 내용이 빠진 메시지가 온다.
# [기대] 처리하지 않고 다시 넣지도 않고 DLQ로 보낸다.
def test_형식이_틀린_메시지는_다시_하지_않고_DLQ로_보낸다() -> None:
    ch = FakeChannel()
    calls = []
    consumer = Consumer(ch, lambda req: calls.append(req) or OK_RESULT)
    assert consumer.on_message(SimpleNamespace(delivery_tag=1), b"not json") == "invalid"
    assert consumer.on_message(SimpleNamespace(delivery_tag=2), b'{"schemaVersion":"1.0"}') == "invalid"
    assert ch.calls == [("reject", (1, False)), ("reject", (2, False))] and calls == []


# [상황] 처리 중 저장소 연결이 계속 끊긴다.
# [기대] 1초·2초 간격으로 세 번 시도한 뒤 DLQ로 보낸다.
def test_처리_오류는_세_번_해_보고_DLQ로_보낸다() -> None:
    ch = FakeChannel()
    tries = []
    sleeps: list[float] = []

    def boom(req: Any) -> dict:
        tries.append(1)
        raise RuntimeError("MinIO 끊김")

    consumer = Consumer(ch, boom, attempts=3, backoff=1.0, sleep=sleeps.append)
    assert consumer.on_message(SimpleNamespace(delivery_tag=3), _event_body()) == "failed"
    assert len(tries) == 3 and sleeps == [1.0, 2.0] and ch.calls == [("reject", (3, False))]


# [상황] 첫 시도는 잠깐 끊겨 실패하고 두 번째는 성공한다.
# [기대] 결과를 발행하고 완료(ack)한다.
def test_두_번째_시도에_성공하면_ack한다() -> None:
    ch = FakeChannel()
    state = {"n": 0}

    def flaky(req: Any) -> dict:
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("잠깐 끊김")
        return OK_RESULT

    assert Consumer(ch, flaky, sleep=lambda s: None).on_message(SimpleNamespace(delivery_tag=4), _event_body()) == "ok"
    assert [c[0] for c in ch.calls] == ["publish", "ack"]


# [상황] 정합 단계 진행률(3장 중 1장)을 알린다. 다음엔 Redis가 꺼진 상태로 알린다.
# [기대] 그 렌탈의 진행률 채널에 세트·시점·단계·진행 수가 JSON으로 나간다(웹 화면 진행 막대용).
#        Redis가 꺼져 있어도 검사는 멈추지 않는다.
def test_진행률은_렌탈_채널에_JSON으로_내고_실패해도_멈추지_않는다() -> None:
    sent: list[tuple[str, str]] = []
    redis = SimpleNamespace(publish=lambda ch, msg: sent.append((ch, msg)))
    req = parse_request(json.loads(_event_body()))
    RedisProgress(redis, "trace:").report(req, "ALIGN", 1, 3)
    channel, msg = sent[0]
    assert channel == f"trace:progress:{pid('rental1')}"
    assert json.loads(msg) == {"captureSetId": pid("set1"), "checkpointId": pid("cp1"), "stage": "ALIGN",
                               "done": 1, "total": 3}

    def down(ch: str, msg: str) -> None:
        raise ConnectionError("redis down")

    RedisProgress(SimpleNamespace(publish=down)).report(req, "ALIGN", 2, 3)  # 예외 없이 지나간다


# [상황] 메시지 큐·Redis·워커 설정을 환경 변수에서 읽는다(gitops와 같은 이름).
# [기대] 값이 채워지고 비밀번호는 출력에 안 보인다. 빠진 값은 이름만 알린다. Redis 설정이 없으면 진행률 없이 돈다.
def test_설정은_gitops_이름으로_읽고_빠지면_이름만_알린다() -> None:
    env = {"RABBITMQ_HOST": "mq", "RABBITMQ_USERNAME": "u", "RABBITMQ_PASSWORD": "secret", "RABBITMQ_VHOST": "/trace"}
    amqp = load_amqp(env)
    assert (amqp.host, amqp.port, amqp.vhost) == ("mq", 5672, "/trace") and "secret" not in repr(amqp)
    with pytest.raises(ConfigError, match="RABBITMQ_PASSWORD"):
        load_amqp({"RABBITMQ_HOST": "mq", "RABBITMQ_USERNAME": "u"})
    assert load_redis({}) is None
    redis = load_redis({"REDIS_HOST": "r", "REDIS_DATABASE": "2", "REDIS_PASSWORD": "pw"})
    assert redis is not None and redis.db == 2 and "pw" not in repr(redis)
    cfg = load_worker_config({"TRACE_DERIVED_BUCKET": "d"})
    assert (cfg.captures_bucket, cfg.derived_bucket, cfg.dims) == ("trace-captures", "d", DIMS)


# ---- 실제 RabbitMQ (CI 서비스, TRACE_ML_TEST_AMQP=1) ----

amqp_only = pytest.mark.skipif(not os.environ.get("TRACE_ML_TEST_AMQP"), reason="실제 RabbitMQ 없음")


# [상황] 워커가 꺼진 동안 촬영 세트 3개와 깨진 메시지 1개가 쌓였다(진짜 RabbitMQ가 있을 때만 돈다).
# [기대] 워커를 켜면 4건을 모두 처리해 큐가 비고, 결과 3건이 세트마다 나가며, 깨진 메시지는 DLQ에 있다.
@amqp_only
def test_워커를_끈_채_올린_3건이_워커를_켜면_모두_처리된다() -> None:
    import pika

    from trace_ml.inspect.runner import connect, declare, serve

    env = {k: os.environ[k] for k in ("RABBITMQ_HOST", "RABBITMQ_USERNAME", "RABBITMQ_PASSWORD") if k in os.environ}
    env.setdefault("RABBITMQ_HOST", "localhost")
    settings = load_amqp({"RABBITMQ_USERNAME": "guest", "RABBITMQ_PASSWORD": "guest", **env})
    conn = connect(settings)
    ch = conn.channel()
    declare(ch)
    ch.queue_purge(QUEUE)
    out_q = f"test.completed.{uuid.uuid4().hex[:8]}"
    ch.queue_declare(queue=out_q, exclusive=True)
    ch.queue_bind(queue=out_q, exchange="trace.events", routing_key=OUT_KEY)

    s3 = MemoryS3()
    for n in range(3):  # 워커가 꺼져 있는 동안 세트 3개(사진 3장)가 올라온다
        body, _ = synthetic_jpeg("TOP", seed=10 + n)
        event = uploaded_event([(pid(f"p{n}"), "TOP", body)], event_id=pid(f"evt{n}"), set_id=pid(f"set{n}"))
        s3.objects[("trace-captures", event["photos"][0]["objectKey"])] = (body, {})
        ch.basic_publish("trace.events", "trace.capture.uploaded", json.dumps(event).encode(),
                         pika.BasicProperties(delivery_mode=2, content_type="application/json"))
    ch.basic_publish("trace.events", "trace.capture.uploaded", b"broken",
                     pika.BasicProperties(delivery_mode=2))
    time.sleep(0.5)
    assert ch.queue_declare(queue=QUEUE, passive=True).method.message_count == 4

    handled = serve({**env, "RABBITMQ_USERNAME": settings.username, "RABBITMQ_PASSWORD": settings.password},
                    Store(s3), idle_timeout=2, progress=None)
    assert handled == 4
    assert ch.queue_declare(queue=QUEUE, passive=True).method.message_count == 0
    results = []
    for _ in range(3):
        method, props, body = ch.basic_get(out_q, auto_ack=True)
        assert method is not None
        results.append(json.loads(body))
        assert props.message_id == results[-1]["eventId"]
    assert sorted(r["captureSetId"] for r in results) == [pid(f"set{n}") for n in range(3)]
    assert all(len(r["alignments"]) == 1 for r in results)
    dead = ch.basic_get(DEAD_KEY, auto_ack=True)
    assert dead[0] is not None and dead[2] == b"broken"
    conn.close()


# [상황] 결과 메시지를 받을 큐가 없어 전달이 안 되는 상황.
# [기대] 조용히 사라지지 않고 실패로 처리해 DLQ로 보낸다(결과 유실 방지).
def test_결과_이벤트는_받을_큐가_없으면_실패하도록_mandatory로_보낸다() -> None:
    seen = {}

    class Ch(FakeChannel):
        def basic_publish(self, exchange: str, routing_key: str, body: bytes, properties: Any,
                          mandatory: bool) -> None:
            seen["mandatory"] = mandatory
            if mandatory:
                raise RuntimeError("unroutable")

    ch = Ch()
    result = Consumer(ch, lambda req: OK_RESULT, sleep=lambda s: None).on_message(SimpleNamespace(delivery_tag=9),
                                                                                    _event_body())
    assert seen["mandatory"] is True and result == "failed" and ch.calls == [("reject", (9, False))]
