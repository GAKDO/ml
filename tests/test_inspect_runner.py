"""ML2 브로커: 처리 → 발행 → ack, 형식 오류·계속 실패는 DLQ, 쌓인 작업 처리(스토리 2)."""

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


def test_처리가_끝나면_결과를_보낸_뒤에_ack한다() -> None:
    ch = FakeChannel()
    result = Consumer(ch, lambda req: OK_RESULT).on_message(SimpleNamespace(delivery_tag=7), _event_body())
    assert result == "ok"
    assert [c[0] for c in ch.calls] == ["publish", "ack"]
    exchange, key, body, props = ch.calls[0][1]
    assert (exchange, key, body["eventId"]) == ("trace.events", OUT_KEY, pid("out"))
    assert props.message_id == pid("out") and props.delivery_mode == 2 and props.content_type == "application/json"


def test_형식이_틀린_메시지는_다시_하지_않고_DLQ로_보낸다() -> None:
    ch = FakeChannel()
    calls = []
    consumer = Consumer(ch, lambda req: calls.append(req) or OK_RESULT)
    assert consumer.on_message(SimpleNamespace(delivery_tag=1), b"not json") == "invalid"
    assert consumer.on_message(SimpleNamespace(delivery_tag=2), b'{"schemaVersion":"1.0"}') == "invalid"
    assert ch.calls == [("reject", (1, False)), ("reject", (2, False))] and calls == []


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
