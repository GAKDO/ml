"""RabbitMQ 소비·발행과 Redis 진행률(집 PC inspect-worker 프로세스).

큐 trace.inspect.requested는 durable이라 집 PC가 꺼져 있던 동안의 이벤트가 쌓였다가 켜지면 처리된다(FR-OPS-01).
결과를 보낸 뒤(publisher confirm) ack한다. 형식 오류는 바로, 처리 오류는 몇 번 다시 해 본 뒤 trace.dlx로 보낸다.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..labelstudio.config import ConfigError
from ..store import Store
from .detect import Detector
from .faces import load_dims
from .features import load_refiner
from .worker import InspectRequest, InvalidMessage, Progress, WorkerConfig, parse_request, process

log = logging.getLogger(__name__)

EXCHANGE = "trace.events"
DLX = "trace.dlx"
QUEUE = "trace.inspect.requested"
DEAD_KEY = "trace.inspect.requested.dead"
IN_KEY = "trace.capture.uploaded"
OUT_KEY = "trace.inspect.completed"


@dataclass(frozen=True)
class AmqpSettings:
    host: str
    port: int = 5672
    vhost: str = "/"
    username: str = ""
    password: str = field(default="", repr=False)


@dataclass(frozen=True)
class RedisSettings:
    host: str
    port: int = 6379
    db: int = 0
    password: str = field(default="", repr=False)
    prefix: str = "trace:"


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name) or str(default)
    if not raw.isdigit():
        raise ConfigError(f"{name}는 숫자여야 함")
    return int(raw)


def load_amqp(env: Mapping[str, str]) -> AmqpSettings:
    """gitops infra.env·trace-secrets 이름 그대로."""
    missing = [n for n in ("RABBITMQ_HOST", "RABBITMQ_USERNAME", "RABBITMQ_PASSWORD") if not env.get(n)]
    if missing:
        raise ConfigError("환경 변수가 없음: " + ", ".join(missing))
    return AmqpSettings(env["RABBITMQ_HOST"], _int(env, "RABBITMQ_PORT", 5672), env.get("RABBITMQ_VHOST") or "/",
                        env["RABBITMQ_USERNAME"], env["RABBITMQ_PASSWORD"])


def load_redis(env: Mapping[str, str]) -> RedisSettings | None:
    if not env.get("REDIS_HOST"):
        return None
    return RedisSettings(env["REDIS_HOST"], _int(env, "REDIS_PORT", 6379), _int(env, "REDIS_DATABASE", 0),
                         env.get("REDIS_PASSWORD", ""), env.get("REDIS_KEY_PREFIX") or "trace:")


def load_worker_config(env: Mapping[str, str]) -> WorkerConfig:
    return WorkerConfig(dims=load_dims(env),
                        captures_bucket=env.get("TRACE_CAPTURES_BUCKET") or "trace-captures",
                        derived_bucket=env.get("TRACE_DERIVED_BUCKET") or "trace-derived")


class RedisProgress:
    """채널 {prefix}progress:{rentalId}에 JSON을 낸다. 실패해도 정합은 계속한다(진행률은 보조)."""

    def __init__(self, client: Any, prefix: str = "trace:") -> None:
        self._redis = client
        self._prefix = prefix

    def report(self, req: InspectRequest, stage: str, done: int, total: int) -> None:
        message = json.dumps({"captureSetId": req.capture_set_id, "checkpointId": req.checkpoint_id,
                              "stage": stage, "done": done, "total": total})
        try:
            self._redis.publish(f"{self._prefix}progress:{req.rental_id}", message)
        except Exception as exc:  # noqa: BLE001 - 진행률 실패로 작업을 멈추지 않는다
            log.warning("진행률 보내기 실패: %s", type(exc).__name__)


def make_redis_progress(settings: RedisSettings) -> RedisProgress:
    import redis

    client = redis.Redis(host=settings.host, port=settings.port, db=settings.db,
                         password=settings.password or None, socket_timeout=3)
    return RedisProgress(client, settings.prefix)


def declare(channel: Any) -> None:
    """교환기·큐·DLQ를 선언한다(rental-server와 같은 규칙: trace.dlx는 topic, 우리 것만 모이는 dead 키)."""
    channel.exchange_declare(exchange=EXCHANGE, exchange_type="topic", durable=True)
    channel.exchange_declare(exchange=DLX, exchange_type="topic", durable=True)
    channel.queue_declare(queue=QUEUE, durable=True,
                          arguments={"x-dead-letter-exchange": DLX, "x-dead-letter-routing-key": DEAD_KEY})
    channel.queue_bind(queue=QUEUE, exchange=EXCHANGE, routing_key=IN_KEY)
    channel.queue_declare(queue=DEAD_KEY, durable=True)
    channel.queue_bind(queue=DEAD_KEY, exchange=DLX, routing_key=DEAD_KEY)


Handler = Callable[[InspectRequest], dict[str, Any]]


class Consumer:
    """메시지 하나 = 세트 하나. 처리 → 발행(confirm) → ack."""

    def __init__(self, channel: Any, handler: Handler, attempts: int = 3, backoff: float = 2.0,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self._ch = channel
        self._handle = handler
        self._attempts = attempts
        self._backoff = backoff
        self._sleep = sleep

    def on_message(self, method: Any, body: bytes) -> str:
        """결과: 'ok' | 'invalid' | 'failed'. 테스트가 결과를 본다."""
        tag = method.delivery_tag
        try:
            req = parse_request(json.loads(body))
        except (InvalidMessage, ValueError) as exc:
            log.error("형식 오류로 DLQ: %s", exc)
            self._ch.basic_reject(delivery_tag=tag, requeue=False)
            return "invalid"
        for attempt in range(1, self._attempts + 1):
            try:
                event = self._handle(req)
                self.publish(event)
                self._ch.basic_ack(delivery_tag=tag)
                log.info("세트 %s 처리: 정합 %d, 실패 %d, 흠집 %d", req.capture_set_id, len(event["alignments"]),
                         len(event["failedPhotos"]), len(event["defects"]))
                return "ok"
            except Exception as exc:  # noqa: BLE001 - 어떤 오류든 몇 번 다시 해 본 뒤 DLQ
                log.warning("세트 %s 처리 실패(%d/%d): %s", req.capture_set_id, attempt, self._attempts, exc)
                if attempt < self._attempts:
                    self._sleep(self._backoff * 2 ** (attempt - 1))
        self._ch.basic_reject(delivery_tag=tag, requeue=False)
        return "failed"

    def publish(self, event: dict[str, Any]) -> None:
        import pika

        props = pika.BasicProperties(content_type="application/json", delivery_mode=2,
                                     message_id=event["eventId"], type=OUT_KEY)
        self._ch.basic_publish(exchange=EXCHANGE, routing_key=OUT_KEY,
                               body=json.dumps(event, ensure_ascii=False).encode(), properties=props,
                               mandatory=True)  # 받을 큐가 없으면(capture-server 큐 미선언) 예외 → 재시도·DLQ, 유실 없음

    def run(self, idle_timeout: float | None = None) -> int:
        """메시지를 처리한다. idle_timeout초 동안 아무것도 없으면 돌아온다(None이면 계속). 처리한 수를 돌려준다."""
        self._ch.basic_qos(prefetch_count=1)
        handled = 0
        for method, _props, body in self._ch.consume(QUEUE, inactivity_timeout=idle_timeout):
            if method is None:
                break
            self.on_message(method, body)
            handled += 1
        self._ch.cancel()
        return handled


def connect(settings: AmqpSettings) -> Any:
    import pika

    params = pika.ConnectionParameters(host=settings.host, port=settings.port, virtual_host=settings.vhost,
                                       credentials=pika.PlainCredentials(settings.username, settings.password),
                                       heartbeat=600, blocked_connection_timeout=300)
    return pika.BlockingConnection(params)


def serve(env: Mapping[str, str], store: Store, detector: Detector | None = None,
          idle_timeout: float | None = None, progress: Progress | None = None) -> int:
    """설정을 읽고 연결해 처리한다. 연결이 끊기면 잠시 뒤 다시 붙는다(idle_timeout이 없을 때)."""
    import pika

    config = load_worker_config(env)
    refiner = load_refiner(env)  # TRACE_ALIGN=features일 때만
    amqp = load_amqp(env)
    if progress is None:
        redis_settings = load_redis(env)
        progress = make_redis_progress(redis_settings) if redis_settings else None
    total = 0
    while True:
        try:
            conn = connect(amqp)
            channel = conn.channel()
            channel.confirm_delivery()
            declare(channel)
            consumer = Consumer(channel, lambda req: process(req, store, config, detector, progress, refiner=refiner))
            log.info("inspect-worker 시작: 큐 %s, 기본 기종 %s(이벤트 device가 없을 때)", QUEUE, config.dims.code)
            total += consumer.run(idle_timeout)
            conn.close()
            return total
        except pika.exceptions.AMQPConnectionError as exc:
            if idle_timeout is not None:
                raise
            log.warning("RabbitMQ 연결 끊김(%s), 10초 뒤 다시 붙음", type(exc).__name__)
            time.sleep(10)
