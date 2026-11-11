"""`trace-ml review sql|export`: 사람 검수 결과를 재학습용 YOLO 폴더로 내보낸다."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from ..labelstudio.config import ConfigError
from ..settings import MinioSettings
from ..store import StoreError, make_store
from .review_export import (
    CsvReviewSource,
    Fetch,
    PgReviewSource,
    ReviewExportError,
    ReviewSource,
    export,
    psql_command,
)


def add_parsers(sub: argparse._SubParsersAction) -> None:
    rv = sub.add_parser("review", help="사람 검수 결과 → 재학습 데이터")
    rv_sub = rv.add_subparsers(dest="command", required=True)
    q = rv_sub.add_parser("sql", help="학교 서버에서 검수 결과 CSV를 뽑는 명령을 보인다")
    q.add_argument("--since", type=_date, help="이 날짜(UTC) 이후 검수 완료분만. 예: 2026-11-08")
    e = rv_sub.add_parser("export", help="검수 결과를 YOLO 분할 폴더(images·labels/train)로")
    e.add_argument("--rows", type=Path, help="review sql로 뽑은 CSV. 없으면 CAPTURE_DATABASE_URL로 직접 읽는다")
    e.add_argument("--out", type=Path, required=True, help="빈 폴더(예: D:\\trace\\cache\\review-20261112)")
    e.add_argument("--since", type=_date, help="이 날짜(UTC) 이후 검수 완료분만")


def _date(text: str) -> datetime:
    try:
        return datetime.fromisoformat(text).replace(tzinfo=UTC)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"날짜는 YYYY-MM-DD: {text!r}") from exc


def _source(args: argparse.Namespace, env: Mapping[str, str]) -> ReviewSource:
    if args.rows:
        return CsvReviewSource(args.rows)
    dsn = env.get("CAPTURE_DATABASE_URL")
    if not dsn:
        raise ConfigError("--rows CSV 또는 환경 변수 CAPTURE_DATABASE_URL이 필요함")
    return PgReviewSource(dsn)


def _fetch(env: Mapping[str, str]) -> tuple[Fetch, str]:
    """사진은 gakdo.site Compose MinIO(trace-captures)에 있다. 학습 키(학교 저장소)와 다른 키를 쓴다."""
    names = ("CAPTURE_MINIO_ENDPOINT", "CAPTURE_MINIO_ACCESS_KEY", "CAPTURE_MINIO_SECRET_KEY")
    missing = [n for n in names if not env.get(n)]
    if missing:
        raise ConfigError("환경 변수가 없음: " + ", ".join(missing))
    store = make_store(MinioSettings(endpoint=env[names[0]], access_key=env[names[1]], secret_key=env[names[2]]))
    return store.get, env.get("TRACE_CAPTURES_BUCKET") or "trace-captures"


def run(args: argparse.Namespace, env: Mapping[str, str],
        fetch: Callable[[Mapping[str, str]], tuple[Fetch, str]] = _fetch) -> int:
    if args.command == "sql":
        print("# 학교 서버 ~/gitops/compose에서 돌린 뒤 reviews.csv를 집 PC로 가져오세요(scp)")
        print(psql_command(args.since))
        return 0
    try:
        source = _source(args, env)
        get, bucket = fetch(env)
        result = export(source, get, bucket, args.out, since=args.since)
    except ConfigError as exc:
        print(f"설정 오류: {exc}", file=sys.stderr)
        return 2
    except (ReviewExportError, StoreError) as exc:
        print(f"실패: {exc}", file=sys.stderr)
        return 1
    confirmed = " · ".join(f"{k} {v}" for k, v in sorted(result.confirmed.items())) or "없음"
    dismissed = " · ".join(f"{k} {v}" for k, v in sorted(result.dismissed.items())) or "없음"
    print(f"사진 {result.photos}장: 흠집 라벨 {result.positives} · 음성 예시 {result.negatives} → {result.out_dir}")
    print(f"흠집 맞음 {confirmed} / 흠집 아님 {dismissed}")
    for problem in result.skipped:
        print(f"  건너뜀: {problem}")
    print(f"학습: trace-ml train <버전> --tile 640 --extra-train {result.out_dir}")
    return 0
