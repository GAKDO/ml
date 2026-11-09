"""`trace-ml retrain status|gate|promote|rollback|current`."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

from ..dataset.review_cli import _date, _source
from ..dataset.review_export import ReviewExportError
from ..labelstudio.config import ConfigError
from .notify import Post, notify
from .pipeline import Entry, ModelRegistry, RetrainError, count, gate, model_files, promote, rollback
from .release import write_release

DEFAULT_REGISTRY = Path(os.path.expanduser("~/.cache/trace-ml/models/registry.json"))


def add_parsers(sub: argparse._SubParsersAction) -> None:
    rt = sub.add_parser("retrain", help="검수 누적 → 재학습 → 게이트 → 레지스트리 → 서버·Jetson 동시 교체")
    rt.add_argument("--registry", type=Path, help="모델 레지스트리 JSON(기본 TRACE_MODEL_REGISTRY 또는 ~/.cache)")
    c = rt.add_subparsers(dest="command", required=True)
    s = c.add_parser("status", help="지난 재학습 뒤 쌓인 검수 수와 재학습 기준 도달 여부")
    s.add_argument("--rows", type=Path, help="review sql로 뽑은 CSV. 없으면 CAPTURE_DATABASE_URL")
    s.add_argument("--since", type=_date, help="이 날짜 뒤만 센다(기본: 레지스트리의 마지막 재학습 검수 시각)")
    s.add_argument("--min-photos", type=int, default=120, help="재학습 기준: 새로 검수된 사진 수")
    s.add_argument("--min-confirmed", type=int, default=20, help="재학습 기준: 새로 확정된 흠집 수")
    s.add_argument("--json", action="store_true", help="결과를 JSON 한 줄로(예약 작업용)")
    s.add_argument("--notify", action="store_true",
                   help="기준에 도달하면 TRACE_ALERT_WEBHOOK_URL(Discord·Slack 웹훅)로 운영자에게 알린다")
    g = c.add_parser("gate", help="같은 val로 잰 evaluate 두 개를 비교해 새 모델이 나을 때만 통과")
    g.add_argument("--baseline", type=Path, required=True, help="현재 모델 evaluate JSON")
    g.add_argument("--candidate", type=Path, required=True, help="새 모델 evaluate JSON(--model-version 필수)")
    g.add_argument("--min-recall-gain", type=float, default=0.0, help="사진 재현율이 이만큼 이상 올라야 통과")
    g.add_argument("--out", type=Path, required=True, help="gate.json")
    p = c.add_parser("promote", help="게이트를 통과한 모델을 현재 모델로 올리고 서버·Jetson 배포 묶음을 만든다")
    p.add_argument("--model-dir", type=Path, required=True, help="best.onnx·model.json 폴더(train의 registered/ 등)")
    p.add_argument("--gate", type=Path, help="gate.json. 처음(레지스트리가 빌 때) 지금 모델을 올릴 때만 생략")
    p.add_argument("--reviews", type=Path, help="이번 학습에 쓴 review export 폴더(다음 status의 기준 시각)")
    p.add_argument("--metrics", help="MODEL_METRICS 값(예: 0.216/0.073). 릴리스에 함께 적는다")
    p.add_argument("--notify", action="store_true", help="올린 뒤 웹훅으로 배포할 버전을 알린다")
    c.add_parser("rollback", help="현재 모델을 바로 앞 모델로 되돌린다(그 릴리스를 다시 배포)")
    c.add_parser("current", help="현재 모델과 교체 이력")


def _registry(args: argparse.Namespace, env: Mapping[str, str]) -> ModelRegistry:
    path = args.registry or (Path(env["TRACE_MODEL_REGISTRY"]) if env.get("TRACE_MODEL_REGISTRY") else DEFAULT_REGISTRY)
    return ModelRegistry(path)


def run(args: argparse.Namespace, env: Mapping[str, str], post: Post | None = None) -> int:
    try:
        return _run(args, env, post)
    except ConfigError as exc:
        print(f"설정 오류: {exc}", file=sys.stderr)
        return 2
    except (RetrainError, ReviewExportError, OSError, ValueError, KeyError) as exc:
        print(f"실패: {exc}", file=sys.stderr)
        return 1


def _alert(args: argparse.Namespace, env: Mapping[str, str], post: Post | None, text: str) -> None:
    if getattr(args, "notify", False):
        print(notify(env, f"[TRACE 모델] {text}", **({"post": post} if post else {})))


def _run(args: argparse.Namespace, env: Mapping[str, str], post: Post | None = None) -> int:
    reg = _registry(args, env)
    if args.command == "status":
        since = args.since or reg.last_reviews_until()
        st = count(_source(args, env).photos(None), since, min_photos=args.min_photos,
                   min_confirmed=args.min_confirmed)
        summary = (f"{st.since or '처음'} 이후 검수: 세트 {st.sets} · 사진 {st.photos}/{st.min_photos}"
                   f" · 흠집 맞음 {st.confirmed}/{st.min_confirmed} · 흠집 아님 {st.dismissed}")
        if st.ready:
            _alert(args, env, post, f"재학습 기준 도달. {summary}. 집 PC에서 review export → train → gate 순서로 진행")
        if args.json:
            print(json.dumps({"ready": st.ready, "since": st.since, "photos": st.photos, "sets": st.sets,
                              "confirmed": st.confirmed, "dismissed": st.dismissed}, ensure_ascii=False))
            return 0
        print(summary)
        if st.ready:
            print("재학습 기준 도달: deploy/training/README의 '재학습 파이프라인' 2단계부터 진행하세요")
        else:
            print("아직 기준 전: 시연·테스트로 검수를 더 쌓으세요")
        return 0
    if args.command == "gate":
        if args.out.exists():
            raise RetrainError(f"결과 파일이 이미 있음(덮어쓰지 않음): {args.out}")
        doc = gate(args.baseline, args.candidate, min_recall_gain=args.min_recall_gain)
        for ch in doc["checks"]:
            mark = "통과" if ch["ok"] else "실패"
            vals = "" if ch["baseline"] is None else f" {ch['baseline']} → {ch['candidate']}"
            print(f"  [{mark}] {ch['name']}{vals}")
        args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        verdict = "통과: retrain promote로 올리세요" if doc["passed"] else "불통과: 현재 모델을 그대로 둡니다"
        print(f"{doc['candidate']['modelVersion']} 게이트 {verdict} ({args.out})")
        return 0 if doc["passed"] else 3
    if args.command == "promote":
        gate_doc = json.loads(args.gate.read_text(encoding="utf-8")) if args.gate else None
        version, _, _ = model_files(args.model_dir)
        out = reg.path.parent / "releases" / version
        if out.exists():  # 레지스트리에 올리기 전에 막아, 묶음 없는 현재 모델이 생기지 않게 한다
            raise RetrainError(f"릴리스 폴더가 이미 있음(덮어쓰지 않음): {out}")
        def bundle(e: Entry) -> None:
            try:
                write_release(e, args.model_dir, out.parent, args.metrics)
            except BaseException:
                shutil.rmtree(out, ignore_errors=True)  # 반쯤 쓴 묶음이 다음 실행을 막지 않게
                raise

        entry = promote(reg, args.model_dir, gate_doc=gate_doc, reviews=args.reviews, release=str(out),
                        before_add=bundle)
        print(f"현재 모델: {entry.modelVersion} (앞 모델 {entry.previous or '없음'}, {entry.kind})")
        print(f"배포 묶음: {out}")
        print("  학교 서버: scp -r 이 폴더 → bash deploy-server.sh")
        print("  부스 Jetson: 같은 폴더 → bash deploy-jetson.sh")
        print("  두 곳 모두 '확인 끝'과 같은 해시 앞자리가 나와야 끝(모델 일치 원칙)")
        _alert(args, env, post, f"새 모델 {entry.modelVersion} 등록(앞 {entry.previous or '없음'}, onnx {entry.onnxSha256[:12]})."
               " 학교 서버와 부스 Jetson에 같은 묶음을 배포하세요")
        return 0
    if args.command == "rollback":
        entry = rollback(reg)
        print(f"현재 모델을 {entry.modelVersion}로 되돌림. 서버·Jetson에 {entry.release} 묶음을 다시 배포하세요")
        return 0
    cur = reg.current()
    print(f"현재 모델: {cur.modelVersion if cur else '없음'} ({reg.path})")
    for e in reg.history():
        print(f"  {e.promotedAt}  {e.kind:<9} {e.modelVersion}  onnx {e.onnxSha256[:12]}  앞 {e.previous or '-'}")
    return 0
