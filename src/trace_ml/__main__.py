"""명령줄 진입점. 실제 기능은 이 패키지 아래 모듈로 더한다.

    trace-ml                      버전
    trace-ml labelstudio setup    프로젝트·소스(trace-raw)·타깃(trace-labels)을 없을 때만 만든다
    trace-ml labelstudio sync     새 사진을 작업으로 가져오고 주석을 trace-labels로 내보낸다
    trace-ml labelstudio mark-clean MB02 [MB03 ...] [--dry-run]   흠집 없는 기기의 남은 사진에 no_defect 제출
    trace-ml labelstudio review MB05 [MB07 ...] [--out 폴더] [--with-clean]   폴리곤 검수용 모아보기 JPG·summary.csv·zip
    trace-ml db init              ml 스키마 표 3개를 없을 때만 만든다
    trace-ml dataset device add|list   기기 → 분할(TRAIN·VAL·TEST)
    trace-ml dataset build|list   제출된 주석으로 데이터셋 버전을 만든다(덮어쓰기 없음)
    trace-ml train v001           데이터셋 버전으로 학습, 가중치·ONNX·지표를 버전과 함께 기록
    trace-ml models list          모델 → 데이터셋 버전
    trace-ml evaluate --weights best.pt --data <타일 폴더>   기기 면 판정(문제 있음/없음) 정확도
    trace-ml pretrain scan|build|train   AI Hub 등 공개 데이터로 사전 학습(부스 학습의 --base)
    trace-ml rehearse             사진 없이 합성 사진으로 학습→ONNX→model.json 끝까지 시험(로컬만)
    trace-ml inspect worker       inspect-worker(ML2): 정합·펼친 면·텍스처 → trace.inspect.completed
    trace-ml align photo|eval|pair    윤곽 4점 정합 확인·정합 오차(px·mm) 재기
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from pathlib import Path

from . import __version__
from .dataset import cli as dataset_cli
from .inspect import cli as inspect_cli
from .labelstudio import clean as ls_clean
from .labelstudio import review as ls_review
from .labelstudio import setup as ls_setup
from .labelstudio.client import LabelStudio, LabelStudioError
from .labelstudio.config import ConfigError, Settings, load_settings
from .pretrain import cli as pretrain_cli
from .store import StoreError
from .training import cli as training_cli


def make_client(settings: Settings) -> LabelStudio:
    return LabelStudio(settings.url, settings.api_key)


def make_review_fetch(settings: Settings) -> ls_review.Fetch:
    return ls_review.minio_fetch(settings)


def _labelstudio(command: str, env: Mapping[str, str], devices: list[str] | None = None,
                 dry_run: bool = False, out: str = "labelstudio-review", include_clean: bool = False) -> int:
    try:
        settings = load_settings(env)
    except ConfigError as exc:
        print(f"설정 오류: {exc}", file=sys.stderr)
        return 2
    if command in ("setup", "sync") and devices:
        print(f"기기 ID는 mark-clean·review에서만 씁니다 ({command}는 받지 않음)", file=sys.stderr)
        return 2
    if command != "review" and include_clean:
        print("--with-clean은 review에서만 씁니다", file=sys.stderr)
        return 2
    if command != "mark-clean" and dry_run:
        print(f"--dry-run은 mark-clean에서만 씁니다 ({command}는 받지 않음)", file=sys.stderr)
        return 2
    if command in ("mark-clean", "review") and not devices:
        print(f"기기 ID를 하나 이상 주세요 (예: trace-ml labelstudio {command} MB05 MB07)", file=sys.stderr)
        return 2
    ls = make_client(settings)
    try:
        if command == "setup":
            result = ls_setup.setup(ls, settings)
            made = ", ".join(result.created) or "없음(이미 있음)"
            print(f"프로젝트 id {result.project_id} · 새로 만든 것: {made}")
            for warning in result.warnings:
                print(f"주의: {warning}")
        elif command == "review":
            reviewed = ls_review.review_sheets(ls, settings, devices or [], Path(out),
                                               make_review_fetch(settings), include_clean=include_clean)
            for device, (drawn, clean, todo) in reviewed.counts.items():
                print(f"{device}: 폴리곤 {drawn}장 · no_defect {clean}장 · 아직 안 함 {todo}장")
            print(f"모아보기 {reviewed.sheets}장 → {reviewed.out_dir} · 압축 {reviewed.zip_path}")
        elif command == "mark-clean":
            cleaned = ls_clean.mark_clean(ls, settings, devices, dry_run=dry_run)
            verb = "제출할 예정" if dry_run else "제출"
            for device, (made, skipped) in cleaned.counts.items():
                note = " ← 사진이 없음, 기기 ID 확인" if made == 0 and skipped == 0 else ""
                print(f"{device}: no_defect {verb} {made}장 · 이미 주석 있어 건너뜀 {skipped}장{note}")
            if not dry_run:
                print("trace-labels에 저장하려면 trace-ml labelstudio sync를 실행하세요")
        else:
            synced = ls_setup.sync(ls, settings)
            print(f"프로젝트 id {synced.project_id} · 작업 {synced.tasks_before} → {synced.tasks_after}"
                  f" (+{synced.added}) · 주석 내보내기 완료")
    except (LabelStudioError, StoreError) as exc:
        print(f"실패: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None, env: Mapping[str, str] | None = None,
         deps: dataset_cli.Deps | None = None) -> int:
    args_list = sys.argv[1:] if argv is None else argv
    if not args_list:
        print(f"trace-ml {__version__}")
        return 0
    parser = argparse.ArgumentParser(prog="trace-ml")
    sub = parser.add_subparsers(dest="tool", required=True)
    ls_parser = sub.add_parser("labelstudio", help="Label Studio 연동(ML3)")
    ls_parser.add_argument("command", choices=["setup", "sync", "mark-clean", "review"])
    ls_parser.add_argument("devices", nargs="*", help="mark-clean: 흠집 없는 기기 ID (예: MB02 MB03)")
    ls_parser.add_argument("--dry-run", action="store_true", help="mark-clean: 제출하지 않고 장수만 보인다")
    ls_parser.add_argument("--out", default="labelstudio-review", help="review: 모아보기를 쓸 폴더")
    ls_parser.add_argument("--with-clean", action="store_true", help="review: no_defect 사진도 넣어 빠진 흠집을 본다")
    dataset_cli.add_parsers(sub)
    training_cli.add_parsers(sub)
    inspect_cli.add_parsers(sub)
    pretrain_cli.add_parsers(sub)
    args = parser.parse_args(args_list)
    environ = os.environ if env is None else env
    if args.tool == "labelstudio":
        return _labelstudio(args.command, environ, args.devices, args.dry_run, args.out, args.with_clean)
    if args.tool in ("inspect", "align"):
        return inspect_cli.run(args, environ)
    if args.tool == "pretrain":
        return pretrain_cli.run(args)
    if args.tool == "rehearse":
        return training_cli.run_rehearse(args)
    if args.tool == "evaluate":
        return training_cli.run_evaluate(args)
    deps = deps or dataset_cli.Deps(environ)
    if args.tool in ("train", "models"):
        return training_cli.run(args, deps)
    return dataset_cli.run(args, deps)


if __name__ == "__main__":
    raise SystemExit(main())
