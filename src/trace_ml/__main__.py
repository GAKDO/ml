"""명령줄 진입점. 실제 기능은 이 패키지 아래 모듈로 더한다.

    trace-ml                      버전
    trace-ml labelstudio setup    프로젝트·소스(trace-raw)·타깃(trace-labels)을 없을 때만 만든다
    trace-ml labelstudio sync     새 사진을 작업으로 가져오고 주석을 trace-labels로 내보낸다
    trace-ml db init              ml 스키마 표 3개를 없을 때만 만든다
    trace-ml dataset device add|list   기기 → 분할(TRAIN·VAL·TEST)
    trace-ml dataset build|list   제출된 주석으로 데이터셋 버전을 만든다(덮어쓰기 없음)
    trace-ml train v001           데이터셋 버전으로 학습, 가중치·ONNX·지표를 버전과 함께 기록
    trace-ml models list          모델 → 데이터셋 버전
    trace-ml align photo|eval|pair    윤곽 4점 정합 확인·정합 오차(px·mm) 재기
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping

from . import __version__
from .dataset import cli as dataset_cli
from .inspect import cli as inspect_cli
from .labelstudio import setup as ls_setup
from .labelstudio.client import LabelStudio, LabelStudioError
from .labelstudio.config import ConfigError, Settings, load_settings
from .training import cli as training_cli


def make_client(settings: Settings) -> LabelStudio:
    return LabelStudio(settings.url, settings.api_key)


def _labelstudio(command: str, env: Mapping[str, str]) -> int:
    try:
        settings = load_settings(env)
    except ConfigError as exc:
        print(f"설정 오류: {exc}", file=sys.stderr)
        return 2
    ls = make_client(settings)
    try:
        if command == "setup":
            result = ls_setup.setup(ls, settings)
            made = ", ".join(result.created) or "없음(이미 있음)"
            print(f"프로젝트 id {result.project_id} · 새로 만든 것: {made}")
            for warning in result.warnings:
                print(f"주의: {warning}")
        else:
            synced = ls_setup.sync(ls, settings)
            print(f"프로젝트 id {synced.project_id} · 작업 {synced.tasks_before} → {synced.tasks_after}"
                  f" (+{synced.added}) · 주석 내보내기 완료")
    except LabelStudioError as exc:
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
    ls_parser.add_argument("command", choices=["setup", "sync"])
    dataset_cli.add_parsers(sub)
    training_cli.add_parsers(sub)
    inspect_cli.add_parsers(sub)
    args = parser.parse_args(args_list)
    environ = os.environ if env is None else env
    if args.tool == "labelstudio":
        return _labelstudio(args.command, environ)
    if args.tool == "align":
        return inspect_cli.run(args, environ)
    deps = deps or dataset_cli.Deps(environ)
    if args.tool in ("train", "models"):
        return training_cli.run(args, deps)
    return dataset_cli.run(args, deps)


if __name__ == "__main__":
    raise SystemExit(main())
