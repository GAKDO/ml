"""`trace-ml train`, `trace-ml models list`."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ..dataset.cli import Deps
from ..labelstudio.config import ConfigError
from ..registry import RegistryError
from ..settings import load_minio
from ..store import StoreError
from .fetch import FetchError
from .record import train

DEFAULT_CACHE = Path(os.path.expanduser("~/.cache/trace-ml"))


def add_parsers(sub: argparse._SubParsersAction) -> None:
    t = sub.add_parser("train", help="데이터셋 버전으로 학습하고 결과를 기록한다(ML1)")
    t.add_argument("version", help="데이터셋 버전(v001)")
    t.add_argument("--base", default="yolo11s-seg.pt", help="기본 가중치")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--tile", type=int, default=640,
                   help="부스(BO2)와 같은 타일 크기로 잘라 학습(기본 640, 0이면 사진 통째로)")
    t.add_argument("--overlap", type=float, default=0.2, help="타일 겹침(부스와 같게 0.2)")
    t.add_argument("--background-ratio", type=float, default=1.0,
                   help="train에 둘 흠집 없는 타일 수 = 흠집 타일 수 × 이 값")
    t.add_argument("--batch", type=int, default=-1, help="-1이면 GPU 메모리에 맞춰 자동")
    t.add_argument("--imgsz", type=int, default=1280, help="--tile 0일 때만 씀. 타일 학습은 타일 크기")
    t.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="받은 데이터셋·학습 폴더")
    m = sub.add_parser("models", help="학습 기록")
    m.add_argument("command", choices=["list"])


def run(args: argparse.Namespace, deps: Deps) -> int:
    try:
        reg = deps.registry()
        if args.tool == "models":
            for r in reg.runs():
                mask = r.metrics.get("test", r.metrics.get("val", {})).get("mask", {})
                print(f"{r.dataset_version}  {r.model_key}  기본 {r.base_weights}  mask mAP50 {mask.get('map50', '-')}")
            return 0
        result = train(deps.store(), reg, load_minio(deps.env), args.version, base_weights=args.base,
                       epochs=args.epochs, imgsz=args.imgsz, cache_dir=args.cache, trainer=deps.trainer,
                       tile=args.tile, overlap=args.overlap, background_ratio=args.background_ratio,
                       batch=args.batch)
        print(f"학습 기록: {args.version} → {result.prefix} (best.pt·best.onnx·model.json·metrics.json·run.json)")
        return 0
    except FetchError as exc:
        print(f"학습 중단: {exc}", file=sys.stderr)
        for p in exc.problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    except ConfigError as exc:
        print(f"설정 오류: {exc}", file=sys.stderr)
        return 2
    except (RegistryError, StoreError, RuntimeError) as exc:
        print(f"실패: {exc}", file=sys.stderr)
        return 1
