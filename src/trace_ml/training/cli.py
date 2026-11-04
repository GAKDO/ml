"""`trace-ml train`, `trace-ml models list`."""

from __future__ import annotations

import argparse
import json
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
    r = sub.add_parser("rehearse", help="사진 없이 합성 사진으로 타일→학습→ONNX→model.json 끝까지 시험(로컬만)")
    r.add_argument("--out", type=Path, default=DEFAULT_CACHE / "rehearsal", help="결과 폴더(model/에 best.onnx·model.json)")
    r.add_argument("--photos", type=int, default=40, help="합성 사진 수(1920×1080, 70/20/10 분할)")
    r.add_argument("--epochs", type=int, default=30)
    r.add_argument("--base", default="yolo11s-seg.pt", help="기본 가중치(실제 학습과 같게)")
    r.add_argument("--batch", type=int, default=-1, help="-1이면 GPU 메모리에 맞춰 자동. CPU면 8 정도")


def run_rehearse(args: argparse.Namespace, rehearse_fn=None) -> int:
    """MinIO·DB 없이 돈다. 실패하면 어느 단계인지 보여 준다."""
    from .rehearse import rehearse

    fn = rehearse_fn or rehearse
    try:
        result = fn(args.out, photos=args.photos, epochs=args.epochs, base_weights=args.base, batch=args.batch)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"리허설 실패: {exc}", file=sys.stderr)
        return 1
    mask = result.metrics.get("val", {}).get("mask", {})
    print(f"타일: {json.dumps(result.counts, ensure_ascii=False)}")
    print(f"val mask mAP50 {mask.get('map50', '-')} (합성 사진이라 참고만)")
    print(f"ONNX 부스 모양 확인: {result.onnx_check}")
    print(f"Jetson으로 옮길 파일: {result.onnx}, {result.model_json} (testOnly: 서버에 안 올림)")
    return 0


def run(args: argparse.Namespace, deps: Deps) -> int:
    if args.tool == "train" and (args.tile < 0 or args.background_ratio < 0 or not 0 <= args.overlap < 1):
        print("설정 오류: --tile·--background-ratio는 0 이상, --overlap은 0 이상 1 미만", file=sys.stderr)
        return 2
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
