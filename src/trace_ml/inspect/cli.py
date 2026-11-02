"""inspect-worker·정합 명령.

    trace-ml inspect worker [--idle 초]   큐 trace.inspect.requested 처리(집 PC, 자동 시작)
    trace-ml align photo <사진> --face TOP [--out 펼친.png]
    trace-ml align eval [--n 100] [--faces TOP,SF,SL] [--json 결과.json]
    trace-ml align pair <부스 사진> <폰 사진> --face TOP
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping
from typing import Any

from ..labelstudio.config import ConfigError
from ..settings import MinioSettings, _get
from ..store import Store, make_store
from .faces import DEVICE_MODELS, FACES, face_size_px, load_dims


def add_parsers(sub: Any) -> None:
    ins = sub.add_parser("inspect", help="inspect-worker(ML2)")
    ins_sub = ins.add_subparsers(dest="command", required=True)
    w = ins_sub.add_parser("worker", help="trace.inspect.requested를 처리")
    w.add_argument("--idle", type=float, default=None, help="이 초 동안 메시지가 없으면 끝냄(기본: 계속)")

    al = sub.add_parser("align", help="윤곽 4점 정합 확인·오차 재기")
    al_sub = al.add_subparsers(dest="command", required=True)
    p = al_sub.add_parser("photo", help="사진 한 장을 펼친다")
    p.add_argument("image")
    p.add_argument("--face", required=True, choices=FACES)
    p.add_argument("--out", default="rectified.png")
    e = al_sub.add_parser("eval", help="합성 사진으로 꼭짓점 오차(px·mm)")
    e.add_argument("--n", type=int, default=100)
    e.add_argument("--faces", default="TOP,SF,SL")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--json", default=None)
    pr = al_sub.add_parser("pair", help="같은 면의 두 사진(부스·폰) 사이 정합 오차")
    pr.add_argument("image_a")
    pr.add_argument("image_b")
    pr.add_argument("--face", required=True, choices=FACES)
    pr.add_argument("--json", default=None)


def load_worker_minio(env: Mapping[str, str]) -> MinioSettings:
    """worker 역할 키(trace-worker)가 없으면 팀 키를 쓴다. 값은 보여 주지 않는다."""
    values = {
        "MINIO_ENDPOINT": _get(env, "MINIO_ENDPOINT"),
        "WORKER_MINIO_ACCESS_KEY": _get(env, "WORKER_MINIO_ACCESS_KEY", "MINIO_ACCESS_KEY"),
        "WORKER_MINIO_SECRET_KEY": _get(env, "WORKER_MINIO_SECRET_KEY", "MINIO_SECRET_KEY"),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ConfigError("환경 변수가 없음: " + ", ".join(missing))
    return MinioSettings(endpoint=values["MINIO_ENDPOINT"], access_key=values["WORKER_MINIO_ACCESS_KEY"],
                         secret_key=values["WORKER_MINIO_SECRET_KEY"])


def _read(path: str) -> Any:
    import cv2

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ConfigError(f"사진을 읽지 못함: {path}")
    return image


def _dump(result: dict[str, Any], path: str | None) -> None:
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


def run(args: argparse.Namespace, env: Mapping[str, str], store: Store | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        dims = load_dims(env)
        if args.tool == "inspect":
            from .runner import serve

            store = store or make_store(load_worker_minio(env))
            handled = serve(env, store, idle_timeout=args.idle)
            print(f"처리한 메시지 {handled}개")
            return 0
        if args.command == "photo":
            import cv2

            from .align import AlignFailure, align_photo

            result = align_photo(_read(args.image), face_size_px(args.face, dims))
            if isinstance(result, AlignFailure):
                print(f"실패: {result.reason}", file=sys.stderr)
                return 1
            if not cv2.imwrite(args.out, result.rectified):
                print(f"실패: 펼친 이미지를 쓰지 못함: {args.out}", file=sys.stderr)
                return 1
            print(json.dumps({"out": args.out, "errorPx": result.error_px,
                              "corners": result.corners.round(2).tolist()}, ensure_ascii=False))
            return 0
        if args.command == "eval":
            from .measure import synthetic

            faces = [f.strip() for f in args.faces.split(",") if f.strip()]
            bad = [f for f in faces if f not in FACES]
            if bad:
                raise ConfigError(f"모르는 면: {', '.join(bad)}")
            from .measure import synthetic_booth_half

            out = {"method": "CONTOUR", "deviceModel": dims.code, "n": args.n, "seed": args.seed,
                   "phone": {f: synthetic(f, dims, args.n, seed=args.seed).as_dict() for f in faces},
                   "boothHalf": {f"TOP-{half}": synthetic_booth_half("TOP", dims, args.n, half, seed=args.seed).as_dict()
                                 for half in ("front", "back")}}
            _dump(out, args.json)
            return 0
        from .measure import pair

        res = pair(_read(args.image_a), _read(args.image_b), args.face, dims)
        if isinstance(res, str):
            print(f"실패: {res}", file=sys.stderr)
            return 1
        _dump({"face": args.face, "deviceModel": dims.code, "matches": res.matches,
               "errorPxA": res.errors[0], "errorPxB": res.errors[1], "displacement": res.stats.as_dict()}, args.json)
        return 0
    except ConfigError as exc:
        print(f"설정 오류: {exc}", file=sys.stderr)
        return 2


__all__ = ["DEVICE_MODELS", "add_parsers", "load_worker_minio", "run"]
