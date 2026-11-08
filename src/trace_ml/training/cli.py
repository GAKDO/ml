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
from .yolo import existing_run

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
    t.add_argument("--extra-train", type=Path,
                   help="trace-ml review export 폴더(사람 검수 결과). train에만 더하고 run.json에 남긴다")
    # 데이터가 적어 몇 에포크 만에 과적합할 때 쓰는 옵션. 주지 않으면 ultralytics 기본값 그대로
    t.add_argument("--freeze", type=int, help="앞쪽 레이어 N개를 얼린다(사전학습 특징 유지, 예: 10=백본)")
    t.add_argument("--lr0", type=float, help="시작 학습률. 주면 optimizer auto 대신 AdamW로 고정(auto는 lr0를 무시)")
    t.add_argument("--optimizer", choices=["auto", "SGD", "AdamW"], help="기본 auto")
    t.add_argument("--patience", type=int, help="val이 좋아지지 않으면 이만큼 뒤에 멈춘다(기본 100)")
    t.add_argument("--cos-lr", action="store_true", help="코사인 학습률 감소")
    t.add_argument("--model-name", help="model.json modelVersion 앞부분(기본: --base 파일 이름). 예: yolo11m-aihub")
    t.add_argument("--from-run", type=Path,
                   help="학습하지 않고, yolo로 따로 학습한 폴더(weights/best.pt·best.onnx)를 이 버전으로 기록한다")
    t.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                   help="ultralytics train 인자를 그대로 넘긴다(여러 번 가능). 예: --set degrees=10 --set mixup=0.1")
    e = sub.add_parser("evaluate", help="기기 면 판정(문제 있음/없음)이 라벨과 맞는지 센다. MinIO·DB 없이 돈다")
    e.add_argument("--weights", type=Path, required=True, help="best.pt")
    e.add_argument("--data", type=Path, required=True, help="타일 데이터셋 폴더(예: D:\\trace\\cache\\datasets\\v001-tile640)")
    e.add_argument("--split", default="val", choices=["train", "val", "test"])
    e.add_argument("--conf", type=float, default=0.25, help="검출로 셀 최소 신뢰도")
    e.add_argument("--min-photos", type=int, default=1,
                   help="기기 면을 '문제 있음'으로 볼 최소 사진 수. 2 이상이면 한 장에만 잡힌 먼지·반사를 거른다")
    e.add_argument("--model-version", help="결과에 남길 model.json modelVersion(가중치 경로·해시는 항상 남김)")
    e.add_argument("--out", type=Path, help="결과 JSON을 쓸 파일")
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


def train_extra(args: argparse.Namespace) -> dict[str, object]:
    """주어진 옵션만 ultralytics train 인자로 넘긴다."""
    extra: dict[str, object] = {}
    if args.freeze is not None:
        extra["freeze"] = args.freeze
    if args.patience is not None:
        extra["patience"] = args.patience
    if args.cos_lr:
        extra["cos_lr"] = True
    optimizer = args.optimizer
    if args.lr0 is not None:
        extra["lr0"] = args.lr0
        optimizer = optimizer if optimizer not in (None, "auto") else "AdamW"
    if optimizer:
        extra["optimizer"] = optimizer
    extra.update(parse_overrides(getattr(args, "overrides", None) or []))
    return extra


# 학습 코드가 직접 정하는 인자(바꾸면 기록·타일·클래스 이름이 어긋남)와 전용 옵션이 있는 인자
# resume는 --base 체크포인트의 옛 데이터셋으로 이어 학습해 기록과 어긋난다
_RESERVED = {"data", "model", "epochs", "imgsz", "batch", "project", "name", "exist_ok", "single_cls", "classes",
             "resume"}
_DEDICATED = {"freeze": "--freeze", "lr0": "--lr0", "optimizer": "--optimizer", "patience": "--patience",
              "cos_lr": "--cos-lr"}


def _value(text: str) -> object:
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def parse_overrides(items: list[str]) -> dict[str, object]:
    """`--set KEY=VALUE` 목록을 ultralytics 인자 dict로. 숫자·true/false는 그 타입으로 바꾼다."""
    out: dict[str, object] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or not key or not raw.strip():
            raise ConfigError(f"--set은 KEY=VALUE 형식: {item!r}")
        if key in _RESERVED:
            raise ConfigError(f"--set {key}는 학습 코드가 정하는 값이라 바꿀 수 없음")
        value = _value(raw.strip())
        if key == "save" and value is False:
            raise ConfigError("--set save=false는 조기 종료 때 가중치가 남지 않아 기록할 수 없음")
        if key in _DEDICATED:
            raise ConfigError(f"--set {key} 대신 {_DEDICATED[key]}를 쓰세요")
        if key in out:
            raise ConfigError(f"--set {key}를 두 번 줌")
        out[key] = value
    return out


def run_evaluate(args: argparse.Namespace, predict=None) -> int:
    from .evaluate import EvaluateError
    from .evaluate import run_evaluate as evaluate_fn

    if not 0 < args.conf < 1 or args.min_photos < 1:
        print("설정 오류: --conf는 0과 1 사이, --min-photos는 1 이상", file=sys.stderr)
        return 2
    if args.out and args.out.exists():
        # 긴 추론 뒤에 이전 결과(다른 가중치의 점수)를 덮어쓰지 않도록 먼저 멈춘다
        print(f"설정 오류: 결과 파일이 이미 있음(덮어쓰지 않음): {args.out}", file=sys.stderr)
        return 2
    try:
        kwargs = {"predict": predict} if predict else {}
        report = evaluate_fn(args.weights, args.data, split=args.split, conf=args.conf,
                             min_photos=args.min_photos, model_version=getattr(args, "model_version", None),
                             **kwargs)
    except (EvaluateError, RuntimeError, OSError) as exc:
        print(f"평가 실패: {exc}", file=sys.stderr)
        return 1
    print(f"모델 {report.model_version or '-'} · {report.weights} (sha256 {report.weights_sha256[:12]})")
    print(f"{args.split} · conf {args.conf} · 기기 면은 흠집 사진 {args.min_photos}장 이상이면 '문제 있음'")
    print("기기 면           라벨  판정  사진(라벨/검출/전체)  검출 클래스")
    for r in report.faces:
        mark = "" if r.truth == r.pred else "  ← 틀림"
        classes = ", ".join(f"{k} {v}" for k, v in sorted(r.classes.items())) or "-"
        print(f"{r.face:<16} {'있음' if r.truth else '없음':>4}  {'있음' if r.pred else '없음':>4}  "
              f"{r.labeled_photos:>4}/{r.detected_photos:>4}/{r.photos:<4}        {classes}{mark}")
    for unit, s in (("사진", report.photo), ("기기 면", report.face)):
        print(f"{unit}: 정확도 {s['accuracy']} · 정밀도 {s['precision']} · 재현율 {s['recall']}"
              f" (TP {s['tp']} · FP {s['fp']} · FN {s['fn']} · TN {s['tn']})")
    print("문턱별(기기 면 정확도 / 사진 정밀도·재현율):")
    for row in report.sweep:
        print(f"  conf {row['conf']:.2f}: {row['face']['accuracy']} / {row['photo']['precision']}·{row['photo']['recall']}")
    if args.out:
        args.out.write_text(report.to_json(), encoding="utf-8")
        print(f"결과: {args.out}")
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
        from_run = getattr(args, "from_run", None)
        if from_run and (args.extra_train or not from_run.is_dir()):
            # 가중치가 쓰지 않은 검수분을 run.json에 적으면 기록이 틀린다
            raise ConfigError("--from-run은 있는 폴더여야 하고 --extra-train과 같이 쓰지 않음")
        trainer = existing_run(from_run) if from_run else deps.trainer
        result = train(deps.store(), reg, load_minio(deps.env), args.version, base_weights=args.base,
                       epochs=args.epochs, imgsz=args.imgsz, cache_dir=args.cache, trainer=trainer,
                       tile=args.tile, overlap=args.overlap, background_ratio=args.background_ratio,
                       batch=args.batch, extra=train_extra(args), model_name=args.model_name,
                       extra_train=args.extra_train, imported_from=from_run)
        print(f"학습 기록: {args.version} → {result.prefix} (best.pt·best.onnx·model.json·metrics.json·run.json)")
        if from_run:
            print(f"부스·서버로 옮길 파일: {from_run / 'registered'} (best.onnx·model.json)")
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
