"""`trace-ml pretrain scan|build|train`. MinIO·DB 없이 로컬 폴더만 쓴다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..codes import DEFECT_CLASSES
from ..training.tiles import tile_dataset
from ..training.yolo import TrainSpec, train_yolo
from .aihub import index_images, read_items
from .build import MapError, build, load_map, scan


def add_parsers(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("pretrain", help="AI Hub 등 공개 데이터로 사전 학습(부스 학습의 --base)")
    ps = p.add_subparsers(dest="command", required=True)
    s = ps.add_parser("scan", help="라벨 JSON을 읽어 라벨 이름·개수를 보여 준다(map.yaml 만들기 전)")
    s.add_argument("roots", nargs="+", type=Path, help="AI Hub 데이터 폴더(원천·라벨링 폴더가 모두 들어 있는 위쪽)")
    s.add_argument("--write-map", type=Path, help="찾은 라벨 이름으로 map.yaml 초안을 쓴다(값은 null)")
    b = ps.add_parser("build", help="YOLO-seg 데이터셋 + 640 타일을 만든다")
    b.add_argument("roots", nargs="+", type=Path)
    b.add_argument("--map", type=Path, required=True, help="라벨 이름 → scratch|dent|crack|stain|null")
    b.add_argument("--out", type=Path, required=True, help="내장 SSD 폴더(예: D:\\trace\\pretrain\\aihub)")
    b.add_argument("--max-images", type=int, default=0, help="0이면 전부")
    b.add_argument("--val", type=float, default=0.1)
    b.add_argument("--keep-empty", type=float, default=0.1, help="흠집 없는 사진을 남길 비율")
    b.add_argument("--max-side", type=int, default=2048,
                   help="긴 변을 이 크기로 줄인다(AI Hub 4032px → 부스 사진 배율에 가깝게, 타일·디스크 1/4). 0이면 원본")
    b.add_argument("--tile", type=int, default=640)
    b.add_argument("--overlap", type=float, default=0.2)
    b.add_argument("--background-ratio", type=float, default=1.0)
    t = ps.add_parser("train", help="build 결과로 사전 학습")
    t.add_argument("dataset", type=Path, help="build의 --out")
    t.add_argument("--model", default="yolo11m-seg.pt", help="COCO 가중치에서 출발. 클수록 정확·느림(s·m·l)")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch", type=int, default=-1)
    t.add_argument("--patience", type=int, default=20, help="val이 이만큼 좋아지지 않으면 일찍 멈춤")
    t.add_argument("--name", default="pretrain")


def run(args: argparse.Namespace) -> int:
    try:
        if args.command == "scan":
            return _scan(args)
        if args.command == "build":
            return _build(args)
        return _train(args)
    except (MapError, OSError, ValueError, RuntimeError) as exc:
        print(f"실패: {exc}", file=sys.stderr)
        return 1


def _scan(args: argparse.Namespace) -> int:
    r = scan(read_items(args.roots))
    print(f"JSON {r.files}개 · 사진 항목 {r.items} · 폴리곤 있는 사진 {r.with_shapes} · 사진 못 찾음 {r.no_image}"
          f" · 상자만 있는 객체 {r.boxes_only}")
    for label, n in r.labels.most_common():
        print(f"  {n:>8}  {label}")
    if not r.labels:
        print("폴리곤 라벨을 찾지 못함: 라벨 JSON 하나를 Claude에게 보여 주세요", file=sys.stderr)
        return 1
    if args.write_map:
        draft = "".join(f"{json.dumps(k, ensure_ascii=False)}: null\n" for k, _ in r.labels.most_common())
        args.write_map.write_text("# 값: scratch | dent | crack | stain | null(버림)\n" + draft, encoding="utf-8")
        print(f"map 초안: {args.write_map}")
    return 0


def _build(args: argparse.Namespace) -> int:
    mapping = load_map(args.map)
    raw = args.out / "raw"
    report = build(read_items(args.roots, index_images(args.roots)), mapping, raw, val=args.val,
                   max_images=args.max_images, keep_empty=args.keep_empty, max_side=args.max_side)
    print(f"사진 {report.images} · 폴리곤 {dict(report.shapes)}")
    if report.dropped_labels:
        print(f"버린 라벨: {dict(report.dropped_labels.most_common(10))}")
    if report.skipped:
        print(f"건너뜀: {dict(report.skipped)}")
    if not report.images["val"] or not sum(report.shapes.values()):
        raise ValueError("val 사진이나 매핑된 흠집이 없음: map.yaml을 확인")
    tiled = tile_dataset(raw, args.out / f"tile{args.tile}", dict(enumerate(DEFECT_CLASSES)), size=args.tile,
                         overlap=args.overlap, background_ratio=args.background_ratio)
    (args.out / "build.json").write_text(json.dumps({"tile": args.tile, "overlap": args.overlap,
                                                     "counts": tiled.counts}, indent=2), encoding="utf-8")
    print(f"타일: {tiled.counts} → {tiled.data_yaml}")
    return 0


def _train(args: argparse.Namespace) -> int:
    info = json.loads((args.dataset / "build.json").read_text(encoding="utf-8"))
    data_yaml = args.dataset / f"tile{info['tile']}" / "data.yaml"
    has_test = bool(info["counts"]["test"]["tiles"])
    out = train_yolo(TrainSpec(data_yaml.resolve(), args.model, args.epochs, info["tile"],
                               (args.dataset / "runs").resolve(), args.name, has_test, args.batch,
                               extra={"patience": args.patience, "cos_lr": True, "close_mosaic": 10}))
    print(f"사전 학습 가중치: {out.weights}")
    print(f"val mask mAP50 {out.metrics['val']['mask']['map50']:.3f}")
    print(f"다음: trace-ml train v001 --base {out.weights}")
    return 0
