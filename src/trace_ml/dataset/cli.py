"""`trace-ml db|dataset` 명령. 저장소·기록부는 만드는 함수를 주입받아 테스트에서 바꿔 끼운다."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping

from ..labelstudio.config import ConfigError
from ..registry import SPLITS, Device, PgRegistry, Registry, RegistryError
from ..settings import load_db, load_minio
from ..store import Store, StoreError, make_store
from .annotations import AnnotationErrors
from .builder import BuildError, Commit, build, git_commit
from .splits import SplitError, check_change


def add_parsers(sub: argparse._SubParsersAction) -> None:
    db = sub.add_parser("db", help="ml 스키마 표 만들기")
    db.add_argument("command", choices=["init"])

    ds = sub.add_parser("dataset", help="데이터셋 버전(ML1)")
    ds_sub = ds.add_subparsers(dest="command", required=True)
    b = ds_sub.add_parser("build", help="제출된 주석으로 새 버전을 만든다")
    b.add_argument("--version", help="v001 형식. 없으면 다음 번호")
    b.add_argument("--allow-dirty", action="store_true", help="커밋하지 않은 변경이 있어도 만든다(manifest에 표시)")
    b.add_argument("--allow-no-val", action="store_true", help="val 기기가 없어도 만든다(학습은 못 함)")
    ds_sub.add_parser("list", help="버전 목록")
    dev = ds_sub.add_parser("device", help="기기 분할")
    dev_sub = dev.add_subparsers(dest="device_command", required=True)
    add = dev_sub.add_parser("add", help="기기를 등록하거나 고친다")
    add.add_argument("code")
    add.add_argument("--split", choices=SPLITS)
    add.add_argument("--model", help="기종 이름")
    add.add_argument("--color")
    dev_sub.add_parser("list", help="기기 → 분할")


class Deps:
    """명령이 쓰는 외부 자원. 기본값은 실제 PostgreSQL·MinIO·git."""

    def __init__(self, env: Mapping[str, str], registry: Callable[[], Registry] | None = None,
                 store: Callable[[], Store] | None = None, commit: Callable[[], Commit] | None = None,
                 trainer: Callable | None = None) -> None:
        self.env = env
        self.registry = registry or (lambda: PgRegistry(load_db(env)))
        self.store = store or (lambda: make_store(load_minio(env)))
        self.commit = commit or git_commit
        if trainer is None:
            from ..training.yolo import train_yolo

            trainer = train_yolo
        self.trainer = trainer


def run(args: argparse.Namespace, deps: Deps) -> int:
    try:
        return _run(args, deps)
    except AnnotationErrors as exc:
        print(f"빌드 중단: {exc}", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    except ConfigError as exc:
        print(f"설정 오류: {exc}", file=sys.stderr)
        return 2
    except (BuildError, SplitError, RegistryError, StoreError) as exc:
        print(f"실패: {exc}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace, deps: Deps) -> int:
    reg = deps.registry()
    if args.tool == "db":
        if not isinstance(reg, PgRegistry):
            raise RegistryError("db init은 PostgreSQL 기록부에서만")
        reg.init_schema()
        print("ml 표 3개 확인 완료(없으면 만듦)")
        return 0
    if args.command == "list":
        for v in reg.versions():
            imgs = v.counts.get("images", {})
            print(f"{v.version}  {v.built_at:%Y-%m-%d %H:%M}  {v.builder_commit[:8]}  "
                  f"train {imgs.get('train', 0)} · val {imgs.get('val', 0)} · test {imgs.get('test', 0)}")
        return 0
    if args.command == "device":
        known = reg.devices()
        if args.device_command == "list":
            for d in sorted(known.values(), key=lambda d: d.code):
                print(f"{d.code}  {d.split:5}  {d.model_name}  {d.color or ''}".rstrip())
            return 0
        before = known.get(args.code)
        split = args.split or (before.split if before else None)
        if split is None:
            raise SplitError("새 기기는 --split이 필요함")
        model = args.model or (before.model_name if before else None)
        if not model:
            raise SplitError("새 기기는 --model이 필요함")
        device = Device(args.code, model, split, args.color or (before.color if before else None))
        check_change(device, known, reg.used_devices())
        reg.save_device(device)
        print(f"{device.code} → {device.split}")
        return 0
    result = build(deps.store(), reg, load_minio(deps.env), deps.commit(),
                   version=args.version, allow_dirty=args.allow_dirty, require_val=not args.allow_no_val)
    imgs = result.counts["images"]
    print(f"{result.version} 만듦: train {imgs['train']} · val {imgs['val']} · test {imgs['test']}"
          f" · manifest {result.manifest_key}")
    for d in result.new_devices:
        print(f"새 기기 자동 배정: {d.code} → {d.split} (기종은 dataset device add {d.code} --model ...로 적기)")
    return 0
