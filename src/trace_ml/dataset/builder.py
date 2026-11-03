"""build: 제출된 주석 → 기기 단위 분할 YOLO 데이터셋 버전(trace-datasets) + ml.dataset_version."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from ..codes import CODES_VERSION, DEFECT_CLASSES
from ..registry import DatasetVersion, Device, Registry, VersionExists
from ..settings import MinioSettings
from ..store import AlreadyExists, Store
from .annotations import AnnotationErrors, LabeledImage, collect
from .splits import assign

VERSION_RE = re.compile(r"^v[0-9]{3}$")
MANIFEST_SCHEMA = "trace.dataset.v1"


class BuildError(Exception):
    """빌드를 시작하지 않거나 멈춤. 이 오류가 나면 버전은 만들어지지 않는다."""


@dataclass(frozen=True)
class Commit:
    sha: str
    dirty: bool


@dataclass
class BuildResult:
    version: str
    manifest_key: str
    counts: dict
    new_devices: list[Device]


def git_commit(repo: Path | None = None) -> Commit:
    """빌더 코드의 커밋. 외부 명령은 이 함수 하나로 감싼다."""
    cwd = repo or Path(__file__).resolve().parent
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True,
                             check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=cwd,
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BuildError("빌더 커밋을 알 수 없음: ml 레포를 git clone해서 실행하세요") from exc
    return Commit(sha, bool(status))


def next_version(existing: list[str]) -> str:
    nums = [int(v[1:]) for v in existing if VERSION_RE.match(v)]
    return f"v{max(nums, default=0) + 1:03d}"


def data_yaml() -> str:
    """경로는 학습 때(train) 받은 폴더로 채운다. 클래스 번호 = codes.yaml 순서."""
    doc = {"train": "images/train", "val": "images/val", "test": "images/test",
           "names": dict(enumerate(DEFECT_CLASSES))}
    return yaml.safe_dump(doc, allow_unicode=True, sort_keys=False)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_exports(store: Store, bucket: str) -> list[dict]:
    docs, bad = [], []
    for key in store.keys(bucket):
        try:
            doc = json.loads(store.get(bucket, key))
        except (json.JSONDecodeError, UnicodeDecodeError):
            bad.append(f"{bucket}/{key}: JSON이 아님")
            continue
        if isinstance(doc, dict):
            docs.append(doc)
    if bad:
        raise AnnotationErrors(bad)
    return docs


def _counts(images: list[LabeledImage], table: dict[str, str]) -> dict:
    by_split = {s.lower(): 0 for s in ("TRAIN", "VAL", "TEST")}
    background = dict(by_split)
    instances = {c: 0 for c in DEFECT_CLASSES}
    for img in images:
        split = table[img.device].lower()
        by_split[split] += 1
        background[split] += 0 if img.polygons else 1
        for p in img.polygons:
            instances[DEFECT_CLASSES[p.cls]] += 1
    devices = {s: sorted(c for c, v in table.items() if v == s) for s in ("TRAIN", "VAL", "TEST")}
    return {"images": by_split, "background": background, "instances": instances, "devices": devices}


def build(store: Store, registry: Registry, settings: MinioSettings, commit: Commit, *,
          version: str | None = None, allow_dirty: bool = False, now: datetime | None = None,
          require_val: bool = False) -> BuildResult:
    if commit.dirty and not allow_dirty:
        raise BuildError("커밋하지 않은 변경이 있음: 커밋한 뒤 빌드하거나 --allow-dirty")
    images = collect(_load_exports(store, settings.labels_bucket))
    if not images:
        raise BuildError(f"{settings.labels_bucket}에 제출된 주석이 없음: Label Studio에서 sync 먼저")

    versions = registry.versions()
    name = version or next_version([v.version for v in versions])
    if not VERSION_RE.match(name):
        raise BuildError(f"버전 이름은 v001처럼 v + 숫자 3자리: {name!r}")
    if any(v.version == name for v in versions):
        raise VersionExists(f"데이터셋 버전 {name}이 이미 있음(덮어쓸 수 없음)")
    if next(store.keys(settings.datasets_bucket, f"{name}/"), None) is not None:
        raise AlreadyExists(f"{settings.datasets_bucket}/{name}/에 객체가 이미 있음(덮어쓸 수 없음)")

    table, new_devices = assign({img.device for img in images}, registry.devices())
    counts = _counts(images, table)
    # SC-001: 한 기기가 두 분할에 동시에 없다. table은 기기 → 분할 하나라 늘 참이지만 결과로 다시 확인한다
    placed = [c for codes in counts["devices"].values() for c in codes]
    if len(placed) != len(set(placed)):
        raise BuildError("한 기기가 두 분할에 있음")
    # 버전에 쓰인 기기는 분할을 바꿀 수 없으므로, val 없는 버전은 쓰기 전에 막는다(학습은 val이 있어야 돈다)
    if require_val and not counts["images"]["val"]:
        raise BuildError("val 사진이 없음: 아직 버전에 쓰이지 않은 기기 하나를 "
                         "trace-ml dataset device add <기기> --split VAL --model <기종> 으로 VAL에 두세요")

    # 먼저 모든 사진을 받아 해시를 낸다. 여기까지는 아무것도 쓰지 않는다
    items, files = [], []
    for img in images:
        split = table[img.device].lower()
        label = img.yolo_text().encode()
        label_key = f"labels/{split}/{img.stem}.txt"
        files.append((label_key, label, "text/plain"))
        items.append({
            "split": split, "device": img.device, "taskId": img.task_id, "annotationId": img.annotation_id,
            "image": {"bucket": img.bucket, "key": img.key, "sha256": _sha(store.get(img.bucket, img.key))},
            "label": {"key": label_key, "sha256": _sha(label), "instances": len(img.polygons)},
        })
    files.append(("data.yaml", data_yaml().encode(), "application/yaml"))

    built_at = (now or datetime.now(UTC)).replace(microsecond=0)
    manifest = {
        "schema": MANIFEST_SCHEMA, "version": name, "builtAt": built_at.isoformat().replace("+00:00", "Z"),
        "builderCommit": commit.sha, "builderDirty": commit.dirty, "codesVersion": CODES_VERSION,
        "classes": list(DEFECT_CLASSES), "labelsBucket": settings.labels_bucket,
        "devices": table, "counts": counts, "items": items,
    }
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
    counts["manifestSha256"] = _sha(manifest_bytes)

    for key, body, ctype in files:
        store.put_new(settings.datasets_bucket, f"{name}/{key}", body, ctype)
    manifest_key = f"{name}/manifest.json"
    # manifest는 마지막에 쓴다: manifest가 있으면 버전 파일이 모두 있다는 뜻
    store.put_new(settings.datasets_bucket, manifest_key, manifest_bytes, "application/json",
                  {"trace-dataset-version": name})
    registry.add_version(DatasetVersion(name, manifest_key, commit.sha, counts, built_at), new_devices)
    return BuildResult(name, manifest_key, counts, new_devices)

