"""데이터셋 버전 → 집 PC 로컬 YOLO 폴더. 사진·라벨·manifest 해시를 모두 대조한다."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..registry import DatasetVersion
from ..settings import MinioSettings
from ..store import Store


class FetchError(Exception):
    """받은 내용이 버전을 만들 때와 다름. 학습을 시작하지 않는다."""

    def __init__(self, message: str, problems: list[str] | None = None) -> None:
        super().__init__(message)
        self.problems = problems or []


@dataclass(frozen=True)
class Fetched:
    data_yaml: Path
    manifest: dict
    manifest_sha256: str


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _place(path: Path, want: str, download) -> str | None:
    """이미 받은 파일이 같은 해시면 다시 받지 않는다. 다르면 해시를 돌려준다."""
    if path.exists() and _sha(path.read_bytes()) == want:
        return None
    body = download()
    if _sha(body) != want:
        return _sha(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return None


def fetch(store: Store, settings: MinioSettings, version: DatasetVersion, dest: Path) -> Fetched:
    raw = store.get(settings.datasets_bucket, version.manifest_key)
    sha = _sha(raw)
    expected = version.counts.get("manifestSha256")
    if expected and expected != sha:
        raise FetchError(f"{version.version} manifest가 기록과 다름(바뀐 것으로 보임)")
    manifest = json.loads(raw)
    root = dest / version.version
    problems: list[str] = []
    for item in manifest["items"]:
        img, label, split = item["image"], item["label"], item["split"]
        name = img["key"].rsplit("/", 1)[-1]
        if _place(root / "images" / split / name, img["sha256"],
                  lambda img=img: store.get(img["bucket"], img["key"])):
            problems.append(f"사진이 바뀜: {img['bucket']}/{img['key']}")
        stem = name.rsplit(".", 1)[0]
        if _place(root / "labels" / split / f"{stem}.txt", label["sha256"],
                  lambda label=label: store.get(settings.datasets_bucket, f"{version.version}/{label['key']}")):
            problems.append(f"라벨이 바뀜: {version.version}/{label['key']}")
    if problems:
        raise FetchError(f"{version.version}과 다른 파일 {len(problems)}개: 학습하지 않음", problems)
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
    data = {"path": str(root.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
            "names": dict(enumerate(manifest["classes"]))}
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return Fetched(data_yaml, manifest, sha)
