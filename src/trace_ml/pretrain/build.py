"""AI Hub Item → YOLO-seg 원본 데이터셋(images|labels/<분할>) → 부스와 같은 640 타일.

- 클래스 표(map.yaml): AI Hub 라벨 이름 → 우리 클래스(scratch·dent·crack·stain) 또는 null(버림)
- 분할은 사진 파일 이름 해시로 고정(다시 빌드해도 같은 사진은 같은 분할)
- 사진은 jpg로 내장 SSD에 복사한다(외장 HDD에서 바로 학습하면 읽기가 느려 GPU가 논다)
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from ..codes import DEFECT_CLASSES
from .aihub import Item


class MapError(Exception):
    """클래스 표가 잘못됨(우리 클래스가 아닌 값)."""


@dataclass
class ScanReport:
    files: int = 0
    unreadable: int = 0
    items: int = 0
    no_image: int = 0
    with_shapes: int = 0
    boxes_only: int = 0
    labels: Counter = field(default_factory=Counter)


@dataclass
class BuildReport:
    images: dict[str, int] = field(default_factory=lambda: {"train": 0, "val": 0, "test": 0})
    shapes: Counter = field(default_factory=Counter)
    dropped_labels: Counter = field(default_factory=Counter)
    skipped: Counter = field(default_factory=Counter)


def scan(items: Iterable[Item]) -> ScanReport:
    r = ScanReport()
    sources: set[Path] = set()
    for it in items:
        if it.source:
            sources.add(it.source)
        if it.image is None and not it.shapes:
            r.unreadable += it.source is not None and it.width == 0 and not it.boxes_only
        r.items += 1
        r.no_image += it.image is None
        r.boxes_only += it.boxes_only
        if it.shapes:
            r.with_shapes += 1
        r.labels.update(s.label for s in it.shapes)
    r.files = len(sources)
    return r


def load_map(path: Path) -> dict[str, str | None]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise MapError("map.yaml은 '라벨 이름: 클래스' 표여야 함")
    out: dict[str, str | None] = {}
    for k, v in raw.items():
        if v is not None and v not in DEFECT_CLASSES:
            raise MapError(f"{k}: {v}는 우리 클래스가 아님({', '.join(DEFECT_CLASSES)} 또는 null)")
        out[str(k).strip()] = v
    return out


def split_of(path: Path, val: float, test: float) -> str:
    """파일 이름 해시로 분할. 외장하드 드라이브 글자가 바뀌거나 폴더를 옮겨도 같은 사진은 같은 분할."""
    h = int(hashlib.sha256(path.name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val else "test" if h < val + test else "train"


def _read(path: Path) -> np.ndarray | None:
    import cv2

    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)  # 한글 경로


def build(items: Iterable[Item], mapping: dict[str, str | None], dest: Path, *, val: float = 0.1,
          test: float = 0.05, max_images: int = 0, keep_empty: float = 0.1) -> BuildReport:
    """dest/images|labels/<분할>/*.jpg|txt. 매핑된 흠집이 없는 사진은 keep_empty 비율만 둔다(오탐 학습용)."""
    import cv2

    report = BuildReport()
    for split in ("train", "val", "test"):
        (dest / "images" / split).mkdir(parents=True, exist_ok=True)
        (dest / "labels" / split).mkdir(parents=True, exist_ok=True)
    total = 0
    for it in items:
        if max_images and total >= max_images:
            break
        if it.image is None:
            report.skipped["사진 없음"] += 1
            continue
        kept = []
        for s in it.shapes:
            cls = mapping.get(s.label)
            if cls is None:
                report.dropped_labels[s.label] += 1
            else:
                kept.append((DEFECT_CLASSES.index(cls), s.points))
        if not kept and split_of(it.image, keep_empty, 0) != "val":  # 빈 사진은 keep_empty 비율만
            report.skipped["흠집 없음(일부만 둠)"] += 1
            continue
        image = _read(it.image)
        if image is None:
            report.skipped["사진 읽기 실패"] += 1
            continue
        h, w = image.shape[:2]
        sx = w / it.width if it.width else 1.0  # JSON 크기와 실제 사진 크기가 다르면 맞춘다
        sy = h / it.height if it.height else 1.0
        lines = []
        for cls, pts in kept:
            coords = [(min(max(x * sx / w, 0.0), 1.0), min(max(y * sy / h, 0.0), 1.0)) for x, y in pts]
            lines.append(f"{cls} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in coords))
            report.shapes[DEFECT_CLASSES[cls]] += 1
        split = split_of(it.image, val, test)
        stem = hashlib.sha256(str(it.image).encode()).hexdigest()[:16]  # 데이터마다 같은 파일 이름이 있어도 안 겹치게
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            report.skipped["jpg 쓰기 실패"] += 1
            continue
        (dest / "images" / split / f"{stem}.jpg").write_bytes(buf.tobytes())
        (dest / "labels" / split / f"{stem}.txt").write_text("".join(f"{x}\n" for x in lines), encoding="utf-8")
        report.images[split] += 1
        total += 1
    return report
