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
    box_labels: Counter = field(default_factory=Counter)  # 상자만 있는 객체의 라벨


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
        r.box_labels.update(it.box_labels)
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


def _unit(path: Path, salt: str = "") -> float:
    """파일 이름 해시를 0~1로. salt가 다르면 서로 독립인 값."""
    return int(hashlib.sha256((salt + path.name).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def split_of(path: Path, val: float, test: float) -> str:
    """파일 이름 해시로 분할. 외장하드 드라이브 글자가 바뀌거나 폴더를 옮겨도 같은 사진은 같은 분할."""
    h = _unit(path)
    return "val" if h < val else "test" if h < val + test else "train"


def _read(path: Path) -> np.ndarray | None:
    import cv2

    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)  # 한글 경로


def build(items: Iterable[Item], mapping: dict[str, str | None], dest: Path, *, val: float = 0.1,
          test: float = 0.05, max_images: int = 0, keep_empty: float = 0.1, max_side: int = 0) -> BuildReport:
    """dest/images|labels/<분할>/*.jpg|txt. 매핑된 흠집이 없는 사진은 keep_empty 비율만 둔다(오탐 학습용)."""
    import cv2

    if max_side < 0:  # 음수면 축소 비율이 음수가 되어 사진이 1×1로 저장된다
        raise ValueError("max_side는 0 이상이어야 함(0은 줄이지 않음)")
    for sub in ("images", "labels"):  # 다시 빌드하면 옛 분할 사본이 남아 train·val이 섞이므로 지우지 않고 멈춘다
        if (dest / sub).exists() and any(f.is_file() for f in (dest / sub).rglob("*")):
            raise ValueError(f"{dest / sub}에 이전 빌드 결과가 있음: 다른 --out을 쓰거나 폴더를 직접 지운 뒤 다시 실행")
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
        if any(mapping.get(b) is not None for b in it.box_labels):
            # 쓰는 흠집이 상자로만 표시된 사진은 그 흠집을 폴리곤으로 못 넣으므로, 넣으면 흠집 자리를 배경으로 배운다
            report.skipped["상자로만 표시된 흠집 있음"] += 1
            continue
        kept = []
        for s in it.shapes:
            cls = mapping.get(s.label)
            if cls is None:
                report.dropped_labels[s.label] += 1
            else:
                kept.append((DEFECT_CLASSES.index(cls), s.points))
        if not kept and _unit(it.image, "empty") >= keep_empty:  # 빈 사진은 keep_empty 비율만(분할 해시와 독립)
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
        if max_side and max(h, w) > max_side:  # 라벨은 0~1 비율이라 사진만 줄이면 된다
            r = max_side / max(h, w)
            image = cv2.resize(image, (max(1, round(w * r)), max(1, round(h * r))), interpolation=cv2.INTER_AREA)
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
