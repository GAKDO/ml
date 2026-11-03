"""받은 데이터셋을 부스 추론(BO2)과 같은 640 타일로 자른다.

부스는 원본 해상도 사진을 640×640, 20% 겹침 타일로 잘라 엔진에 넣는다(booth `detect/tiling.py`).
학습도 같은 크기·같은 배율로 봐야 흠집 크기가 맞으므로, 사진을 같은 규칙으로 자르고 폴리곤을 타일 안으로 자른다.
"""

from __future__ import annotations

import hashlib
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

SPLITS = ("train", "val", "test")
Point = tuple[float, float]


@dataclass(frozen=True)
class Tile:
    x: int
    y: int
    w: int
    h: int


@dataclass
class TiledDataset:
    data_yaml: Path
    size: int
    overlap: float
    counts: dict[str, dict[str, int]] = field(default_factory=dict)


def _starts(length: int, size: int, step: int) -> list[int]:
    if length <= size:
        return [0]
    starts = list(range(0, length - size, step))
    starts.append(length - size)
    return sorted(set(starts))


def tiles(width: int, height: int, size: int = 640, overlap: float = 0.2) -> list[Tile]:
    """booth `detect/tiling.tiles`와 같은 규칙. 바꾸면 부스 쪽도 같이 바꾼다."""
    if size <= 0:
        raise ValueError("타일 크기는 양수")
    if not 0 <= overlap < 1:
        raise ValueError("overlap은 0 이상 1 미만")
    step = max(1, int(size * (1 - overlap)))
    return [Tile(x, y, min(size, width), min(size, height))
            for y in _starts(height, size, step) for x in _starts(width, size, step)]


def clip(points: list[Point], x0: float, y0: float, x1: float, y1: float) -> list[Point]:
    """Sutherland–Hodgman: 폴리곤을 사각형 안으로 자른다."""

    def cut(pts: list[Point], inside, cross) -> list[Point]:
        out: list[Point] = []
        for i, cur in enumerate(pts):
            prev = pts[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(cross(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cross(prev, cur))
        return out

    def at_x(xv: float):
        return lambda a, b: (xv, a[1] + (b[1] - a[1]) * (xv - a[0]) / (b[0] - a[0]))

    def at_y(yv: float):
        return lambda a, b: (a[0] + (b[0] - a[0]) * (yv - a[1]) / (b[1] - a[1]), yv)

    pts = list(points)
    for inside, cross in ((lambda p: p[0] >= x0, at_x(x0)), (lambda p: p[0] <= x1, at_x(x1)),
                          (lambda p: p[1] >= y0, at_y(y0)), (lambda p: p[1] <= y1, at_y(y1))):
        if not pts:
            break
        pts = cut(pts, inside, cross)
    return pts


def area(points: list[Point]) -> float:
    return abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1], strict=True))) / 2


def _read_labels(path: Path) -> list[tuple[int, list[Point]]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        nums = line.split()
        if len(nums) >= 7:
            vals = [float(v) for v in nums[1:]]
            out.append((int(nums[0]), list(zip(vals[0::2], vals[1::2], strict=True))))
    return out


def _read_image(path: Path) -> np.ndarray:
    import cv2

    # Windows에서 한글 경로도 읽도록 imread 대신 imdecode
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"사진을 읽지 못함: {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    import cv2

    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError(f"타일을 쓰지 못함: {path}")
    path.write_bytes(buf.tobytes())


def tile_label(polys: list[tuple[int, list[Point]]], width: int, height: int, t: Tile,
               min_area: float) -> list[str]:
    """정규화 폴리곤 → 타일 기준 정규화 폴리곤 줄. 타일 안에 남은 넓이가 min_area px² 미만이면 버린다."""
    lines = []
    for cls, pts in polys:
        px = [(x * width, y * height) for x, y in pts]
        cut = clip(px, t.x, t.y, t.x + t.w, t.y + t.h)
        if len(cut) < 3 or area(cut) < min_area:
            continue
        coords = " ".join(f"{(x - t.x) / t.w:.6f} {(y - t.y) / t.h:.6f}" for x, y in cut)
        lines.append(f"{cls} {coords}")
    return lines


def _keep(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def tile_dataset(src: Path, dest: Path, names: dict[int, str], *, size: int = 640, overlap: float = 0.2,
                 background_ratio: float = 1.0, min_area: float = 16.0) -> TiledDataset:
    """src(images|labels/<분할>) → dest 같은 구조의 타일 데이터셋.

    train은 흠집 타일을 모두 두고, 흠집 없는 타일은 흠집 타일 수 × background_ratio만큼만 고정 순서로 고른다
    (빈 타일이 대부분이라 그대로 두면 모델이 '없음'만 배운다). val·test는 오탐도 재도록 모든 타일을 둔다.
    """
    if size <= 0 or background_ratio < 0:
        raise ValueError("타일 크기는 양수, background_ratio는 0 이상")
    if dest.exists():  # 비율을 바꿔 다시 자를 때 옛 타일이 섞이지 않게 (받은 원본은 src에 그대로 있다)
        shutil.rmtree(dest)
    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        img_dir, lbl_dir = dest / "images" / split, dest / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        def write(name: str, crop: np.ndarray, lines: list[str], img_dir=img_dir, lbl_dir=lbl_dir) -> None:
            _write_image(img_dir / f"{name}.jpg", crop)
            (lbl_dir / f"{name}.txt").write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")

        # 사진 한 장씩 읽고 바로 쓴다. train 빈 타일은 이름·위치만 모았다가 고른 뒤 다시 읽는다(메모리에 사진을 쌓지 않음)
        n_defect, later, n_background = 0, [], 0
        for photo in sorted((src / "images" / split).glob("*.jpg")):
            label_path = src / "labels" / split / f"{photo.stem}.txt"
            polys = _read_labels(label_path) if label_path.exists() else []
            image = _read_image(photo)
            height, width = image.shape[:2]
            for t in tiles(width, height, size, overlap):
                name = f"{photo.stem}_x{t.x}_y{t.y}"
                lines = tile_label(polys, width, height, t, min_area)
                if lines:
                    write(name, image[t.y:t.y + t.h, t.x:t.x + t.w], lines)
                    n_defect += 1
                elif split == "train":
                    later.append((name, photo, t))
                else:
                    write(name, image[t.y:t.y + t.h, t.x:t.x + t.w], [])
                    n_background += 1
        if later:
            if n_defect:
                later = sorted(later, key=lambda b: _keep(b[0]))[:math.ceil(n_defect * background_ratio)]
            by_photo: dict[Path, list[tuple[str, Tile]]] = {}
            for name, photo, t in later:
                by_photo.setdefault(photo, []).append((name, t))
            for photo, picked in sorted(by_photo.items()):
                image = _read_image(photo)
                for name, t in picked:
                    write(name, image[t.y:t.y + t.h, t.x:t.x + t.w], [])
            n_background += len(later)
        counts[split] = {"tiles": n_defect + n_background, "defect": n_defect, "background": n_background}
    data = {"path": str(dest.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
            "names": names}
    data_yaml = dest / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return TiledDataset(data_yaml, size, overlap, counts)
