"""AI Hub 라벨 JSON 읽기. 데이터마다 키 이름이 달라서 모양으로 찾는다.

- COCO형: 한 파일에 images·annotations·categories (segmentation 폴리곤)
- 이미지별형: 사진 하나에 JSON 하나. 안쪽 어딘가에 '라벨 이름 + 점 목록'이 있는 객체들(labelme shapes 포함)

라벨 이름은 그대로 모아 `scan`이 보여 주고, 우리 클래스(codes.yaml)로 옮기는 표는 사람이 정한다(map.yaml).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
LABEL_KEYS = ("label", "class_name", "category_name", "class", "category", "defect_type", "defect", "type", "name")
POINT_KEYS = ("points", "polygon", "segmentation", "coordinates", "xy", "poly")
FILE_KEYS = ("file_name", "filename", "imagePath", "image_path", "image_name", "img_name", "name", "path")
Point = tuple[float, float]


@dataclass
class Shape:
    label: str
    points: list[Point]  # 사진 px


@dataclass
class Item:
    """사진 한 장과 그 폴리곤들. width·height를 JSON에서 못 찾으면 0(사진을 읽어 채운다)."""

    image: Path | None
    shapes: list[Shape] = field(default_factory=list)
    width: int = 0
    height: int = 0
    source: Path | None = None
    box_labels: list[str] = field(default_factory=list)  # 폴리곤 없이 상자만 있던 객체의 라벨

    @property
    def boxes_only(self) -> int:
        return len(self.box_labels)


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_points(raw: Any) -> list[list[Point]]:
    """여러 모양의 점 목록 → 폴리곤들. [[x,y],..] / [x1,y1,..] / [{"x":..,"y":..}] / COCO [[x1,y1,..],..]."""
    if isinstance(raw, dict):
        for key in ("points", "polygon", "coordinates", "exterior"):
            if key in raw:
                return parse_points(raw[key])
        xs, ys = raw.get("x"), raw.get("y")
        if isinstance(xs, list) and isinstance(ys, list) and len(xs) == len(ys):
            pts = [(_num(x), _num(y)) for x, y in zip(xs, ys, strict=True)]
            return [[(x, y) for x, y in pts if x is not None and y is not None]]
        return []
    if not isinstance(raw, list) or not raw:
        return []
    first = raw[0]
    if isinstance(first, dict) and "x" in first and "y" in first:
        pts = [(_num(p.get("x")), _num(p.get("y"))) for p in raw if isinstance(p, dict)]
        return [[(x, y) for x, y in pts if x is not None and y is not None]]
    if isinstance(first, (list, tuple)) and len(first) == 2 and all(_num(v) is not None for v in first):
        return [[(float(p[0]), float(p[1])) for p in raw if isinstance(p, (list, tuple)) and len(p) == 2]]
    if isinstance(first, (list, tuple, dict)):  # 폴리곤 여러 개(COCO) 또는 한 겹 더 감싼 것
        out: list[list[Point]] = []
        for part in raw:
            out += parse_points(part)
        return out
    nums = [_num(v) for v in raw]
    if all(v is not None for v in nums) and len(nums) >= 6 and len(nums) % 2 == 0:
        return [list(zip(nums[0::2], nums[1::2], strict=True))]  # type: ignore[arg-type]
    return []


def _label(obj: dict) -> str | None:
    for key in LABEL_KEYS:
        v = obj.get(key)
        if isinstance(v, (str, int)) and str(v).strip():
            return str(v).strip()
        if isinstance(v, dict):  # {"class": {"name": "긁힘"}} 꼴
            inner = _label(v)
            if inner:
                return inner
    return None


def _walk(node: Any) -> Iterator[dict]:
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def _image_name(doc: Any) -> str | None:
    for obj in _walk(doc):
        for key in FILE_KEYS:
            v = obj.get(key)
            if isinstance(v, str) and v.lower().endswith(IMAGE_EXTS):
                return v.replace("\\", "/").rsplit("/", 1)[-1]
    return None


def _size(doc: Any) -> tuple[int, int]:
    for obj in _walk(doc):
        w = obj.get("width", obj.get("imageWidth", obj.get("img_width")))
        h = obj.get("height", obj.get("imageHeight", obj.get("img_height")))
        if _num(w) and _num(h):
            return int(float(w)), int(float(h))
    return 0, 0


def shapes_in(doc: Any) -> tuple[list[Shape], list[str]]:
    """이미지별 JSON 안의 '라벨 + 점' 객체들과, 상자만 있는 객체의 라벨."""
    shapes: list[Shape] = []
    boxes: list[str] = []
    for obj in _walk(doc):
        label = _label(obj)
        if not label:
            continue
        key = next((k for k in POINT_KEYS if k in obj), None)
        if key is None:
            if any(k in obj for k in ("bbox", "box", "bounding_box", "rect")):
                boxes.append(label)
            continue
        if obj.get("shape_type") in ("rectangle", "point", "line") and key == "points":
            boxes.append(label)
            continue
        for poly in parse_points(obj[key]):
            if len(poly) >= 3:
                shapes.append(Shape(label, poly))
    return shapes, boxes


def is_coco(doc: Any) -> bool:
    return isinstance(doc, dict) and isinstance(doc.get("images"), list) and isinstance(doc.get("annotations"), list)


def read_coco(path: Path, doc: dict, index: dict[str, Path]) -> list[Item]:
    cats = {c.get("id"): str(c.get("name", c.get("id"))) for c in doc.get("categories", []) if isinstance(c, dict)}
    items: dict[Any, Item] = {}
    for im in doc["images"]:
        name = str(im.get("file_name", "")).replace("\\", "/").rsplit("/", 1)[-1]
        items[im.get("id")] = Item(index.get(name.lower()), [], int(im.get("width") or 0), int(im.get("height") or 0), path)
    for ann in doc["annotations"]:
        item = items.get(ann.get("image_id"))
        if item is None:
            continue
        label = cats.get(ann.get("category_id")) or _label(ann) or "?"
        polys = parse_points(ann.get("segmentation"))
        if not polys:
            item.box_labels.append(label)
        for poly in polys:
            if len(poly) >= 3:
                item.shapes.append(Shape(label, poly))
    return list(items.values())


def index_images(roots: list[Path]) -> dict[str, Path]:
    """파일 이름(소문자) → 경로. AI Hub는 원천(사진)·라벨링(JSON) 폴더가 나뉘어 있어 이름으로 짝짓는다."""
    out: dict[str, Path] = {}
    for root in roots:
        for p in root.rglob("*"):
            if p.suffix.lower() in IMAGE_EXTS:
                out.setdefault(p.name.lower(), p)  # AI Hub는 JSON의 .JPG와 실제 .jpg가 섞여 있어 대소문자 무시
    return out


def read_items(roots: list[Path], index: dict[str, Path] | None = None) -> Iterator[Item]:
    """roots 아래 모든 JSON을 읽어 사진별 Item. 읽지 못한 JSON은 건너뛴다(scan이 센다)."""
    index = index if index is not None else index_images(roots)
    for root in roots:
        for path in sorted(root.rglob("*.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                yield Item(None, source=path)
                continue
            if is_coco(doc):
                try:  # 형식이 깨진 COCO 하나 때문에 전체가 멈추지 않게
                    coco_items = list(read_coco(path, doc, index))
                except (AttributeError, TypeError, ValueError, KeyError):
                    yield Item(None, source=path)
                    continue
                yield from coco_items
                continue
            name = _image_name(doc)
            image = index.get(name.lower()) if name else None
            if image is None:  # JSON과 같은 이름의 사진
                image = next((index[(path.stem + e).lower()] for e in IMAGE_EXTS if (path.stem + e).lower() in index), None)
            shapes, boxes = shapes_in(doc)
            w, h = _size(doc)
            yield Item(image, shapes, w, h, path, boxes)
