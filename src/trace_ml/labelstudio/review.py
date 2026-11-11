"""검수용 모아보기: 제출된 폴리곤을 원본 사진 위에 그리고, 흠집마다 확대 조각을 붙인 JPG를 만든다.

민서 PC의 Label Studio·MinIO는 밖에서 열 수 없으므로, 결과 폴더(zip)를 검수하는 사람에게 넘긴다.
"""

from __future__ import annotations

import csv
import re
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .clean import DEVICE_RE, _device_of, _image_key
from .client import LabelStudio, LabelStudioError
from .config import NO_DEFECT, Settings
from .setup import find_project

# BGR. 라벨마다 눈에 띄는 색
COLORS: dict[str, tuple[int, int, int]] = {
    "scratch": (60, 20, 230), "dent": (0, 165, 255), "crack": (255, 80, 60), "stain": (200, 60, 200),
}
WIDTH = 1600   # 전체 사진 폭
CROP = 360     # 확대 조각 한 변(px)
PAD = 40       # 흠집 둘레 여유(원본 px)

Fetch = Callable[[str, str], bytes]


@dataclass
class ReviewResult:
    project_id: int
    out_dir: Path
    # 기기 → [폴리곤 있는 사진, no_defect, 아직 안 함]
    counts: dict[str, list[int]] = field(default_factory=dict)
    sheets: int = 0
    zip_path: Path | None = None


def _pick(annotations: list[dict[str, Any]]) -> dict[str, Any] | None:
    alive = [a for a in annotations if not a.get("was_cancelled")]
    if not alive:
        return None
    return max(alive, key=lambda a: (bool(a.get("ground_truth")), str(a.get("updated_at") or ""), a.get("id") or 0))


def _polygons(ann: dict[str, Any]) -> list[tuple[str, list[tuple[float, float]]]]:
    out = []
    for r in ann.get("result") or []:
        if r.get("type") != "polygonlabels":
            continue
        value = r.get("value") or {}
        labels = value.get("polygonlabels") or ["?"]
        out.append((str(labels[0]), [(float(x), float(y)) for x, y in value.get("points") or []]))
    return out


def _split_valid(polys: list[tuple[str, list[tuple[float, float]]]]) -> tuple[list, int]:
    """점이 3개 미만인 폴리곤은 그리지 않고 개수만 센다(상태표에 '잘못된 폴리곤'으로 남김)."""
    good = [p for p in polys if len(p[1]) >= 3]
    return good, len(polys) - len(good)


def _is_no_defect(ann: dict[str, Any]) -> bool:
    return any(r.get("type") == "choices" and NO_DEFECT in ((r.get("value") or {}).get("choices") or [])
               for r in ann.get("result") or [])


def render(image_bytes: bytes, polys: list[tuple[str, list[tuple[float, float]]]], title: str) -> bytes:
    """폴리곤 테두리를 그린 전체 사진 + 흠집마다 확대 조각(원본 화질)을 아래에 붙인다."""
    import cv2  # 무거워서 쓸 때만
    import numpy as np

    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise LabelStudioError(f"사진을 읽지 못함: {title}")
    h, w = img.shape[:2]
    clean = img.copy()
    crops = []
    for n, (label, pts) in enumerate(polys, 1):
        arr = np.array([[x / 100 * w, y / 100 * h] for x, y in pts], np.int32)
        color = COLORS.get(label, (255, 255, 255))
        cv2.polylines(img, [arr], True, color, 2)
        x0, y0 = arr.min(axis=0)
        cv2.putText(img, f"{n}:{label}", (int(x0), max(int(y0) - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        # 확대 조각: 흠집 둘레를 원본에서 잘라 CROP 크기로 키운다. 테두리는 얇게 그려 흠집이 가려지지 않게
        x1, y1 = arr.max(axis=0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(x1 - x0, y1 - y0) / 2 + PAD
        a, b = int(max(cx - half, 0)), int(max(cy - half, 0))
        c, d = int(min(cx + half, w)), int(min(cy + half, h))
        piece = clean[b:d, a:c].copy()
        cv2.polylines(piece, [arr - [a, b]], True, color, 1)
        piece = cv2.resize(piece, (CROP, CROP), interpolation=cv2.INTER_CUBIC)
        cv2.putText(piece, f"{n}:{label}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        crops.append(piece)
    scale = WIDTH / w
    full = cv2.resize(img, (WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.putText(full, title, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    rows = []
    per_row = WIDTH // CROP
    for i in range(0, len(crops), per_row):
        row = crops[i:i + per_row]
        row += [np.zeros((CROP, CROP, 3), np.uint8)] * (per_row - len(row))
        strip = np.hstack(row)
        rows.append(cv2.copyMakeBorder(strip, 0, 0, 0, WIDTH - strip.shape[1], cv2.BORDER_CONSTANT))
    sheet = np.vstack([full, *rows])
    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise LabelStudioError(f"JPG로 저장하지 못함: {title}")
    return buf.tobytes()


def review_sheets(ls: LabelStudio, settings: Settings, devices: list[str], out_dir: Path, fetch: Fetch,
                  *, make_zip: bool = True, include_clean: bool = False) -> ReviewResult:
    """기기마다 폴리곤이 있는 사진을 모아보기로 만들고, 상태 표(summary.csv)와 zip을 남긴다.

    include_clean이면 no_defect 사진도 그대로 넣어 빠뜨린 흠집이 없는지 함께 본다.
    """
    bad = [d for d in devices if not DEVICE_RE.fullmatch(d)]
    if bad:
        raise LabelStudioError(f"기기 ID 형식이 아님: {', '.join(bad)} (예: MB05)")
    project = find_project(ls, settings.project_title)
    if project is None:
        raise LabelStudioError(f"'{settings.project_title}' 프로젝트가 없음: 먼저 setup 실행")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = ReviewResult(project["id"], out_dir, {d: [0, 0, 0] for d in devices})
    rows: list[list[str]] = []
    written: list[Path] = []
    for task in ls.tasks(project["id"]):
        key = _image_key(task)
        device = _device_of(key)
        if device not in result.counts:
            continue
        ann = _pick(ls.annotations(int(task["id"])))
        name = key.rsplit("/", 1)[-1]
        if ann is None:
            result.counts[device][2] += 1
            rows.append([device, str(task["id"]), name, "안 함", ""])
            continue
        polys, broken = _split_valid(_polygons(ann))
        if broken:
            rows.append([device, str(task["id"]), name, f"잘못된 폴리곤 {broken}개(점 3개 미만)", ""])
        if not polys and broken:
            result.counts[device][0] += 1
            continue
        if not polys:
            clean = _is_no_defect(ann)
            result.counts[device][1 if clean else 2] += 1
            rows.append([device, str(task["id"]), name, NO_DEFECT if clean else "빈 제출", ""])
            if not (clean and include_clean):
                continue
        else:
            result.counts[device][0] += 1
        m = re.match(r"^s3://([^/]+)/(.+)$", key)
        if not m or m[1] != settings.raw_bucket:
            raise LabelStudioError(f"작업 {task['id']}: {settings.raw_bucket}의 사진 주소가 아님({key!r})")
        title = f"#{task['id']} {name}"
        sheet = render(fetch(m[1], m[2]), polys, title)
        (out_dir / device).mkdir(exist_ok=True)
        suffix = "" if polys else "__clean"
        path = out_dir / device / f"{Path(name).stem}__task{task['id']}{suffix}.jpg"
        path.write_bytes(sheet)
        written.append(path)
        result.sheets += 1
        if polys:
            rows.append([device, str(task["id"]), name, "폴리곤", " ".join(label for label, _ in polys)])
    rows.sort()
    summary = out_dir / "summary.csv"
    written.append(summary)
    with summary.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["기기", "작업", "사진", "상태", "라벨"])
        writer.writerows(rows)
    if make_zip:
        result.zip_path = out_dir.with_suffix(".zip")
        with zipfile.ZipFile(result.zip_path, "w", zipfile.ZIP_STORED) as z:
            # 이번 실행에서 만든 파일만 넣는다(이전 실행의 JPG가 섞이지 않게)
            for path in sorted(written):
                z.write(path, path.relative_to(out_dir.parent))
    return result


def minio_fetch(settings: Settings) -> Fetch:
    """Label Studio와 같은 키(trace-raw 읽기)로 원본 사진을 읽는다."""
    from ..settings import MinioSettings
    from ..store import make_store

    store = make_store(MinioSettings(endpoint=settings.minio_endpoint, access_key=settings.minio_access_key,
                                     secret_key=settings.minio_secret_key))
    return store.get
