"""사람 검수 결과를 다음 학습 데이터로 내보낸다(11/8 민서 결정: AI가 넓게 잡고 사람이 확정).

- 흠집 맞음(CONFIRMED)은 YOLO 분할 라벨이 된다.
- 흠집 아님(DISMISSED)으로 지운 자리는 라벨을 만들지 않아 배경, 즉 음성 예시가 된다.
  후보가 모두 흠집 아님이었던 사진은 빈 라벨 파일(흠집 없는 사진)로 남는다.
- AI가 아예 못 잡은 흠집은 이 사진들에 라벨이 없으므로, 내보낸 사진은 학습 train에만 더한다(val로 재지 않음).

capture-server의 부스 빠른 검사(QUICK) 세트만 쓴다. 이 세트의 폴리곤은 원본 사진 px라 사진 크기로 0~1로 바꾸면 된다.
정합 세트(FULL)는 표준 좌표 px라 사진 좌표로 되돌려야 해서 지금은 내보내지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from ..codes import DEFECT_CLASSES

Fetch = Callable[[str, str], bytes]

# 내보낸 사진 이름 앞머리. 데이터셋 버전 사진(기기 코드로 시작)과 섞여도 겹치지 않는다
PREFIX = "rv_"
COLUMNS = 17  # QUERY 열 수(CSV 확인용)
_SAFE = re.compile(r"[^A-Za-z0-9_-]")

QUERY = """
select p.public_id, cs.public_id, cs.rental_id, p.face, p.light, p.camera, p.object_key,
       to_json(cs.review_completed_at) #>> '{{}}', d.public_id, d.class, d.photo_polygon::text, d.confidence, d.model_version,
       d.review_status, d.dismiss_reason, dv.serial, dv.asset_no
  from capture.capture_set cs
  join capture.photo p on p.capture_set_id = cs.id
  join inspection.defect d on d.photo_id = p.id
  left join rental.rental r on r.public_id = cs.rental_id
  left join rental.device dv on dv.id = r.device_id
 where cs.profile = 'QUICK' and cs.review_completed_at is not null and p.uploaded
   and d.review_status in ('CONFIRMED', 'DISMISSED')
   and ({since} is null or cs.review_completed_at >= {since})
 order by cs.review_completed_at, p.public_id, d.public_id
"""


def query(since: datetime | None) -> str:
    """since는 datetime만 받아 따옴표 문자열로 넣는다(psql \\copy에는 바인드 변수를 쓸 수 없음)."""
    literal = f"'{since.isoformat()}'::timestamptz" if since else "null::timestamptz"
    return QUERY.format(since=literal).strip()


def psql_command(since: datetime | None) -> str:
    """학교 서버에서 돌릴 명령. DB 포트를 밖에 열지 않아 서버 안에서 CSV로 뽑아 집 PC로 가져온다."""
    one_line = " ".join(query(since).split())
    return ("docker compose exec -T postgres psql -U postgres -d trace -v ON_ERROR_STOP=1 "
            f'-c "\\copy ({one_line}) to stdout with csv header" > reviews.csv')


class ReviewExportError(Exception):
    pass


@dataclass(frozen=True)
class ReviewedDefect:
    defect_id: str
    cls: str
    polygon: list[tuple[float, float]]
    confidence: float
    model_version: str
    decision: str            # CONFIRMED | DISMISSED
    reason: str | None       # DISMISSED일 때만(DUST·TAPE·REFLECTION·PART·OTHER)


@dataclass
class ReviewedPhoto:
    photo_id: str
    set_id: str
    rental_id: str
    face: str
    light: str
    camera: str
    object_key: str
    reviewed_at: datetime
    defects: list[ReviewedDefect] = field(default_factory=list)
    # 기기 식별(rental.device). 학습 때 데이터셋 VAL·TEST 기기 코드와 맞춰 그 기기 사진을 train에서 뺀다
    serial: str | None = None
    asset_no: str | None = None

    def device_ids(self) -> set[str]:
        return {v.strip().upper() for v in (self.serial, self.asset_no) if v and v.strip()}

    @property
    def name(self) -> str:
        raw = f"{PREFIX}{self.set_id}_{self.face}_{self.light}_{self.camera}"
        return _SAFE.sub("-", raw)


class ReviewSource(Protocol):
    def photos(self, since: datetime | None) -> list[ReviewedPhoto]: ...


def group_rows(rows: Iterable[tuple[Any, ...]]) -> list[ReviewedPhoto]:
    """QUERY 결과(사진 × 흠집 한 줄씩)를 사진별로 묶는다. 순서는 검수 완료 시각·사진 ID 순."""
    photos: dict[str, ReviewedPhoto] = {}
    for (photo_id, set_id, rental_id, face, light, camera, key, reviewed_at,
         defect_id, cls, polygon, conf, model_version, decision, reason, serial, asset_no) in rows:
        photo = photos.get(photo_id)
        if photo is None:
            when = reviewed_at if isinstance(reviewed_at, datetime) else datetime.fromisoformat(str(reviewed_at))
            photo = photos[photo_id] = ReviewedPhoto(photo_id, set_id, rental_id, face, light, camera, key, when,
                                                     serial=serial or None, asset_no=asset_no or None)
        points = json.loads(polygon) if isinstance(polygon, str) else polygon
        photo.defects.append(ReviewedDefect(defect_id, cls, [(float(x), float(y)) for x, y in points or []],
                                            float(conf), model_version, decision, reason or None))
    return list(photos.values())


class PgReviewSource:
    """capture-server DB(학교 서버 Compose의 PostgreSQL). 읽기만 한다."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def photos(self, since: datetime | None) -> list[ReviewedPhoto]:
        import psycopg

        try:
            with psycopg.connect(self._dsn, connect_timeout=10) as conn, conn.cursor() as cur:
                conn.read_only = True
                cur.execute(query(since))
                return group_rows(cur.fetchall())
        except psycopg.Error as exc:
            # DSN에 비밀번호가 있을 수 있어 예외 문장 대신 종류만 보인다
            raise ReviewExportError(f"capture DB 읽기 실패: {type(exc).__name__}") from exc


class CsvReviewSource:
    """학교 서버에서 psql_command로 뽑은 CSV(머리줄 포함). 열 순서는 QUERY와 같다."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def photos(self, since: datetime | None) -> list[ReviewedPhoto]:
        import csv

        if not self._path.is_file():
            raise ReviewExportError(f"CSV가 없음: {self._path}")
        with self._path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))[1:]
        bad = [i + 2 for i, r in enumerate(rows) if len(r) != COLUMNS]
        if bad:
            raise ReviewExportError(f"열이 {COLUMNS}개가 아닌 줄: {bad[:5]} (review sql 명령으로 다시 뽑으세요)")
        photos = group_rows(rows)
        return [p for p in photos if since is None or p.reviewed_at >= since]


@dataclass(frozen=True)
class ExportResult:
    out_dir: Path
    photos: int
    positives: int
    negatives: int
    confirmed: dict[str, int]
    dismissed: dict[str, int]
    skipped: list[str]
    summary_sha256: str


def yolo_lines(defects: list[ReviewedDefect], width: int, height: int, decision: str = "CONFIRMED",
               classes: tuple[str, ...] = DEFECT_CLASSES) -> tuple[list[str], list[str]]:
    """decision(기본 흠집 맞음)인 것만 'cls x1 y1 x2 y2 ...'(0~1)로. 점 3개 미만·모르는 클래스는 빼고 사유를 돌려준다."""
    lines, problems = [], []
    for d in defects:
        if d.decision != decision:
            continue
        if d.cls not in classes:
            problems.append(f"{d.defect_id}: 모르는 클래스 {d.cls}")
            continue
        if len(d.polygon) < 3:
            problems.append(f"{d.defect_id}: 점이 3개 미만")
            continue
        coords = " ".join(f"{min(max(x / width, 0.0), 1.0):.6f} {min(max(y / height, 0.0), 1.0):.6f}"
                          for x, y in d.polygon)
        lines.append(f"{classes.index(d.cls)} {coords}")
    return lines, problems


def _size(body: bytes) -> tuple[int, int] | None:
    import cv2  # 무거워서 쓸 때만
    import numpy as np

    # 타일 단계(tiles._read_image)와 같은 플래그로 읽어야 크기가 같다. 부스 사진은 OpenCV로 만들어 EXIF 방향이 없고
    # 폴리곤도 그 사진의 px라, 같은 플래그면 둘이 어긋나지 않는다
    img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img.shape[1], img.shape[0]


def export(source: ReviewSource, fetch: Fetch, bucket: str, out_dir: Path, *, since: datetime | None = None,
           now: Callable[[], datetime] = lambda: datetime.now(UTC)) -> ExportResult:
    """out_dir/images/train·labels/train(YOLO 분할 구조) + reviews.jsonl + summary.json."""
    if out_dir.exists() and any(out_dir.iterdir()):
        # 지난 내보내기와 섞이면 어느 검수에서 왔는지 알 수 없어 빈 폴더에만 쓴다
        raise ReviewExportError(f"폴더가 비어 있지 않음(덮어쓰지 않음): {out_dir}")
    photos = source.photos(since)
    if not photos:
        raise ReviewExportError("내보낼 검수 결과가 없음(검수 완료된 부스 빠른 검사 세트가 없음)")
    # hard/: 흠집 아님으로 지운 자리(라벨 아님). 타일로 자를 때 이 자리가 걸친 타일을 음성 예시로 꼭 남긴다
    try:
        return _export(photos, fetch, bucket, out_dir, since, now)
    except BaseException:
        # 사진 받기가 중간에 끊기면(SSH 터널 등) 이번에 만든 것만 지워 같은 --out으로 다시 돌릴 수 있게 한다.
        # 시작할 때 폴더가 비어 있었으므로 이 목록 밖의 파일은 없다
        for d in ("images", "labels", "hard"):
            shutil.rmtree(out_dir / d, ignore_errors=True)
        for name in ("reviews.jsonl", "summary.json"):
            (out_dir / name).unlink(missing_ok=True)
        raise


def _export(photos: list[ReviewedPhoto], fetch: Fetch, bucket: str, out_dir: Path, since: datetime | None,
            now: Callable[[], datetime]) -> ExportResult:
    img_dir, lbl_dir, hard_dir = (out_dir / d / "train" for d in ("images", "labels", "hard"))
    for d in (img_dir, lbl_dir, hard_dir):
        d.mkdir(parents=True, exist_ok=True)
    confirmed: Counter[str] = Counter()
    dismissed: Counter[str] = Counter()
    skipped: list[str] = []
    positives = negatives = 0
    records: list[str] = []
    for photo in photos:
        body = fetch(bucket, photo.object_key)
        size = _size(body)
        if size is None:
            skipped.append(f"{photo.photo_id}: 사진을 읽지 못함({photo.object_key})")
            continue
        lines, problems = yolo_lines(photo.defects, *size)
        if problems:
            # 흠집 맞음인데 라벨로 못 바꾼 것이 있으면 그 자리가 배경으로 학습돼 진짜 흠집을 놓치게 가르치므로 사진째 뺀다
            skipped.extend(problems)
            skipped.append(f"{photo.photo_id}: 흠집 맞음을 라벨로 못 바꿔 사진을 뺌")
            continue
        hard, hard_problems = yolo_lines(photo.defects, *size, decision="DISMISSED")
        skipped.extend(hard_problems)
        if hard:
            (hard_dir / f"{photo.name}.txt").write_text("".join(f"{line}\n" for line in hard), encoding="utf-8")
        (img_dir / f"{photo.name}.jpg").write_bytes(body)
        (lbl_dir / f"{photo.name}.txt").write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
        positives += bool(lines)
        negatives += not lines
        for d in photo.defects:
            if d.decision == "CONFIRMED":
                confirmed[d.cls] += 1
            else:
                dismissed[d.reason or "OTHER"] += 1
        records.append(json.dumps({
            "name": photo.name, "photoId": photo.photo_id, "setId": photo.set_id, "rentalId": photo.rental_id,
            "face": photo.face, "light": photo.light, "camera": photo.camera,
            "device": {"serial": photo.serial, "assetNo": photo.asset_no},
            "image": {"bucket": bucket, "key": photo.object_key, "sha256": hashlib.sha256(body).hexdigest(),
                      "width": size[0], "height": size[1]},
            "reviewedAt": photo.reviewed_at.isoformat(),
            "defects": [{"defectId": d.defect_id, "class": d.cls, "decision": d.decision, "reason": d.reason,
                         "confidence": d.confidence, "modelVersion": d.model_version} for d in photo.defects],
        }, ensure_ascii=False))
    if not records:
        raise ReviewExportError("내보낸 사진이 0장: " + ("; ".join(skipped[:5]) or "모든 사진을 건너뜀"))
    (out_dir / "reviews.jsonl").write_text("".join(f"{r}\n" for r in records), encoding="utf-8")
    summary = {
        "schema": "trace.review-export.v1", "exportedAt": now().replace(microsecond=0).isoformat(),
        "since": since.isoformat() if since else None, "bucket": bucket,
        "photos": positives + negatives, "positives": positives, "negatives": negatives,
        "confirmed": dict(sorted(confirmed.items())), "dismissed": dict(sorted(dismissed.items())),
        "skipped": skipped, "classes": list(DEFECT_CLASSES),
        "reviewsSha256": hashlib.sha256((out_dir / "reviews.jsonl").read_bytes()).hexdigest(),
    }
    raw = json.dumps(summary, ensure_ascii=False, indent=2).encode()
    (out_dir / "summary.json").write_bytes(raw)
    return ExportResult(out_dir, positives + negatives, positives, negatives, dict(confirmed), dict(dismissed),
                        skipped, hashlib.sha256(raw).hexdigest())


def read_summary(folder: Path) -> tuple[dict[str, Any], str]:
    """train --extra-train이 쓰는 폴더 확인. summary.json과 그 해시를 돌려준다."""
    path = folder / "summary.json"
    if not path.is_file() or not (folder / "images" / "train").is_dir():
        raise ReviewExportError(f"검수 내보내기 폴더가 아님(summary.json·images/train 없음): {folder}")
    try:
        raw = path.read_bytes()
        summary = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise ReviewExportError(f"summary.json을 읽지 못함: {folder} ({exc})") from exc
    schema = summary.get("schema") if isinstance(summary, dict) else None
    if schema != "trace.review-export.v1":
        raise ReviewExportError(f"모르는 내보내기 형식: {schema}")
    return summary, hashlib.sha256(raw).hexdigest()


def held_out_names(folder: Path, held_out: set[str]) -> list[str]:
    """reviews.jsonl에서 기기 시리얼·자산번호가 held_out(데이터셋 VAL·TEST 기기 코드, 대문자)에 든 사진 이름."""
    path = folder / "reviews.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
    except (OSError, ValueError) as exc:
        raise ReviewExportError(f"reviews.jsonl을 읽지 못함: {folder} ({exc})") from exc
    names = []
    for row in rows:
        device = row.get("device") or {}
        ids = {str(v).strip().upper() for v in (device.get("serial"), device.get("assetNo")) if v}
        if ids & held_out:
            names.append(row["name"])
    return sorted(names)
