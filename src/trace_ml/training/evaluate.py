"""기기 판정 평가: 타일 검출을 사진·기기 면으로 모아 '문제 있음/없음'이 라벨과 맞는지 센다.

mAP는 폴리곤 모양·클래스까지 따지므로 흠집이 적은 부스 데이터에서 크게 흔들린다. 부스가 실제로 내는 판정은
"이 면에 흠집이 있나"이므로, 같은 단위(사진, 기기 면)로 맞힌 수를 따로 낸다. 경진대회 지표로도 쓴다.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

# MB13_TOP_L1_CAM1_01_x1024_y440 → 사진 MB13_TOP_L1_CAM1_01, 기기 면 MB13_TOP
_TILE = re.compile(r"^(?P<photo>(?P<device>[A-Z]{2,4}[0-9]{2})_(?P<face>[A-Z0-9]+)_[A-Z0-9]+_CAM[0-9]+_[0-9]+)"
                   r"(?:_x[0-9]+_y[0-9]+)?$")

Detection = tuple[str, float]  # (클래스 이름, 신뢰도)
Predict = Callable[[Path, list[Path], float], dict[str, list[Detection]]]


class EvaluateError(Exception):
    """평가 입력이 맞지 않음."""


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, truth: bool, pred: bool) -> None:
        if truth and pred:
            self.tp += 1
        elif pred:
            self.fp += 1
        elif truth:
            self.fn += 1
        else:
            self.tn += 1

    def summary(self) -> dict[str, Any]:
        total = self.tp + self.fp + self.fn + self.tn
        return {**asdict(self),
                "precision": round(self.tp / (self.tp + self.fp), 3) if self.tp + self.fp else None,
                "recall": round(self.tp / (self.tp + self.fn), 3) if self.tp + self.fn else None,
                "accuracy": round((self.tp + self.tn) / total, 3) if total else None}


@dataclass
class FaceResult:
    face: str
    truth: bool
    pred: bool
    photos: int
    labeled_photos: int
    detected_photos: int
    classes: dict[str, int] = field(default_factory=dict)


@dataclass
class Report:
    split: str
    conf: float
    min_photos: int
    photo: dict[str, Any]
    face: dict[str, Any]
    faces: list[FaceResult]
    sweep: list[dict[str, Any]]
    # 점수와 모델을 잇는 값. 가중치 경로·해시는 항상, model_version은 --model-version을 줄 때만
    model_version: str | None = None
    weights: str = ""
    weights_sha256: str = ""
    # 평가 데이터 지문(사진 이름 + 라벨 내용). 게이트는 두 평가의 지문이 같을 때만 점수를 비교한다
    data_sha256: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def parse_tile(stem: str) -> tuple[str, str]:
    """타일(또는 사진) 이름 → (사진, 기기 면). 규칙에 안 맞으면 EvaluateError."""
    m = _TILE.match(stem)
    if not m:
        raise EvaluateError(f"파일 이름이 부스 규칙과 다름: {stem}")
    return m["photo"], f"{m['device']}_{m['face']}"


def read_truth(labels_dir: Path) -> dict[str, list[int]]:
    """YOLO 라벨 txt → 타일 이름별 클래스 번호 목록(빈 목록 = 흠집 없음)."""
    if not labels_dir.is_dir():
        raise EvaluateError(f"라벨 폴더가 없음: {labels_dir}")
    out: dict[str, list[int]] = {}
    for p in sorted(labels_dir.glob("*.txt")):
        rows = [line.split() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        out[p.stem] = [int(r[0]) for r in rows]
    if not out:
        raise EvaluateError(f"라벨이 없음: {labels_dir}")
    return out


def judge(truth: dict[str, list[int]], preds: dict[str, list[Detection]], *, conf: float,
          min_photos: int = 1) -> tuple[Confusion, Confusion, list[FaceResult]]:
    """사진: 타일 하나라도 흠집이면 흠집. 기기 면: 흠집 사진이 min_photos장 이상이면 '문제 있음'.

    min_photos를 2 이상으로 두면 한 장에만 잡힌 먼지·반사를 판정에서 거른다(회차를 바꿔도 남는 것만 손상).
    """
    photo_truth: dict[str, bool] = {}
    photo_pred: dict[str, bool] = {}
    photo_face: dict[str, str] = {}
    face_classes: dict[str, dict[str, int]] = {}
    for tile, classes in truth.items():
        photo, face = parse_tile(tile)
        photo_face[photo] = face
        photo_truth[photo] = photo_truth.get(photo, False) or bool(classes)
        hits = [name for name, score in preds.get(tile, []) if score >= conf]
        photo_pred[photo] = photo_pred.get(photo, False) or bool(hits)
        counts = face_classes.setdefault(face, {})
        for name in hits:
            counts[name] = counts.get(name, 0) + 1
    unknown = sorted(set(preds) - set(truth))
    if unknown:
        raise EvaluateError(f"라벨 없는 타일의 검출 {len(unknown)}개(예: {unknown[0]}): 같은 split 폴더인지 확인")

    photos = Confusion()
    for photo, t in photo_truth.items():
        photos.add(t, photo_pred[photo])

    faces: dict[str, FaceResult] = {}
    for photo, face in photo_face.items():
        r = faces.setdefault(face, FaceResult(face, False, False, 0, 0, 0, face_classes.get(face, {})))
        r.photos += 1
        r.labeled_photos += photo_truth[photo]
        r.detected_photos += photo_pred[photo]
    face_conf = Confusion()
    for r in faces.values():
        r.truth = r.labeled_photos > 0
        r.pred = r.detected_photos >= min_photos
        face_conf.add(r.truth, r.pred)
    return photos, face_conf, sorted(faces.values(), key=lambda r: r.face)


def evaluate(truth: dict[str, list[int]], preds: dict[str, list[Detection]], *, split: str, conf: float,
             min_photos: int = 1, sweep: Iterable[float] = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)) -> Report:
    """preds는 가장 낮은 sweep 값 이하 신뢰도까지 모은 검출이어야 한다(문턱별로 다시 거른다)."""
    photos, faces, rows = judge(truth, preds, conf=conf, min_photos=min_photos)
    swept = []
    for c in sorted(set(sweep) | {conf}):
        p, f, _ = judge(truth, preds, conf=c, min_photos=min_photos)
        swept.append({"conf": c, "photo": p.summary(), "face": f.summary()})
    return Report(split, conf, min_photos, photos.summary(), faces.summary(), rows, swept)


def predict_yolo(weights: Path, images: list[Path], conf: float) -> dict[str, list[Detection]]:  # pragma: no cover
    """ultralytics로 타일마다 검출. GPU가 있는 집 PC에서만 돈다."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('ultralytics가 없음: python -m pip install -e ".[train]"') from exc
    model = YOLO(str(weights))
    out: dict[str, list[Detection]] = {}
    for start in range(0, len(images), 32):
        batch = images[start:start + 32]
        for path, result in zip(batch, model.predict([str(p) for p in batch], conf=conf, imgsz=640,
                                                     verbose=False), strict=True):
            names = result.names
            boxes = result.boxes
            out[path.stem] = [] if boxes is None else [
                (names[int(c)], float(s)) for c, s in zip(boxes.cls.tolist(), boxes.conf.tolist(), strict=True)]
    return out


def run_evaluate(weights: Path, data_dir: Path, *, split: str = "val", conf: float = 0.25, min_photos: int = 1,
                 predict: Predict = predict_yolo, model_version: str | None = None) -> Report:
    """data_dir(타일 데이터셋)의 split을 가중치로 검출해 판정을 센다. 결과에 가중치 경로·SHA-256을 남긴다."""
    images_dir, labels_dir = data_dir / "images" / split, data_dir / "labels" / split
    if not weights.is_file():
        raise EvaluateError(f"가중치가 없음: {weights}")
    truth = read_truth(labels_dir)
    # 라벨 없는 사진을 조용히 빼면 판정 수가 줄어 지표가 틀어지므로, 이름이 하나라도 어긋나면 멈춘다
    images = sorted(images_dir.glob("*.jpg"))
    stems = {p.stem for p in images}
    if stems != set(truth):
        no_label, no_image = sorted(stems - set(truth)), sorted(set(truth) - stems)
        example = (no_label or no_image)[0]
        raise EvaluateError(f"사진과 라벨 이름이 다름(라벨 없는 사진 {len(no_label)}장 · 사진 없는 라벨 "
                            f"{len(no_image)}개, 예: {example}): {images_dir}")
    preds = predict(weights, images, min(0.1, conf))
    report = evaluate(truth, preds, split=split, conf=conf, min_photos=min_photos)
    return replace(report, model_version=model_version, weights=str(weights),
                   weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                   data_sha256=data_fingerprint(labels_dir, images))


def data_fingerprint(labels_dir: Path, images: list[Path]) -> str:
    """같은 val인지 가리는 값. 사진 이름과 라벨 파일 내용을 정렬해 해시한다(사진 바이트는 크고 느려서 뺀다)."""
    h = hashlib.sha256()
    for img in sorted(images, key=lambda p: p.name):
        h.update(img.name.encode() + b"\0")
        h.update((labels_dir / f"{img.stem}.txt").read_bytes() + b"\0")
    return h.hexdigest()
