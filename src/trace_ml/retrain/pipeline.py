"""재학습 파이프라인의 판단 부분(11/9 강사님 요청).

사람이 검수한 데이터가 기준만큼 쌓였는지 세고(status), 새 모델이 지금 모델보다 나을 때만 통과시키고(gate),
통과한 모델만 레지스트리에 '현재 모델'로 올린다(promote). 학습 자체는 집 PC의 trace-ml train이 한다.
모델 일치 원칙: 학교 서버와 부스 Jetson은 늘 레지스트리의 현재 모델 하나를 같은 파일로 쓴다.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..dataset.review_export import ReviewedPhoto

REGISTRY_SCHEMA = "trace.model-registry.v1"
GATE_SCHEMA = "trace.model-gate.v1"
_VERSION = re.compile(r"^[A-Za-z0-9._+-]{1,64}$")
Now = Callable[[], datetime]


class RetrainError(Exception):
    pass


def _now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------- 1. 누적 확인 ----------

@dataclass(frozen=True)
class Status:
    since: str | None
    photos: int
    sets: int
    confirmed: int
    dismissed: int
    min_photos: int
    min_confirmed: int

    @property
    def ready(self) -> bool:
        # 확정 흠집이 너무 적으면 음성 예시만 늘어 재현율이 떨어질 수 있어 두 기준을 모두 본다
        return self.photos >= self.min_photos and self.confirmed >= self.min_confirmed


def count(photos: list[ReviewedPhoto], since: datetime | None, *, min_photos: int, min_confirmed: int) -> Status:
    """since(지난 재학습에 쓴 마지막 검수 시각) 뒤에 검수가 끝난 사진·흠집 수."""
    new = [p for p in photos if since is None or p.reviewed_at > since]
    defects = [d for p in new for d in p.defects]
    return Status(since.isoformat() if since else None, len(new), len({p.set_id for p in new}),
                  sum(d.decision == "CONFIRMED" for d in defects), sum(d.decision == "DISMISSED" for d in defects),
                  min_photos, min_confirmed)


def reviews_until(export_dir: Path) -> datetime | None:
    """review export 폴더(reviews.jsonl)에서 가장 늦은 검수 시각. 다음 status는 이 시각 뒤만 센다."""
    path = export_dir / "reviews.jsonl"
    if not path.is_file():
        raise RetrainError(f"검수 내보내기 폴더가 아님(reviews.jsonl 없음): {export_dir}")
    times = [datetime.fromisoformat(json.loads(line)["reviewedAt"])
             for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return max(times) if times else None


# ---------- 2. 게이트 ----------

def _load(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RetrainError(f"평가 결과를 읽지 못함: {path} ({exc})") from exc
    if not isinstance(doc, dict) or not {"photo", "face", "faces", "split"} <= set(doc):
        raise RetrainError(f"trace-ml evaluate 결과가 아님: {path}")
    return doc


def _same_data(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """두 평가가 같은 데이터·같은 조건인지. 다르면 점수를 비교하는 의미가 없다."""
    problems = [f"{k}가 다름({a.get(k)} vs {b.get(k)})" for k in ("split", "conf", "min_photos") if a.get(k) != b.get(k)]
    if not a.get("data_sha256") or a.get("data_sha256") != b.get("data_sha256"):
        problems.append("평가 데이터 지문(data_sha256)이 없거나 다름: 두 모델을 같은 --data로 다시 재세요")
    faces_a = {f["face"]: f["photos"] for f in a["faces"]}
    faces_b = {f["face"]: f["photos"] for f in b["faces"]}
    if faces_a != faces_b:
        problems.append("평가한 기기 면·사진 수가 다름: 두 모델을 같은 --data(새 데이터셋 val)로 다시 재세요")
    return problems


def _num(v: Any) -> float:
    return -1.0 if v is None else float(v)


def gate(baseline: Path, candidate: Path, *, min_recall_gain: float = 0.0, now: Now = _now) -> dict[str, Any]:
    """나빠진 항목이 하나도 없고, 하나 이상 좋아졌을 때만 통과. 결과 dict는 그대로 gate.json이 된다."""
    a, b = _load(baseline), _load(candidate)
    if not b.get("model_version"):
        raise RetrainError("후보 평가에 modelVersion이 없음: evaluate --model-version <model.json 값>으로 다시 재세요")
    mismatch = _same_data(a, b)
    if mismatch:
        raise RetrainError("같은 조건의 평가가 아님: " + "; ".join(mismatch))
    rows = [
        # (이름, 기준, 후보, 통과 조건)
        ("기기 면 정확도", a["face"]["accuracy"], b["face"]["accuracy"], _num(b["face"]["accuracy"]) >= _num(a["face"]["accuracy"])),
        ("깨끗한 면 오탐(FP)", a["face"]["fp"], b["face"]["fp"], b["face"]["fp"] <= a["face"]["fp"]),
        ("놓친 흠집 면(FN)", a["face"]["fn"], b["face"]["fn"], b["face"]["fn"] <= a["face"]["fn"]),
        ("사진 재현율", a["photo"]["recall"], b["photo"]["recall"],
         _num(b["photo"]["recall"]) >= _num(a["photo"]["recall"]) + min_recall_gain),
        ("사진 정밀도", a["photo"]["precision"], b["photo"]["precision"],
         _num(b["photo"]["precision"]) >= _num(a["photo"]["precision"]) - 0.05),
    ]
    better = (_num(b["face"]["accuracy"]) > _num(a["face"]["accuracy"])
              or _num(b["photo"]["recall"]) > _num(a["photo"]["recall"])
              or _num(b["photo"]["precision"]) > _num(a["photo"]["precision"]))
    checks = [{"name": n, "baseline": x, "candidate": y, "ok": ok} for n, x, y, ok in rows]
    checks.append({"name": "하나 이상 좋아짐", "baseline": None, "candidate": None, "ok": better})
    return {
        "schema": GATE_SCHEMA, "passed": all(c["ok"] for c in checks), "decidedAt": now().isoformat(),
        "split": b["split"], "conf": b["conf"], "minPhotos": b["min_photos"], "minRecallGain": min_recall_gain,
        "baseline": {"modelVersion": a.get("model_version"), "weightsSha256": a.get("weights_sha256"),
                     "file": str(baseline)},
        "candidate": {"modelVersion": b["model_version"], "weightsSha256": b.get("weights_sha256"),
                      "file": str(candidate)},
        "checks": checks,
    }


# ---------- 3. 레지스트리 ----------

@dataclass(frozen=True)
class Entry:
    modelVersion: str
    onnxSha256: str
    modelJsonSha256: str
    promotedAt: str
    previous: str | None
    kind: str                     # PROMOTE | BOOTSTRAP | ROLLBACK
    gate: dict[str, Any] | None = None
    reviewsUntil: str | None = None
    reviewsSummarySha256: str | None = None
    release: str | None = None


class ModelRegistry:
    """현재 모델과 교체 이력(JSON 파일 하나, 집 PC D:\\trace\\models\\registry.json). 이력은 지우지 않고 더하기만."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _doc(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"schema": REGISTRY_SCHEMA, "current": None, "history": []}
        doc = json.loads(self.path.read_text(encoding="utf-8"))
        if doc.get("schema") != REGISTRY_SCHEMA:
            raise RetrainError(f"모르는 레지스트리 형식: {self.path}")
        return doc

    def history(self) -> list[Entry]:
        return [Entry(**e) for e in self._doc()["history"]]

    def current(self) -> Entry | None:
        doc = self._doc()
        found = [e for e in self.history() if e.modelVersion == doc["current"]]
        return found[-1] if found else None

    def last_reviews_until(self) -> datetime | None:
        times = [e.reviewsUntil for e in self.history() if e.reviewsUntil]
        return max(datetime.fromisoformat(t) for t in times) if times else None

    def add(self, entry: Entry) -> None:
        doc = self._doc()
        doc["history"].append(asdict(entry))
        doc["current"] = entry.modelVersion
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)  # 쓰다 끊겨도 옛 파일이 남는다


def model_files(model_dir: Path) -> tuple[str, Path, Path]:
    onnx, info = model_dir / "best.onnx", model_dir / "model.json"
    if not onnx.is_file() or not info.is_file():
        raise RetrainError(f"best.onnx·model.json이 없음: {model_dir}")
    # Windows PowerShell로 고친 model.json은 BOM이 붙을 수 있다
    version = json.loads(info.read_text(encoding="utf-8-sig")).get("modelVersion", "")
    if not _VERSION.match(version):
        raise RetrainError(f"model.json modelVersion이 비었거나 폴더 이름으로 쓸 수 없음: {version!r}")
    return version, onnx, info


def promote(registry: ModelRegistry, model_dir: Path, *, gate_doc: dict[str, Any] | None,
            reviews: Path | None = None, release: str | None = None, now: Now = _now,
            before_add: Callable[[Entry], None] | None = None) -> Entry:
    """gate_doc 없이 부르면 처음 한 번(레지스트리가 빌 때)만 허용: 지금 쓰는 모델(v004)을 기준으로 올린다."""
    version, onnx, info = model_files(model_dir)
    cur = registry.current()
    if any(e.modelVersion == version for e in registry.history()):
        raise RetrainError(f"{version}은 이미 레지스트리에 있음(되돌리려면 retrain rollback)")
    if gate_doc is None:
        if cur is not None:
            raise RetrainError(f"현재 모델 {cur.modelVersion}이 있어 게이트 없이 올릴 수 없음")
    else:
        if not gate_doc.get("passed"):
            raise RetrainError("게이트를 통과하지 못한 모델은 올리지 않음")
        if gate_doc["candidate"]["modelVersion"] != version:
            raise RetrainError(f"게이트 후보 {gate_doc['candidate']['modelVersion']}와 model.json {version}이 다름")
        if cur is None or gate_doc["baseline"]["modelVersion"] != cur.modelVersion:
            raise RetrainError(f"게이트 기준 모델이 현재 모델({cur.modelVersion if cur else '없음'})이 아님:"
                               " 현재 모델 가중치로 evaluate --model-version을 다시 재세요")
    until = summary_sha = None
    if reviews is not None:
        last = reviews_until(reviews)
        until = last.isoformat() if last else None
        summary_sha = sha256(reviews / "summary.json")
    entry = Entry(version, sha256(onnx), sha256(info), now().isoformat(), cur.modelVersion if cur else None,
                  "PROMOTE" if gate_doc else "BOOTSTRAP", gate_doc, until, summary_sha, release)
    if before_add is not None:
        before_add(entry)  # 배포 묶음을 먼저 다 쓴다: 실패하면 레지스트리는 그대로라 같은 명령을 다시 돌릴 수 있다
    registry.add(entry)
    return entry


def rollback(registry: ModelRegistry, *, now: Now = _now) -> Entry:
    """현재 모델을 바로 앞 모델로 되돌린다(이력에 ROLLBACK을 더함). 서버·Jetson에는 그 모델 릴리스를 다시 배포."""
    cur = registry.current()
    if cur is None or cur.previous is None:
        raise RetrainError("되돌릴 앞 모델이 없음")
    prev = [e for e in registry.history() if e.modelVersion == cur.previous and e.kind != "ROLLBACK"][-1]
    # previous는 그 모델의 앞 모델: rollback을 한 번 더 하면 한 단계 더 뒤로 간다
    entry = Entry(prev.modelVersion, prev.onnxSha256, prev.modelJsonSha256, now().isoformat(), prev.previous,
                  "ROLLBACK", None, None, None, prev.release)
    registry.add(entry)
    return entry
