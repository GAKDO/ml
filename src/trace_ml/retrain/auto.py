"""`trace-ml retrain auto|deploy`: 재학습 파이프라인을 사람 손 없이 잇는다(11/10 강사님 피드백).

ml #33까지는 판단 부품(status·gate·promote)만 있고, 사람이 학교 서버·집 PC·Jetson을 오가며 명령 약 14개로 이었다.
auto는 집 PC 예약 작업 하나로 검수 CSV 받기 → 기준 확인 → 내보내기 → 이어 학습 → 같은 val로 두 모델 평가 → 게이트
→ 레지스트리 등록·배포 묶음까지 하고, 단계별 시간·지표를 report.json·report.md로 남긴다.
배포는 사람이 한 번 승인(`retrain deploy`)하면 학교 서버와 Jetson에 같은 묶음을 올리고 두 해시가 같은지 확인한다.
(AI가 후보를 내고 사람이 확정하는 서비스 원칙을 모델 교체에도 그대로 적용)
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from ..dataset.review_export import CsvReviewSource, psql_command
from .notify import Post, notify
from .pipeline import ModelRegistry, RetrainError, count, gate, model_files

AUTO_SCHEMA = "trace.retrain-auto.v1"
# README 9절의 수동 절차에서 사람이 치던 명령 수(psql·scp·status·export·train·evaluate 2·gate·promote·scp 2·deploy 2·확인)
MANUAL_COMMANDS = 14
_REGISTERED = re.compile(r"부스·서버로 옮길 파일: (.+?) \(best\.onnx")
_CONFIRMED = re.compile(r"확인 끝: (\S+) \(([0-9a-f]{12})\)")

# (argv, 표준 입력 없음) → (종료 코드, 표준 출력, 표준 오류). 테스트에서 바꿔 끼운다.
# 출력과 오류를 나눠야 원격 docker·ssh 경고가 검수 CSV에 섞이지 않는다.
Runner = Callable[[list[str]], tuple[int, str, str]]


def _run(argv: list[str]) -> tuple[int, str, str]:  # pragma: no cover - 실제 프로세스
    env = None
    if argv[:3] == [sys.executable, "-m", "trace_ml"]:
        # Windows(cp949)에서도 하위 trace-ml이 UTF-8로 찍어야 '부스·서버로 옮길 파일' 줄을 읽는다
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          check=False, env=env)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def trace_ml(*args: str) -> list[str]:
    """같은 파이썬으로 trace-ml 하위 명령을 부른다(venv가 달라 다른 trace-ml이 불리는 일을 막음)."""
    return [sys.executable, "-m", "trace_ml", *args]


@dataclass(frozen=True)
class Target:
    """배포 대상 하나. host는 user@주소, ssh 키 로그인이 되어 있어야 한다(비밀번호 입력 불가)."""
    host: str
    port: int = 22
    dir: str = "~/trace-releases"
    script: str = "deploy-server.sh"


@dataclass(frozen=True)
class AutoConfig:
    dataset: str                          # 이어 학습할 데이터셋 버전(예: v004)
    evalData: str                         # 두 모델을 잴 타일 데이터셋(검수분 안 섞은 원래 val)
    baseWeights: str                      # 처음(레지스트리 현재 모델의) best.pt. 이후는 상태 파일이 이어받음
    workRoot: str                         # 실행마다 <workRoot>/<시각>/ 폴더가 생김
    modelName: str = "yolo11m-aihub-rv"
    trainArgs: list[str] = field(default_factory=lambda: ["--tile", "640"])
    minPhotos: int = 120
    minConfirmed: int = 20
    reviewSsh: Target | None = None       # 학교 서버(검수 CSV를 받는 곳). 없으면 CAPTURE_DATABASE_URL
    composeDir: str = "~/gitops/compose"
    server: Target | None = None          # deploy 대상: 학교 서버 inspect-worker
    jetson: Target | None = None          # deploy 대상: 부스 Jetson

    @staticmethod
    def load(path: Path) -> AutoConfig:
        try:
            doc = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            raise RetrainError(f"자동 재학습 설정을 읽지 못함: {path} ({exc})") from exc
        for key, script in (("reviewSsh", "deploy-server.sh"), ("server", "deploy-server.sh"),
                            ("jetson", "deploy-jetson.sh")):
            if doc.get(key):
                doc[key] = Target(**{"script": script, **doc[key]})
        try:
            return AutoConfig(**doc)
        except TypeError as exc:
            raise RetrainError(f"설정 항목이 틀림: {exc}") from exc


def _rq(path: str) -> str:
    """원격 셸에 넘길 경로를 인용한다. 앞의 ~/ 는 홈 확장이 되도록 밖에 둔다(공백·메타문자로 명령이 갈라지지 않게)."""
    if path == "~":
        return path
    if path.startswith("~/"):
        return "~/" + shlex.quote(path[2:])
    return shlex.quote(path)


def _ssh(t: Target, command: str) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-p", str(t.port), t.host, command]


def _scp(t: Target, src: Path, dest: str) -> list[str]:
    return ["scp", "-o", "BatchMode=yes", "-P", str(t.port), "-r", str(src), f"{t.host}:{dest}"]


@dataclass
class Step:
    name: str
    startedAt: str
    seconds: float = 0.0
    ok: bool = False
    detail: str = ""


class Report:
    """단계별 시각·소요 시간·결과. 발표 숫자의 원본이라 실패해도 항상 남긴다."""

    def __init__(self, work: Path, now: Callable[[], datetime]) -> None:
        self.work, self._now = work, now
        self.steps: list[Step] = []
        self.doc: dict[str, Any] = {"schema": AUTO_SCHEMA, "startedAt": now().isoformat(), "status": "RUNNING",
                                    "manualCommandsBefore": MANUAL_COMMANDS}

    def step(self, name: str, fn: Callable[[], str]) -> str:
        s = Step(name, self._now().isoformat())
        self.steps.append(s)
        t0 = time.monotonic()
        try:
            s.detail = fn()
            s.ok = True
            return s.detail
        except Exception as exc:  # 어떤 실패든 그 단계에 이유를 남긴다
            s.detail = str(exc) if isinstance(exc, RetrainError) else f"{type(exc).__name__}: {exc}"
            raise
        finally:
            s.seconds = round(time.monotonic() - t0, 1)

    def write(self, status: str, **extra: Any) -> Path:
        self.doc.update(status=status, finishedAt=self._now().isoformat(), steps=[asdict(s) for s in self.steps],
                        totalSeconds=round(sum(s.seconds for s in self.steps), 1), **extra)
        (self.work / "report.json").write_text(json.dumps(self.doc, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.work / "report.md").write_text(self.markdown(), encoding="utf-8")
        return self.work / "report.json"

    def markdown(self) -> str:
        d = self.doc
        lines = [f"# 자동 재학습 {d['startedAt']}", "", f"- 결과: **{d['status']}**",
                 (f"- 전체 {d['totalSeconds']}초, 사람이 칠 명령 {MANUAL_COMMANDS}개 → "
                  f"{'승인 1개(retrain deploy)' if d['status'] == 'PROMOTED' else '0개'}")]
        if d.get("reviews"):
            r = d["reviews"]
            lines.append(f"- 새 검수: 사진 {r['photos']}/{r['minPhotos']} · 흠집 맞음 {r['confirmed']}/{r['minConfirmed']}"
                         f" · 흠집 아님 {r['dismissed']}")
        if d.get("gate"):
            lines += ["", "| 항목 | 현재 모델 | 새 모델 | 통과 |", "| --- | --- | --- | --- |"]
            lines += [f"| {c['name']} | {'' if c['baseline'] is None else c['baseline']} | "
                      f"{'' if c['candidate'] is None else c['candidate']} | {'O' if c['ok'] else 'X'} |"
                      for c in d["gate"]["checks"]]
        lines += ["", "| 단계 | 시작 | 초 | 결과 |", "| --- | --- | --- | --- |"]
        lines += [f"| {s.name} | {s.startedAt} | {s.seconds} | {'성공' if s.ok else '실패'} |" for s in self.steps]
        if d.get("error"):
            lines += ["", f"오류: {d['error']}"]
        return "\n".join(lines) + "\n"


class _Lock:
    """예약 실행이 겹치면(학습이 하루 넘게 걸릴 때) 두 번째는 바로 멈춘다."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> Self:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RetrainError(f"다른 자동 재학습이 돌고 있음(멈춘 게 확실하면 지우세요): {self.path}") from exc
        except OSError as exc:
            raise RetrainError(f"잠금 파일을 만들지 못함: {exc}") from exc
        try:
            os.write(fd, str(os.getpid()).encode())
        except OSError as exc:
            os.close(fd)
            self.path.unlink(missing_ok=True)  # 이번 시도가 만든 잠금만 지운다
            raise RetrainError(f"잠금 파일을 쓰지 못함: {exc}") from exc
        os.close(fd)
        return self

    def __exit__(self, *_: object) -> None:
        self.path.unlink(missing_ok=True)


def _state_path(reg: ModelRegistry) -> Path:
    return reg.path.parent / "auto-state.json"


def _weights_by_version(reg: ModelRegistry) -> dict[str, str]:
    p = _state_path(reg)
    return json.loads(p.read_text(encoding="utf-8")).get("weights", {}) if p.is_file() else {}


def _remember_weights(reg: ModelRegistry, version: str, weights: Path) -> None:
    doc = {"weights": {**_weights_by_version(reg), version: str(weights)}}
    _state_path(reg).write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def _check(res: tuple[int, str, str], what: str) -> str:
    code, out, err = res
    tail = "\n".join((out + err).strip().splitlines()[-5:])
    if code != 0:
        raise RetrainError(f"{what} 실패(종료 코드 {code}): {tail}")
    return tail


def auto(cfg: AutoConfig, reg: ModelRegistry, env: Mapping[str, str], *, runner: Runner = _run,
         now: Callable[[], datetime] = _now, post: Post | None = None) -> tuple[str, Path]:
    """한 번 돈다. (상태, report.json 경로). 상태: NOT_READY | GATE_FAILED | PROMOTED | FAILED."""
    kw = {"post": post} if post else {}
    try:
        lock = _Lock(Path(cfg.workRoot) / ".auto.lock").__enter__()
    except RetrainError as exc:
        # 남은 잠금 파일 때문에 예약 실행이 매일 조용히 멈추지 않게 알린다
        print(notify(env, f"[TRACE 모델] 자동 재학습 시작 못 함: {exc}", **kw))
        raise
    try:
        work = Path(cfg.workRoot) / now().strftime("%Y%m%dT%H%M%SZ")
        work.mkdir(parents=True, exist_ok=True)
        rep = Report(work, now)
        extra: dict[str, Any] = {}
        status = "FAILED"
        try:
            status = _auto(cfg, reg, env, runner, rep, work, extra)
        except RetrainError as exc:
            extra["error"] = str(exc)
        except Exception as exc:  # noqa: BLE001 - CSV 형식·평가 JSON·실행 오류도 리포트와 알림을 남긴다
            extra["error"] = f"{type(exc).__name__}: {exc}"
        path = rep.write(status, **extra)
    finally:
        lock.__exit__(None, None, None)
    _notify(env, status, rep, post)
    return status, path


def _auto(cfg: AutoConfig, reg: ModelRegistry, env: Mapping[str, str], runner: Runner, rep: Report, work: Path,
          extra: dict[str, Any]) -> str:
    cur = reg.current()
    if cur is None:
        raise RetrainError("레지스트리에 현재 모델이 없음: 처음 한 번 retrain promote로 지금 모델을 올리세요")
    extra["previous"] = cur.modelVersion
    base = Path(_weights_by_version(reg).get(cur.modelVersion) or cfg.baseWeights)
    if not base.is_file():
        raise RetrainError(f"현재 모델 {cur.modelVersion}의 best.pt가 없음: {base}")
    rows = work / "reviews.csv"

    def fetch() -> str:
        if cfg.reviewSsh is None:
            if not env.get("CAPTURE_DATABASE_URL"):
                raise RetrainError("설정 reviewSsh 또는 환경 변수 CAPTURE_DATABASE_URL이 필요함")
            return "CAPTURE_DATABASE_URL에서 직접 읽음"
        sql = psql_command(None).removesuffix(" > reviews.csv")
        code, out, err = runner(_ssh(cfg.reviewSsh, f"cd {_rq(cfg.composeDir)} && {sql}"))
        if code != 0:
            raise RetrainError(f"학교 서버에서 검수 CSV를 받지 못함(ssh 키 로그인 확인): {(out + err).strip()[-300:]}")
        rows.write_text(out, encoding="utf-8")
        return f"{len(out.splitlines()) - 1}행"

    rep.step("검수 결과 받기", fetch)

    def check() -> str:
        if cfg.reviewSsh is None:
            from ..dataset.review_export import PgReviewSource
            source: Any = PgReviewSource(env["CAPTURE_DATABASE_URL"])
        else:
            source = CsvReviewSource(rows)
        st = count(source.photos(None), reg.last_reviews_until(), min_photos=cfg.minPhotos,
                   min_confirmed=cfg.minConfirmed)
        extra["reviews"] = {"since": st.since, "photos": st.photos, "sets": st.sets, "confirmed": st.confirmed,
                            "dismissed": st.dismissed, "minPhotos": st.min_photos, "minConfirmed": st.min_confirmed, "ready": st.ready}
        return f"사진 {st.photos}/{st.min_photos} · 흠집 맞음 {st.confirmed}/{st.min_confirmed}"

    rep.step("재학습 기준 확인", check)
    if not extra["reviews"]["ready"]:
        return "NOT_READY"

    review = work / "review"
    src = ["--rows", str(rows)] if cfg.reviewSsh else []
    rep.step("검수 내보내기", lambda: _check(runner(trace_ml("review", "export", *src, "--out", str(review))),
                                         "review export"))

    registered: list[Path] = []

    def train() -> str:
        code, out, err = runner(trace_ml("train", cfg.dataset, "--base", str(base), "--model-name", cfg.modelName,
                                    "--extra-train", str(review), *cfg.trainArgs))
        _check((code, out, err), "train")
        m = _REGISTERED.search(out)
        if not m:
            raise RetrainError("train 출력에서 registered 폴더를 찾지 못함")
        registered.append(Path(m.group(1).strip()))
        return str(registered[0])

    rep.step("이어 학습", train)
    model_dir = registered[0]
    new_weights = model_dir.parent / "weights" / "best.pt"
    version, _, _ = model_files(model_dir)
    extra["candidate"] = version
    ev_cur, ev_new = work / "eval-current.json", work / "eval-candidate.json"
    for name, w, v, out in (("현재 모델 평가", base, cur.modelVersion, ev_cur),
                            ("새 모델 평가", new_weights, version, ev_new)):
        rep.step(name, lambda w=w, v=v, out=out: _check(runner(trace_ml(
            "evaluate", "--weights", str(w), "--data", cfg.evalData, "--model-version", v, "--out", str(out))),
            "evaluate"))

    gate_path = work / "gate.json"

    def judge() -> str:
        doc = gate(ev_cur, ev_new, model_dir=model_dir)
        gate_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        extra["gate"] = doc
        return "통과" if doc["passed"] else "불통과: " + ", ".join(c["name"] for c in doc["checks"] if not c["ok"])

    rep.step("게이트", judge)
    if not extra["gate"]["passed"]:
        return "GATE_FAILED"
    # 등록보다 먼저 새 버전의 가중치 경로를 적는다. 등록 뒤 저장이 실패하면 다음 실행이 옛 가중치로 이어 학습하게 되므로,
    # 버전 키로 미리 적어 두면 promote가 실패해도 해가 없고(쓰이지 않음) 성공하면 바로 맞는 경로가 된다.
    _remember_weights(reg, version, new_weights)
    rep.step("등록·배포 묶음", lambda: _check(runner(trace_ml(
        "retrain", "--registry", str(reg.path), "promote", "--model-dir", str(model_dir), "--gate", str(gate_path),
        "--reviews", str(review))), "promote"))
    new = reg.current()
    if new is None or new.modelVersion != version:
        raise RetrainError("promote 뒤 레지스트리 현재 모델이 새 모델이 아님")
    extra["release"] = new.release
    return "PROMOTED"


def _notify(env: Mapping[str, str], status: str, rep: Report, post: Post | None) -> None:
    d = rep.doc
    text = {
        "NOT_READY": "재학습 기준 전",
        "GATE_FAILED": f"새 모델 {d.get('candidate')}이 게이트에서 떨어져 현재 모델 {d.get('previous')} 유지",
        "PROMOTED": f"새 모델 {d.get('candidate')} 등록(앞 {d.get('previous')}). 승인하려면 집 PC에서 trace-ml retrain deploy",
        "FAILED": f"자동 재학습 실패: {d.get('error', '')[:200]}",
    }[status]
    if status == "NOT_READY":
        if not env.get("TRACE_RETRAIN_NOTIFY_IDLE"):  # 매일 '기준 전'이 울리지 않게 기본은 조용히
            return
        r = d.get("reviews") or {}
        text += f"(사진 {r.get('photos')}/{r.get('minPhotos')}, 흠집 맞음 {r.get('confirmed')}/{r.get('minConfirmed')})"
    print(notify(env, f"[TRACE 모델] {text} · {d['totalSeconds']}초 · {rep.work / 'report.md'}",
                 **({"post": post} if post else {})))


# ---------- 승인 뒤 배포 ----------

def deploy(cfg: AutoConfig, reg: ModelRegistry, env: Mapping[str, str], *, runner: Runner = _run,
           post: Post | None = None) -> dict[str, Any]:
    """레지스트리 현재 모델의 릴리스 묶음을 서버·Jetson에 올리고, 두 곳 '확인 끝' 해시가 레지스트리와 같은지 본다."""
    cur = reg.current()
    if cur is None or not cur.release:
        raise RetrainError("배포할 현재 모델 릴리스가 없음")
    release = Path(cur.release)
    if not release.is_dir():
        raise RetrainError(f"릴리스 폴더가 없음: {release}")
    targets = [(n, t) for n, t in (("학교 서버", cfg.server), ("부스 Jetson", cfg.jetson)) if t]
    if len(targets) < 2:
        raise RetrainError("설정에 server와 jetson이 모두 있어야 함(모델 일치 원칙: 두 곳 동시 교체)")
    results: dict[str, Any] = {"modelVersion": cur.modelVersion, "onnx": cur.onnxSha256[:12], "targets": {}}
    for name, t in targets:
        t0 = time.monotonic()
        dest = f"{t.dir}/{release.name}"
        try:
            _check(runner(_ssh(t, f"mkdir -p {_rq(t.dir)} && rm -rf {_rq(dest)}")), f"{name} 폴더 준비")
            _check(runner(_scp(t, release, f"{t.dir}/")), f"{name} 복사")
            code, out, err = runner(_ssh(t, f"cd {_rq(dest)} && bash {shlex.quote(t.script)}"))
            m = _CONFIRMED.search(out)
            ok = code == 0 and m is not None and m.group(1) == cur.modelVersion and m.group(2) == cur.onnxSha256[:12]
            detail = m.group(0) if m else "\n".join((out + err).strip().splitlines()[-3:])
        except RetrainError as exc:
            ok, detail = False, str(exc)
        results["targets"][name] = {"ok": ok, "detail": detail, "seconds": round(time.monotonic() - t0, 1)}
    results["matched"] = all(r["ok"] for r in results["targets"].values())
    if results["matched"]:
        text = f"{cur.modelVersion} 서버·Jetson 교체 끝, 두 곳 해시 {cur.onnxSha256[:12]} 일치"
    else:
        bad = [n for n, r in results["targets"].items() if not r["ok"]]
        text = (f"{cur.modelVersion} 배포 실패: {', '.join(bad)}. 두 곳을 같게 맞추려면 "
                "trace-ml retrain rollback 뒤 trace-ml retrain deploy")
    results["message"] = text
    print(notify(env, f"[TRACE 모델] {text}", **({"post": post} if post else {})))
    return results
