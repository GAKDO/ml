"""자동 재학습(11/10): 예약 실행 하나로 검수 받기→학습→게이트→등록, 승인 한 번으로 서버·Jetson 동시 배포."""

from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trace_ml.__main__ import main
from trace_ml.retrain.auto import AutoConfig, Target, auto, deploy
from trace_ml.retrain.pipeline import ModelRegistry, RetrainError, promote

NOW = datetime(2026, 11, 11, 18, 0, tzinfo=UTC)


def _csv_text(n: int, confirmed_each: int = 1) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["header"] * 17)
    for i in range(n):
        for k in range(confirmed_each):
            w.writerow([f"PH{i}", f"SET{i // 4}", "RNT1", "TOP", "L1", "CAM1", f"sets/PH{i}.jpg",
                        "2026-11-11T10:00:00+00:00", f"D{i}{k}", "dent", json.dumps([[1, 1], [50, 1], [50, 50]]),
                        "0.42", "v004", "CONFIRMED", "", "C02SERIAL", ""])
    return buf.getvalue()


def _eval(path: Path, version: str, acc: float, fp: int) -> None:
    path.write_text(json.dumps({
        "data_sha256": "d" * 64, "split": "val", "conf": 0.25, "min_photos": 1, "model_version": version,
        "weights_sha256": "ab" * 32, "photo": {"recall": 0.6, "precision": 0.6},
        "face": {"accuracy": acc, "fp": fp, "fn": 0},
        "faces": [{"face": "MB13_TOP", "photos": 6}]}), encoding="utf-8")


def _model(dir_: Path, version: str) -> Path:
    dir_.mkdir(parents=True)
    (dir_ / "best.onnx").write_bytes(version.encode() * 10)
    (dir_ / "model.json").write_text(json.dumps({"modelVersion": version}), encoding="utf-8")
    return dir_


class FakeHosts:
    """학교 서버 ssh·trace-ml 하위 명령을 흉내 낸다. promote만 진짜 CLI로 돌려 레지스트리·묶음을 실제로 만든다."""

    def __init__(self, tmp: Path, *, rows: int = 130, better: bool = True) -> None:
        self.tmp, self.rows, self.better = tmp, rows, better
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> tuple[int, str, str]:
        self.calls.append(argv)
        if argv[0] == "ssh":
            return 0, _csv_text(self.rows), "WARN[0000] 원격 경고는 stderr로 나와 CSV에 섞이면 안 된다\n"
        args = argv[3:]  # python -m trace_ml
        if args[:2] == ["review", "export"]:
            out = Path(args[args.index("--out") + 1])
            out.mkdir(parents=True)
            (out / "reviews.jsonl").write_text(json.dumps({"reviewedAt": "2026-11-11T10:00:00+00:00"}) + "\n")
            (out / "summary.json").write_text("{}")
            return 0, "사진 130장", ""
        if args[0] == "train":
            run = self.tmp / "runs" / "20261111T180000Z"
            _model(run / "registered", "v005")
            (run / "weights").mkdir()
            (run / "weights" / "best.pt").write_bytes(b"pt5")
            return 0, f"부스·서버로 옮길 파일: {run / 'registered'} (best.onnx·model.json, retrain promote --model-dir)\n", ""
        if args[0] == "evaluate":
            v = args[args.index("--model-version") + 1]
            new = v == "v005"
            _eval(Path(args[args.index("--out") + 1]), v, 0.95 if new and self.better else 0.9,
                  0 if new and self.better else (3 if new else 1))
            return 0, "평가", ""
        if args[0] == "retrain":
            code = main(args, env={})
            return code, "", ""
        raise AssertionError(argv)


@pytest.fixture
def setup(tmp_path: Path) -> tuple[AutoConfig, ModelRegistry]:
    reg = ModelRegistry(tmp_path / "models" / "registry.json")
    promote(reg, _model(tmp_path / "v004", "v004"), gate_doc=None)
    base = tmp_path / "v004.pt"
    base.write_bytes(b"pt4")
    cfg = AutoConfig(dataset="v004", evalData=str(tmp_path / "val"), baseWeights=str(base),
                     workRoot=str(tmp_path / "work"), reviewSsh=Target("trace@school", 8822))
    return cfg, reg


def test_검수가_기준을_넘고_새_모델이_나으면_사람_손_없이_등록과_배포_묶음까지_간다(tmp_path, setup):
    cfg, reg = setup
    status, report = auto(cfg, reg, {}, runner=FakeHosts(tmp_path), now=lambda: NOW)
    doc = json.loads(report.read_text(encoding="utf-8"))
    assert status == "PROMOTED" and reg.current().modelVersion == "v005"
    assert [s["name"] for s in doc["steps"]] == ["검수 결과 받기", "재학습 기준 확인", "검수 내보내기", "이어 학습",
                                                 "현재 모델 평가", "새 모델 평가", "게이트", "등록·배포 묶음"]
    assert all(s["ok"] for s in doc["steps"]) and doc["manualCommandsBefore"] == 14
    assert Path(doc["release"], "deploy-jetson.sh").is_file()
    assert "| 기기 면 정확도 | 0.9 | 0.95 | O |" in (report.parent / "report.md").read_text(encoding="utf-8")
    # 다음 재학습은 새 모델 가중치에서 이어 학습한다
    assert json.loads((reg.path.parent / "auto-state.json").read_text())["weights"]["v005"].endswith("best.pt")


def test_검수가_기준_전이면_학습하지_않고_기준까지_남은_수만_남긴다(tmp_path, setup):
    cfg, reg = setup
    hosts = FakeHosts(tmp_path, rows=10)
    status, report = auto(cfg, reg, {}, runner=hosts, now=lambda: NOW)
    assert status == "NOT_READY" and len(hosts.calls) == 1
    assert json.loads(report.read_text())["reviews"]["photos"] == 10


def test_새_모델이_나빠지면_게이트에서_멈추고_현재_모델을_유지한다(tmp_path, setup):
    cfg, reg = setup
    status, report = auto(cfg, reg, {}, runner=FakeHosts(tmp_path, better=False), now=lambda: NOW)
    assert status == "GATE_FAILED" and reg.current().modelVersion == "v004"
    assert "깨끗한 면 오탐" in json.loads(report.read_text())["steps"][-1]["detail"]


def test_실패해도_리포트를_남기고_잠금을_푼다(tmp_path, setup):
    cfg, reg = setup

    def broken(argv: list[str]) -> tuple[int, str, str]:
        return 255, "", "Permission denied (publickey)"

    status, report = auto(cfg, reg, {}, runner=broken, now=lambda: NOW)
    assert status == "FAILED" and "ssh 키" in json.loads(report.read_text())["error"]
    assert not (Path(cfg.workRoot) / ".auto.lock").exists()


def test_승인하면_서버와_Jetson에_같은_묶음을_올리고_두_해시가_같아야_끝난다(tmp_path, setup):
    cfg, reg = setup
    auto(cfg, reg, {}, runner=FakeHosts(tmp_path), now=lambda: NOW)
    cur = reg.current()
    cfg = AutoConfig(**{**cfg.__dict__, "server": Target("trace@school", 8822),
                        "jetson": Target("minseo@jetson2", 22, script="deploy-jetson.sh")})

    def hosts(argv: list[str]) -> tuple[int, str, str]:
        if argv[0] == "ssh" and "bash deploy-" in argv[-1]:
            sha = cur.onnxSha256[:12] if "server" in argv[-1] else "000000000000"
            return 0, f"확인 끝: {cur.modelVersion} ({sha})\n", ""
        return 0, "", ""

    res = deploy(cfg, reg, {}, runner=hosts)
    assert res["targets"]["학교 서버"]["ok"] and not res["targets"]["부스 Jetson"]["ok"]
    assert not res["matched"] and "rollback" in res["message"]


def test_배포_대상이_한_곳뿐이면_모델_일치_원칙으로_거부한다(tmp_path, setup):
    cfg, reg = setup
    auto(cfg, reg, {}, runner=FakeHosts(tmp_path), now=lambda: NOW)
    cfg = AutoConfig(**{**cfg.__dict__, "server": Target("trace@school", 8822)})
    with pytest.raises(RetrainError, match="모델 일치"):
        deploy(cfg, reg, {}, runner=lambda a: (0, "", ""))


def test_설정_파일의_대상은_기본_스크립트가_정해진다(tmp_path):
    p = tmp_path / "auto.json"
    p.write_text(json.dumps({"dataset": "v004", "evalData": "e", "baseWeights": "b", "workRoot": "w",
                             "server": {"host": "a@b", "port": 8822}, "jetson": {"host": "c@d"}}))
    cfg = AutoConfig.load(p)
    assert (cfg.server.script, cfg.jetson.script, cfg.jetson.port) == ("deploy-server.sh", "deploy-jetson.sh", 22)


def test_RetrainError가_아닌_실패도_리포트와_알림을_남긴다(tmp_path, setup):
    cfg, reg = setup
    sent: list[str] = []

    def bad_csv(argv: list[str]) -> tuple[int, str, str]:
        return 0, "header\nonly,three,cols\n", ""

    status, report = auto(cfg, reg, {"TRACE_ALERT_WEBHOOK_URL": "https://hooks.example/x"}, runner=bad_csv,
                          now=lambda: NOW, post=lambda url, body: sent.append(json.dumps(body)) or 204)
    doc = json.loads(report.read_text(encoding="utf-8"))
    assert status == "FAILED" and doc["error"] and doc["steps"][-1]["ok"] is False
    assert sent and not (Path(cfg.workRoot) / ".auto.lock").exists()


def test_잠금이_남아_있으면_시작하지_않고_알린다(tmp_path, setup):
    cfg, reg = setup
    lock = Path(cfg.workRoot) / ".auto.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("123")
    sent: list[str] = []
    with pytest.raises(RetrainError, match="다른 자동 재학습"):
        auto(cfg, reg, {"TRACE_ALERT_WEBHOOK_URL": "https://hooks.example/x"}, runner=FakeHosts(tmp_path),
             now=lambda: NOW, post=lambda url, body: sent.append(json.dumps(body)) or 204)
    assert sent and lock.exists()
