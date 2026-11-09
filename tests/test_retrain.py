"""재학습 파이프라인(11/9): 검수 누적 → 게이트 → 레지스트리 → 서버·Jetson 같은 파일 교체."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from shutil import which

import pytest

from trace_ml.__main__ import main
from trace_ml.dataset.review_export import ReviewedDefect, ReviewedPhoto
from trace_ml.retrain.pipeline import ModelRegistry, RetrainError, count, gate, promote, rollback

T0 = datetime(2026, 11, 9, 1, 0, tzinfo=UTC)


def _photo(i: int, when: datetime, *decisions: str) -> ReviewedPhoto:
    defects = [ReviewedDefect(f"D{i}{k}", "dent", [(0, 0), (1, 0), (1, 1)], 0.3, "v004", d, None)
               for k, d in enumerate(decisions)]
    return ReviewedPhoto(f"P{i}", f"S{i // 4}", "R1", "TOP", "L1", "CAM1", f"k{i}.jpg", when, defects)


def _report(path: Path, version: str | None, *, acc: float, fp: int, fn: int, recall: float, precision: float,
            photos: int = 6) -> Path:
    doc = {"split": "val", "conf": 0.25, "min_photos": 1, "model_version": version, "weights_sha256": "ab" * 32,
           "photo": {"recall": recall, "precision": precision},
           "face": {"accuracy": acc, "fp": fp, "fn": fn},
           "faces": [{"face": "MB13_TOP", "photos": photos}, {"face": "MB13_BOT", "photos": photos}]}
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _model(dir_: Path, version: str) -> Path:
    dir_.mkdir(parents=True)
    (dir_ / "best.onnx").write_bytes(version.encode() * 10)
    (dir_ / "model.json").write_text(json.dumps({"modelVersion": version, "classes": ["scratch", "dent", "stain"]}))
    return dir_


def test_지난_재학습_뒤에_검수된_사진과_확정_흠집이_기준을_넘어야_재학습할_때로_본다():
    photos = [_photo(0, T0, "CONFIRMED")] + [_photo(i, T0 + timedelta(hours=1), "CONFIRMED", "DISMISSED")
                                             for i in range(1, 5)]
    st = count(photos, T0, min_photos=4, min_confirmed=4)
    assert (st.photos, st.confirmed, st.dismissed, st.ready) == (4, 4, 4, True)
    assert not count(photos, T0, min_photos=4, min_confirmed=5).ready


def test_새_모델이_나빠진_항목_없이_하나라도_좋아지면_게이트를_통과한다(tmp_path):
    base = _report(tmp_path / "a.json", "v004", acc=0.833, fp=1, fn=0, recall=0.57, precision=0.6)
    good = _report(tmp_path / "b.json", "v005", acc=1.0, fp=0, fn=0, recall=0.62, precision=0.6)
    assert gate(base, good)["passed"]


def test_깨끗한_면_오탐이_늘면_재현율이_올라도_게이트에서_떨어진다(tmp_path):
    base = _report(tmp_path / "a.json", "v004", acc=1.0, fp=0, fn=0, recall=0.57, precision=0.6)
    worse = _report(tmp_path / "b.json", "v005", acc=0.833, fp=1, fn=0, recall=0.9, precision=0.6)
    doc = gate(base, worse)
    assert not doc["passed"]
    assert [c["name"] for c in doc["checks"] if not c["ok"]] == ["기기 면 정확도", "깨끗한 면 오탐(FP)"]


def test_같은_점수면_좋아진_게_없어_통과하지_않는다(tmp_path):
    a = _report(tmp_path / "a.json", "v004", acc=1.0, fp=0, fn=0, recall=0.57, precision=0.6)
    b = _report(tmp_path / "b.json", "v005", acc=1.0, fp=0, fn=0, recall=0.57, precision=0.6)
    assert not gate(a, b)["passed"]


def test_다른_데이터로_잰_평가는_비교하지_않고_멈춘다(tmp_path):
    a = _report(tmp_path / "a.json", "v004", acc=1.0, fp=0, fn=0, recall=0.5, precision=0.6)
    b = _report(tmp_path / "b.json", "v005", acc=1.0, fp=0, fn=0, recall=0.7, precision=0.6, photos=9)
    with pytest.raises(RetrainError, match="같은 조건"):
        gate(a, b)


def test_게이트_없이는_처음_한_번만_올리고_기준이_현재_모델이_아니면_올리지_않는다(tmp_path):
    reg = ModelRegistry(tmp_path / "registry.json")
    promote(reg, _model(tmp_path / "v004", "v004"), gate_doc=None)
    with pytest.raises(RetrainError, match="게이트 없이"):
        promote(reg, _model(tmp_path / "v005", "v005"), gate_doc=None)
    a = _report(tmp_path / "a.json", "v003", acc=0.8, fp=1, fn=0, recall=0.5, precision=0.6)
    b = _report(tmp_path / "b.json", "v005", acc=1.0, fp=0, fn=0, recall=0.6, precision=0.6)
    with pytest.raises(RetrainError, match="현재 모델"):
        promote(reg, tmp_path / "v005", gate_doc=gate(a, b))
    assert reg.current().modelVersion == "v004"


def test_되돌리면_앞_모델이_현재가_되고_이력은_남는다(tmp_path):
    reg = ModelRegistry(tmp_path / "registry.json")
    promote(reg, _model(tmp_path / "v004", "v004"), gate_doc=None)
    a = _report(tmp_path / "a.json", "v004", acc=0.8, fp=1, fn=0, recall=0.5, precision=0.6)
    b = _report(tmp_path / "b.json", "v005", acc=1.0, fp=0, fn=0, recall=0.6, precision=0.6)
    promote(reg, _model(tmp_path / "v005", "v005"), gate_doc=gate(a, b))
    assert rollback(reg).modelVersion == "v004"
    assert [e.kind for e in reg.history()] == ["BOOTSTRAP", "PROMOTE", "ROLLBACK"]
    with pytest.raises(RetrainError):
        rollback(reg)


@pytest.mark.skipif(sys.platform == "win32" or not which("bash") or not which("sha256sum"),
                    reason="배포 스크립트는 리눅스(학교 서버·Jetson)용")
def test_명령으로_올린_모델은_서버와_Jetson에_같은_해시로_들어간다(tmp_path, capsys):
    reg_path = tmp_path / "models" / "registry.json"
    env = {"TRACE_MODEL_REGISTRY": str(reg_path)}
    assert main(["retrain", "promote", "--model-dir", str(_model(tmp_path / "v004", "v004"))], env=env) == 0
    a = _report(tmp_path / "a.json", "v004", acc=0.8, fp=1, fn=0, recall=0.5, precision=0.6)
    b = _report(tmp_path / "b.json", "v005", acc=1.0, fp=0, fn=0, recall=0.6, precision=0.6)
    assert main(["retrain", "gate", "--baseline", str(a), "--candidate", str(b), "--out",
                 str(tmp_path / "gate.json")], env=env) == 0
    assert main(["retrain", "promote", "--model-dir", str(_model(tmp_path / "v005", "v005")),
                 "--gate", str(tmp_path / "gate.json")], env=env) == 0
    release = reg_path.parent / "releases" / "v005"

    # 가짜 docker(로그에 새 모델 이름)·가짜 trace_booth(엔진 만들고 model.json 복사)
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "docker").write_text('#!/bin/sh\n[ "$2" = logs ] && echo "폰 탐지 모델 v005 (best.onnx)"\nexit 0\n')
    (bin_ / "fakepy").write_text('#!/bin/sh\n# -m trace_booth build-engine --onnx SRC --out OUT\n'
                                 'mkdir -p "$7"; touch "$7/best.engine"; cp "$(dirname "$5")/model.json" "$7/"\n')
    for f in bin_.iterdir():
        f.chmod(0o755)
    compose = tmp_path / "compose"
    (compose / "models").mkdir(parents=True)
    run_env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "TRACE_COMPOSE_DIR": str(compose),
               "TRACE_BOOTH_PYTHON": str(bin_ / "fakepy"), "TRACE_MODELS_ROOT": str(tmp_path / "jetson")}
    for script in ("deploy-server.sh", "deploy-jetson.sh"):
        done = subprocess.run(["bash", str(release / script)], env=run_env, capture_output=True, text=True, check=False)
        assert done.returncode == 0, done.stderr
        assert "확인 끝: v005" in done.stdout
    server = (compose / "models" / "best.onnx").read_bytes()
    jetson = (tmp_path / "jetson" / "current" / "model.json").read_bytes()
    assert server == (release / "best.onnx").read_bytes()
    assert jetson == (release / "model.json").read_bytes()

    # 복사 중 깨진 파일은 어느 쪽에도 들어가지 않는다
    (release / "best.onnx").write_bytes(b"broken")
    broken = subprocess.run(["bash", str(release / "deploy-server.sh")], env=run_env, capture_output=True, text=True, check=False)
    assert broken.returncode == 1 and "해시" in broken.stderr
    assert (compose / "models" / "best.onnx").read_bytes() == server
