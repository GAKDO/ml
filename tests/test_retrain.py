# 이 파일이 확인하는 것: 재학습 파이프라인(11/9). 관리자 검수 결과가 쌓이면 다시 학습하고,
# 새 모델이 더 나을 때만(게이트) 모델 기록부(레지스트리)에 올려, 학교 서버와 부스 Jetson에 같은 파일로 바꿔 넣는다.
# 검수: AI가 찾은 흠집 후보를 관리자가 "흠집 맞음(CONFIRMED)/아님(DISMISSED)"으로 정하는 것.

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


# 테스트 도구: 검수된 사진 한 장(흠집 후보마다 맞음/아님 결정)을 만든다.
def _photo(i: int, when: datetime, *decisions: str) -> ReviewedPhoto:
    defects = [ReviewedDefect(f"D{i}{k}", "dent", [(0, 0), (1, 0), (1, 1)], 0.3, "v004", d, None)
               for k, d in enumerate(decisions)]
    return ReviewedPhoto(f"P{i}", f"S{i // 4}", "R1", "TOP", "L1", "CAM1", f"k{i}.jpg", when, defects)


# 테스트 도구: 모델 평가 결과 파일(기기 면 정확도·오탐·놓침·재현율·정밀도)을 만든다.
def _report(path: Path, version: str | None, *, acc: float, fp: int, fn: int, recall: float, precision: float,
            photos: int = 6, data: str = "d" * 64) -> Path:
    doc = {"data_sha256": data, "split": "val", "conf": 0.25, "min_photos": 1, "model_version": version, "weights_sha256": "ab" * 32,
           "photo": {"recall": recall, "precision": precision},
           "face": {"accuracy": acc, "fp": fp, "fn": fn},
           "faces": [{"face": "MB13_TOP", "photos": photos}, {"face": "MB13_BOT", "photos": photos}]}
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


# 테스트 도구: 모델 폴더(best.onnx·model.json)를 만든다.
def _model(dir_: Path, version: str) -> Path:
    dir_.mkdir(parents=True)
    (dir_ / "best.onnx").write_bytes(version.encode() * 10)
    (dir_ / "model.json").write_text(json.dumps({"modelVersion": version, "classes": ["scratch", "dent", "stain"]}))
    return dir_


# [상황] 지난 재학습 시각 이후 검수된 사진 4장(흠집 맞음 4, 아님 4)이 있다. 기준을 4장·4개, 다음엔 5개로 둔다.
# [기대] 지난 재학습 이전 사진은 세지 않고, 기준 4개면 "재학습할 때", 5개면 아직 아님.
def test_지난_재학습_뒤에_검수된_사진과_확정_흠집이_기준을_넘어야_재학습할_때로_본다():
    photos = [_photo(0, T0, "CONFIRMED")] + [_photo(i, T0 + timedelta(hours=1), "CONFIRMED", "DISMISSED")
                                             for i in range(1, 5)]
    st = count(photos, T0, min_photos=4, min_confirmed=4)
    assert (st.photos, st.confirmed, st.dismissed, st.ready) == (4, 4, 4, True)
    assert not count(photos, T0, min_photos=4, min_confirmed=5).ready


# [상황] 새 모델이 정확도·오탐·재현율 중 좋아진 게 있고 나빠진 건 없다.
# [기대] 게이트를 통과한다.
def test_새_모델이_나빠진_항목_없이_하나라도_좋아지면_게이트를_통과한다(tmp_path):
    base = _report(tmp_path / "a.json", "v004", acc=0.833, fp=1, fn=0, recall=0.57, precision=0.6)
    good = _report(tmp_path / "b.json", "v005", acc=1.0, fp=0, fn=0, recall=0.62, precision=0.6)
    assert gate(base, good)["passed"]


# [상황] 새 모델이 흠집은 더 잘 찾지만(재현율↑) 깨끗한 면을 흠집으로 잘못 잡는 일이 늘었다.
# [기대] 게이트에서 떨어지고, 실패 항목으로 "기기 면 정확도"와 "깨끗한 면 오탐"이 나온다.
#        (깨끗한 노트북에 흠집 판정을 내리면 고객 분쟁이 되므로 오탐 증가는 허용하지 않음)
def test_깨끗한_면_오탐이_늘면_재현율이_올라도_게이트에서_떨어진다(tmp_path):
    base = _report(tmp_path / "a.json", "v004", acc=1.0, fp=0, fn=0, recall=0.57, precision=0.6)
    worse = _report(tmp_path / "b.json", "v005", acc=0.833, fp=1, fn=0, recall=0.9, precision=0.6)
    doc = gate(base, worse)
    assert not doc["passed"]
    assert [c["name"] for c in doc["checks"] if not c["ok"]] == ["기기 면 정확도", "깨끗한 면 오탐(FP)"]


# [상황] 새 모델과 현재 모델 점수가 똑같다.
# [기대] 바꿀 이유가 없으니 통과하지 않는다.
def test_같은_점수면_좋아진_게_없어_통과하지_않는다(tmp_path):
    a = _report(tmp_path / "a.json", "v004", acc=1.0, fp=0, fn=0, recall=0.57, precision=0.6)
    b = _report(tmp_path / "b.json", "v005", acc=1.0, fp=0, fn=0, recall=0.57, precision=0.6)
    assert not gate(a, b)["passed"]


# [상황] 두 평가의 사진 수가 다르거나, 평가 데이터 지문(해시)이 다르다.
# [기대] "같은 조건"·"지문" 오류로 멈춘다(다른 시험지로 잰 점수는 비교하지 않음).
def test_다른_데이터로_잰_평가는_비교하지_않고_멈춘다(tmp_path):
    a = _report(tmp_path / "a.json", "v004", acc=1.0, fp=0, fn=0, recall=0.5, precision=0.6)
    b = _report(tmp_path / "b.json", "v005", acc=1.0, fp=0, fn=0, recall=0.7, precision=0.6, photos=9)
    with pytest.raises(RetrainError, match="같은 조건"):
        gate(a, b)
    c = _report(tmp_path / "c.json", "v005", acc=1.0, fp=0, fn=0, recall=0.7, precision=0.6, data="e" * 64)
    with pytest.raises(RetrainError, match="지문"):
        gate(a, c)


# [상황] 처음 모델은 게이트 없이 올리고, 두 번째도 게이트 없이 올리려 한다. 또 현재 모델이 아닌 v003과 비교한 게이트로 올린다.
# [기대] 처음만 허용되고, 이후엔 "게이트 없이" 오류. 비교 기준이 현재 모델이 아니면 "현재 모델" 오류로 v004가 그대로다.
def test_게이트_없이는_처음_한_번만_올리고_기준이_현재_모델이_아니면_올리지_않는다(tmp_path):
    reg = ModelRegistry(tmp_path / "registry.json")
    promote(reg, _model(tmp_path / "v004", "v004"), gate_doc=None)
    with pytest.raises(RetrainError, match="게이트 없이"):
        promote(reg, _model(tmp_path / "v005", "v005"), gate_doc=None)
    a = _report(tmp_path / "a.json", "v003", acc=0.8, fp=1, fn=0, recall=0.5, precision=0.6)
    b = _report(tmp_path / "b.json", "v005", acc=1.0, fp=0, fn=0, recall=0.6, precision=0.6)
    v005 = tmp_path / "v005"
    with pytest.raises(RetrainError, match="현재 모델"):
        promote(reg, v005, gate_doc=gate(a, b, model_dir=v005))
    assert reg.current().modelVersion == "v004"


# [상황] v004 → v005로 올린 뒤 되돌리기(rollback)를 하고, 한 번 더 되돌린다.
# [기대] v004가 현재 모델이 되고 이력(처음 등록·올림·되돌림)이 남는다. 더 되돌릴 모델이 없으면 오류.
def test_되돌리면_앞_모델이_현재가_되고_이력은_남는다(tmp_path):
    reg = ModelRegistry(tmp_path / "registry.json")
    promote(reg, _model(tmp_path / "v004", "v004"), gate_doc=None)
    a = _report(tmp_path / "a.json", "v004", acc=0.8, fp=1, fn=0, recall=0.5, precision=0.6)
    b = _report(tmp_path / "b.json", "v005", acc=1.0, fp=0, fn=0, recall=0.6, precision=0.6)
    v005 = _model(tmp_path / "v005", "v005")
    promote(reg, v005, gate_doc=gate(a, b, model_dir=v005))
    assert rollback(reg).modelVersion == "v004"
    assert [e.kind for e in reg.history()] == ["BOOTSTRAP", "PROMOTE", "ROLLBACK"]
    with pytest.raises(RetrainError):
        rollback(reg)


# [상황] 명령으로 v004·v005를 올린 뒤, 배포 스크립트를 학교 서버용·Jetson용으로 실행한다(가짜 docker 사용).
#        그다음 배포 묶음의 ONNX를 일부러 망가뜨리고 다시 서버 배포를 한다.
# [기대] 서버와 Jetson에 같은 파일이 들어가고 둘 다 "확인 끝: v005"를 찍는다.
#        망가진 파일은 해시 검사에서 걸려 실패하고, 서버의 기존 모델은 그대로다.
@pytest.mark.skipif(sys.platform == "win32" or not which("bash") or not which("sha256sum"),
                    reason="배포 스크립트는 리눅스(학교 서버·Jetson)용")
def test_명령으로_올린_모델은_서버와_Jetson에_같은_해시로_들어간다(tmp_path, capsys):
    reg_path = tmp_path / "models" / "registry.json"
    env = {"TRACE_MODEL_REGISTRY": str(reg_path)}
    assert main(["retrain", "promote", "--model-dir", str(_model(tmp_path / "v004", "v004"))], env=env) == 0
    a = _report(tmp_path / "a.json", "v004", acc=0.8, fp=1, fn=0, recall=0.5, precision=0.6)
    b = _report(tmp_path / "b.json", "v005", acc=1.0, fp=0, fn=0, recall=0.6, precision=0.6)
    v005 = _model(tmp_path / "v005", "v005")
    assert main(["retrain", "gate", "--baseline", str(a), "--candidate", str(b), "--model-dir", str(v005), "--out",
                 str(tmp_path / "gate.json")], env=env) == 0
    assert main(["retrain", "promote", "--model-dir", str(v005),
                 "--gate", str(tmp_path / "gate.json")], env=env) == 0
    release = reg_path.parent / "releases" / "v005"

    # 가짜 docker(로그에 새 모델 이름)·가짜 trace_booth(엔진 만들고 model.json 복사)
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "docker").write_text('#!/bin/sh\n[ "$2" = logs ] && echo "폰 탐지 모델 v005 (best.onnx)"\n'
                                 '[ "$2" = ps ] && echo inspect-worker\nexit 0\n')
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


# [상황] 배포 묶음을 쓰다 디스크가 가득 차거나, 기록부 저장이 실패하거나, 기록 직후 오류가 난다.
# [기대] 기록 전 실패는 반쯤 쓴 묶음을 치워 다시 돌릴 수 있게 하고,
#        기록된 뒤 실패면 등록된 묶음을 지우지 않는다(기록과 파일이 어긋나지 않게).
def test_배포_묶음을_쓰다_실패하면_레지스트리는_그대로이고_다시_돌릴_수_있다(tmp_path, monkeypatch):
    from trace_ml.retrain import cli as retrain_cli

    reg_path = tmp_path / "models" / "registry.json"
    env = {"TRACE_MODEL_REGISTRY": str(reg_path)}
    model = _model(tmp_path / "v004", "v004")

    def broken(*_args, **_kwargs):
        out = reg_path.parent / "releases" / "v004"
        out.mkdir(parents=True, exist_ok=True)
        (out / "best.onnx").write_bytes(b"half")
        raise OSError("디스크 가득 참")

    monkeypatch.setattr(retrain_cli, "write_release", broken)
    assert main(["retrain", "promote", "--model-dir", str(model)], env=env) == 1
    assert not reg_path.exists() and not (reg_path.parent / "releases" / "v004").exists()
    monkeypatch.undo()

    # 묶음은 썼지만 레지스트리 저장이 실패해도 묶음을 치우고 다시 돌릴 수 있다
    def add_fails(self, entry):
        raise OSError("레지스트리 쓰기 실패")

    monkeypatch.setattr(ModelRegistry, "add", add_fails)
    assert main(["retrain", "promote", "--model-dir", str(model)], env=env) == 1
    assert not (reg_path.parent / "releases" / "v004").exists()
    monkeypatch.undo()

    # 레지스트리에 들어간 뒤 오류가 나면 등록된 묶음은 지우지 않는다
    original = ModelRegistry.add

    def add_then_fail(self, entry):
        original(self, entry)
        raise OSError("기록 뒤 오류")

    monkeypatch.setattr(ModelRegistry, "add", add_then_fail)
    assert main(["retrain", "promote", "--model-dir", str(model)], env=env) == 1
    monkeypatch.undo()
    assert ModelRegistry(reg_path).current().modelVersion == "v004"
    assert (reg_path.parent / "releases" / "v004" / "deploy-server.sh").is_file()


# [상황] 검수 결과가 재학습 기준에 도달했고 알림 주소(디스코드 웹훅)가 설정돼 있다.
# [기대] "재학습 기준 도달" 알림을 한 번 보낸다. 슬랙 주소면 슬랙 형식으로, 주소가 없으면 건너뛰고,
#        알림이 실패하면 HTTP 상태를 알려 준다.
def test_기준에_도달하면_웹훅으로_한_번_알리고_주소가_없으면_건너뛴다(tmp_path, capsys):
    from trace_ml.retrain import cli as retrain_cli
    from trace_ml.retrain.notify import notify, payload

    sent: list[tuple[str, dict]] = []
    rows = tmp_path / "reviews.csv"
    header = ",".join(f"c{i}" for i in range(17))
    lines = [f"P{i},S1,R1,TOP,L1,CAM1,k{i}.jpg,2026-11-09T01:00:00+00:00,D{i},dent,\"[[0,0],[1,0],[1,1]]\","
             f"0.3,v004,CONFIRMED,,SN1,A1" for i in range(3)]
    rows.write_text("\n".join([header, *lines]) + "\n", encoding="utf-8")
    env = {"TRACE_MODEL_REGISTRY": str(tmp_path / "r.json"), "TRACE_ALERT_WEBHOOK_URL": "https://discord.example/x"}
    args = ["retrain", "status", "--rows", str(rows), "--min-photos", "3", "--min-confirmed", "3", "--notify"]
    import argparse

    parser = argparse.ArgumentParser()
    retrain_cli.add_parsers(parser.add_subparsers(dest="tool"))
    assert retrain_cli.run(parser.parse_args(args), env, post=lambda u, p: sent.append((u, p)) or 204) == 0
    assert len(sent) == 1 and "재학습 기준 도달" in sent[0][1]["content"]
    assert "알림 보냄" in capsys.readouterr().out
    assert payload("https://hooks.slack.com/services/x", "hi") == {"text": "hi"}
    assert notify({}, "hi").startswith("알림 건너뜀")
    assert notify({"TRACE_ALERT_WEBHOOK_URL": "u"}, "hi", post=lambda u, p: 500) == "알림 실패: HTTP 500"


# [상황] 게이트를 통과한 뒤 모델 파일(ONNX)이 바뀌었다. 또는 모델 폴더 없이 만든 게이트로 올린다.
# [기대] "ONNX 해시" 오류로 올리지 않는다(검사한 파일과 배포 파일이 같아야 함).
def test_게이트_뒤에_ONNX가_바뀌면_올리지_않는다(tmp_path):
    reg = ModelRegistry(tmp_path / "registry.json")
    promote(reg, _model(tmp_path / "v004", "v004"), gate_doc=None)
    a = _report(tmp_path / "a.json", "v004", acc=0.8, fp=1, fn=0, recall=0.5, precision=0.6)
    b = _report(tmp_path / "b.json", "v005", acc=1.0, fp=0, fn=0, recall=0.6, precision=0.6)
    v005 = _model(tmp_path / "v005", "v005")
    doc = gate(a, b, model_dir=v005)
    (v005 / "best.onnx").write_bytes(b"other")
    with pytest.raises(RetrainError, match="ONNX 해시"):
        promote(reg, v005, gate_doc=doc)
    with pytest.raises(RetrainError, match="ONNX 해시"):
        promote(reg, v005, gate_doc=gate(a, b))  # 모델 폴더 없이 낸 게이트
