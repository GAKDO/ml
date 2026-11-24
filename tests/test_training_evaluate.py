"""이 파일이 확인하는 것: 학습한 모델의 평가(evaluate)와 과적합 대응 학습 옵션.
평가는 조각 단위가 아니라 "기기 면" 단위(예: MB13 윗면에 흠집이 있나 없나)로 맞혔는지 센다. 실제 판정이 면 단위이기 때문.

기기 면 판정 평가(evaluate)와 과적합 대응 학습 옵션.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fakes import SETTINGS
from test_training import FakeTrainer, built

from trace_ml.__main__ import main
from trace_ml.dataset.cli import Deps
from trace_ml.training import cli as training_cli
from trace_ml.training.evaluate import EvaluateError, evaluate, judge, parse_tile, run_evaluate
from trace_ml.training.record import model_version, train

TRUTH = {
    # MB13 윗면: 사진 2장 중 1장에 흠집 라벨
    "MB13_TOP_L1_CAM1_01_x0_y0": [0], "MB13_TOP_L1_CAM1_01_x512_y0": [],
    "MB13_TOP_L2_CAM1_01_x0_y0": [],
    # MB04 바닥: 깨끗
    "MB04_BOT_L1_CAM1_01_x0_y0": [], "MB04_BOT_L2_CAM1_01_x0_y0": [],
    # MB16 바닥: 흠집 2장
    "MB16_BOT_L1_CAM1_01_x0_y0": [1], "MB16_BOT_L2_CAM1_01_x0_y0": [1],
}


# [상황] 조각 파일 이름(MB13_TOP_L1_CAM1_01_x1024_y440 등)을 해석한다.
# [기대] 원래 사진 이름과 기기 면(MB13_TOP)을 읽는다. 규칙에 안 맞는 이름은 오류.
def test_타일_이름에서_사진과_기기_면을_읽는다():
    assert parse_tile("MB13_TOP_L1_CAM1_01_x1024_y440") == ("MB13_TOP_L1_CAM1_01", "MB13_TOP")
    assert parse_tile("MB02_SCR0_L2_CAM2_03") == ("MB02_SCR0_L2_CAM2_03", "MB02_SCR0")
    with pytest.raises(EvaluateError):
        parse_tile("rehearsal_0001")


# [상황] 흠집 있는 면 2개에서 검출이 있고, 깨끗한 면에는 검출이 없다.
# [기대] 면 단위 정확도 100%. 사진 단위로는 1장을 놓친 것으로 세고, 면별 표에 사진 수·종류가 나온다.
def test_깨끗한_면에_검출이_없고_흠집_면에_검출이_있으면_기기_면_판정이_모두_맞는다():
    preds = {"MB13_TOP_L1_CAM1_01_x512_y0": [("scratch", 0.32)], "MB16_BOT_L2_CAM1_01_x0_y0": [("scratch", 0.3)]}
    photos, faces, rows = judge(TRUTH, preds, conf=0.25)
    assert faces.summary()["accuracy"] == 1.0 and faces.tp == 2 and faces.tn == 1
    # 사진 단위: MB16 L1은 놓침
    assert (photos.tp, photos.fn, photos.fp) == (2, 1, 0)
    mb13 = next(r for r in rows if r.face == "MB13_TOP")
    assert (mb13.labeled_photos, mb13.detected_photos, mb13.photos) == (1, 1, 2)
    assert mb13.classes == {"scratch": 1}


# [상황] 깨끗한 면의 사진 한 장에서만 검출이 나왔다(먼지 등).
# [기대] "최소 1장" 기준이면 오검출, "최소 2장" 기준이면 문제 없음으로 본다.
def test_min_photos가_2면_한_장에만_잡힌_먼지는_문제_없음으로_본다():
    preds = {"MB04_BOT_L1_CAM1_01_x0_y0": [("dent", 0.9)]}
    _, loose, _ = judge(TRUTH, preds, conf=0.25, min_photos=1)
    _, strict, _ = judge(TRUTH, preds, conf=0.25, min_photos=2)
    assert loose.fp == 1 and strict.fp == 0 and strict.tn == 1


# [상황] 신뢰도 0.12와 0.6인 검출이 있고, 문턱 0.1과 0.25로 각각 센다.
# [기대] 문턱마다 결과가 따로 나오고, 0.25에서는 0.12 검출을 세지 않는다.
def test_문턱별로_다시_세고_문턱보다_낮은_검출은_세지_않는다():
    preds = {"MB04_BOT_L1_CAM1_01_x0_y0": [("dent", 0.12)], "MB16_BOT_L1_CAM1_01_x0_y0": [("dent", 0.6)]}
    report = evaluate(TRUTH, preds, split="val", conf=0.25, sweep=(0.1,))
    assert [r["conf"] for r in report.sweep] == [0.1, 0.25]
    assert report.sweep[0]["face"]["fp"] == 1 and report.sweep[1]["face"]["fp"] == 0
    assert json.loads(report.to_json())["face"]["tp"] == 1


# [상황] 정답 라벨에 없는 조각의 검출 결과가 섞여 있다.
# [기대] "라벨 없는 타일" 오류로 멈춘다(엉뚱한 데이터로 평가하지 않게).
def test_라벨_없는_타일의_검출이_있으면_멈춘다():
    with pytest.raises(EvaluateError, match="라벨 없는 타일"):
        judge(TRUTH, {"MB99_TOP_L1_CAM1_01_x0_y0": []}, conf=0.25)


# 테스트 도구: 검증용 조각 사진·라벨과 가짜 모델 파일을 임시 폴더에 만든다.
def _dataset(tmp: Path) -> Path:
    for split_dir in ("images/val", "labels/val"):
        (tmp / split_dir).mkdir(parents=True)
    for tile, classes in TRUTH.items():
        (tmp / "images" / "val" / f"{tile}.jpg").write_bytes(b"jpg")
        (tmp / "labels" / "val" / f"{tile}.txt").write_text("".join(f"{c} 0.1 0.1 0.2 0.2 0.1 0.2\n" for c in classes))
    (tmp / "best.pt").write_bytes(b"w")
    return tmp


# [상황] "trace-ml evaluate" 명령으로 평가한다(실제 예측은 가짜).
# [기대] 화면에 기기 면별 표와 정확도가 나오고, 결과 JSON에 모델 파일 해시가 남는다.
#        같은 결과 파일로 다시 돌리면 덮어쓰지 않고 멈춘다.
def test_evaluate_명령은_기기_면_표와_JSON을_남긴다(tmp_path, capsys):
    data = _dataset(tmp_path)
    seen: dict[str, object] = {}

    def fake_predict(weights, images, conf):
        seen.update(weights=weights, n=len(images), conf=conf)
        return {"MB16_BOT_L1_CAM1_01_x0_y0": [("dent", 0.5)], "MB13_TOP_L1_CAM1_01_x0_y0": [("scratch", 0.4)]}

    out = tmp_path / "report.json"
    args = main.__globals__["argparse"].Namespace(weights=data / "best.pt", data=data, split="val", conf=0.25,
                                                   min_photos=1, out=out)
    assert training_cli.run_evaluate(args, predict=fake_predict) == 0
    assert seen == {"weights": data / "best.pt", "n": len(TRUTH), "conf": 0.1}
    text = capsys.readouterr().out
    assert "MB04_BOT" in text and "기기 면: 정확도 1.0" in text
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["face"]["accuracy"] == 1.0
    assert saved["weights"] == str(data / "best.pt") and len(saved["weights_sha256"]) == 64
    assert saved["model_version"] is None
    # 같은 파일로 다시 돌리면 덮어쓰지 않고 멈춘다
    assert training_cli.run_evaluate(args, predict=fake_predict) == 2


# [상황] 없는 모델 파일, 1.5 같은 잘못된 문턱을 준다.
# [기대] 오류로 실패한다(명령은 종료 코드 2).
def test_evaluate는_가중치가_없거나_설정이_틀리면_실패한다(tmp_path, capsys):
    data = _dataset(tmp_path)
    with pytest.raises(EvaluateError, match="가중치"):
        run_evaluate(tmp_path / "none.pt", data, predict=lambda *a: {})
    assert main(["evaluate", "--weights", str(data / "best.pt"), "--data", str(data), "--conf", "1.5"]) == 2


# [상황] 검증 사진 하나의 라벨 파일이 빠져 있다.
# [기대] "라벨 없는 사진 1장" 오류로 멈춘다.
def test_라벨_없는_사진이_있으면_평가하지_않고_멈춘다(tmp_path):
    data = _dataset(tmp_path)
    (data / "labels" / "val" / "MB04_BOT_L1_CAM1_01_x0_y0.txt").unlink()
    with pytest.raises(EvaluateError, match="라벨 없는 사진 1장"):
        run_evaluate(data / "best.pt", data, predict=lambda *a: {})


# [상황] 층 고정(freeze)·학습률(lr0)·코사인 학습률·조기 종료·모델 이름을 주고 학습한다.
# [기대] 준 옵션만 학습기에 넘기고, 학습률을 주면 최적화 방식을 AdamW로 고정한다.
#        모델 이름은 학습 인자가 아니라 따로 기록되고 모델 버전 이름에 쓰인다.
def test_과적합_옵션은_주어진_것만_ultralytics에_넘기고_lr0를_주면_AdamW로_고정한다(tmp_path):
    w = built()
    trainer = FakeTrainer()
    deps = Deps({"MINIO_ENDPOINT": "http://minio.test", "MINIO_ACCESS_KEY": "ak", "MINIO_SECRET_KEY": "sk"},
                registry=lambda: w.registry, store=lambda: w.store, trainer=trainer)
    argv = ["train", "v001", "--epochs", "1", "--tile", "0", "--cache", str(tmp_path), "--freeze", "10",
            "--lr0", "0.0005", "--cos-lr", "--patience", "30", "--model-name", "yolo11m-aihub"]
    assert main(argv, deps=deps) == 0
    assert trainer.specs[0].extra == {"freeze": 10, "lr0": 0.0005, "cos_lr": True, "patience": 30,
                                      "optimizer": "AdamW"}
    run_doc = json.loads(w.store.get(SETTINGS.models_bucket, w.registry.runs()[0].model_key.replace(
        "best.pt", "run.json")))
    # --model-name은 ultralytics 인자가 아니므로 trainArgs가 아니라 따로 남는다(modelVersion에서 잘려도 원래 이름 보존)
    assert run_doc["trainArgs"]["freeze"] == 10 and run_doc["modelName"] == "yolo11m-aihub"
    assert "model_name" not in run_doc["trainArgs"]
    model = json.loads(w.store.get(SETTINGS.models_bucket, w.registry.runs()[0].model_key.replace(
        "best.pt", "model.json")))
    run_name = w.registry.runs()[0].model_key.split("/")[1]
    assert model["modelVersion"] == f"yolo11m-aihub-v001-{run_name}-trt-fp16"


# [상황] --set으로 회전·mixup 같은 사진 변형(증강) 옵션을 준다.
# [기대] 숫자·참거짓·글자 타입을 살려 학습기에 넘기고 학습 기록에도 남는다.
def test_set으로_준_증강_인자는_타입을_살려_ultralytics에_넘기고_trainArgs에_남는다(tmp_path):
    w = built()
    trainer = FakeTrainer()
    deps = Deps({"MINIO_ENDPOINT": "http://minio.test", "MINIO_ACCESS_KEY": "ak", "MINIO_SECRET_KEY": "sk"},
                registry=lambda: w.registry, store=lambda: w.store, trainer=trainer)
    argv = ["train", "v001", "--epochs", "1", "--tile", "0", "--cache", str(tmp_path), "--patience", "15",
            "--set", "degrees=10", "--set", "mixup=0.15", "--set", "save_period=1", "--set", "amp=false",
            "--set", "auto_augment=augmix"]
    assert main(argv, deps=deps) == 0
    assert trainer.specs[0].extra == {"patience": 15, "degrees": 10, "mixup": 0.15, "save_period": 1,
                                      "amp": False, "auto_augment": "augmix"}
    run_doc = json.loads(w.store.get(SETTINGS.models_bucket, w.registry.runs()[0].model_key.replace(
        "best.pt", "run.json")))
    assert run_doc["trainArgs"]["mixup"] == 0.15


# [상황] --set 형식이 틀렸거나(값 없음), 정해진 값(epochs·학습률·이어 학습 등)을 바꾸려 하거나, 같은 값을 두 번 준다.
# [기대] 학습하지 않고 종료 코드 2를 돌려준다(기록과 실제 학습이 어긋나는 것 방지).
@pytest.mark.parametrize("bad", ["degrees", "degrees=", "epochs=5", "single_cls=true", "lr0=0.001",
                                 "resume=true", "save=false",
                                 "mixup=0.1 --set mixup=0.2"])
def test_set이_형식이_틀리거나_정해진_값을_바꾸려_하면_학습하지_않고_2를_돌려준다(tmp_path, bad):
    w = built()
    trainer = FakeTrainer()
    deps = Deps({"MINIO_ENDPOINT": "http://minio.test", "MINIO_ACCESS_KEY": "ak", "MINIO_SECRET_KEY": "sk"},
                registry=lambda: w.registry, store=lambda: w.store, trainer=trainer)
    sets = [x for part in bad.split(" --set ") for x in ("--set", part)]
    argv = ["train", "v001", "--epochs", "1", "--tile", "0", "--cache", str(tmp_path), *sets]
    assert main(argv, deps=deps) == 2
    assert trainer.specs == []


# [상황] 같은 데이터셋으로 다른 시각에 학습한 두 모델, 그리고 아주 긴 모델 이름.
# [기대] 모델 버전 이름이 실행마다 다르고, 검사 워커가 뒤에 붙이는 이름까지 더해도 64자를 넘지 않는다.
def test_modelVersion은_실행마다_다르고_inspect_worker_접미사를_붙여도_64자를_넘지_않는다():
    a = model_version("yolo11m-aihub", "v002", "20261108T120000Z")
    b = model_version("yolo11m-aihub", "v002", "20261108T150000Z")
    assert a == "yolo11m-aihub-v002-20261108T120000Z-trt-fp16" and a != b
    long = model_version("x" * 80, "v002", "20261108T120000Z")
    assert len(long + "+contour-v1") == 64 and long.endswith("-v002-20261108T120000Z-trt-fp16")


# [상황] 추가 옵션 없이 학습한다.
# [기대] 학습기에 넘기는 추가 인자가 없다(예전과 같은 학습).
def test_옵션을_주지_않으면_기존_학습과_같다(tmp_path):
    w = built()
    trainer = FakeTrainer()
    train(w.store, w.registry, SETTINGS, "v001", base_weights="yolo11s-seg.pt", epochs=1, imgsz=640,
          cache_dir=tmp_path, trainer=trainer)
    assert trainer.specs[0].extra == {}
