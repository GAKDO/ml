"""기기 면 판정 평가(evaluate)와 과적합 대응 학습 옵션."""

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


def test_타일_이름에서_사진과_기기_면을_읽는다():
    assert parse_tile("MB13_TOP_L1_CAM1_01_x1024_y440") == ("MB13_TOP_L1_CAM1_01", "MB13_TOP")
    assert parse_tile("MB02_SCR0_L2_CAM2_03") == ("MB02_SCR0_L2_CAM2_03", "MB02_SCR0")
    with pytest.raises(EvaluateError):
        parse_tile("rehearsal_0001")


def test_깨끗한_면에_검출이_없고_흠집_면에_검출이_있으면_기기_면_판정이_모두_맞는다():
    preds = {"MB13_TOP_L1_CAM1_01_x512_y0": [("scratch", 0.32)], "MB16_BOT_L2_CAM1_01_x0_y0": [("scratch", 0.3)]}
    photos, faces, rows = judge(TRUTH, preds, conf=0.25)
    assert faces.summary()["accuracy"] == 1.0 and faces.tp == 2 and faces.tn == 1
    # 사진 단위: MB16 L1은 놓침
    assert (photos.tp, photos.fn, photos.fp) == (2, 1, 0)
    mb13 = next(r for r in rows if r.face == "MB13_TOP")
    assert (mb13.labeled_photos, mb13.detected_photos, mb13.photos) == (1, 1, 2)
    assert mb13.classes == {"scratch": 1}


def test_min_photos가_2면_한_장에만_잡힌_먼지는_문제_없음으로_본다():
    preds = {"MB04_BOT_L1_CAM1_01_x0_y0": [("dent", 0.9)]}
    _, loose, _ = judge(TRUTH, preds, conf=0.25, min_photos=1)
    _, strict, _ = judge(TRUTH, preds, conf=0.25, min_photos=2)
    assert loose.fp == 1 and strict.fp == 0 and strict.tn == 1


def test_문턱별로_다시_세고_문턱보다_낮은_검출은_세지_않는다():
    preds = {"MB04_BOT_L1_CAM1_01_x0_y0": [("dent", 0.12)], "MB16_BOT_L1_CAM1_01_x0_y0": [("dent", 0.6)]}
    report = evaluate(TRUTH, preds, split="val", conf=0.25, sweep=(0.1,))
    assert [r["conf"] for r in report.sweep] == [0.1, 0.25]
    assert report.sweep[0]["face"]["fp"] == 1 and report.sweep[1]["face"]["fp"] == 0
    assert json.loads(report.to_json())["face"]["tp"] == 1


def test_라벨_없는_타일의_검출이_있으면_멈춘다():
    with pytest.raises(EvaluateError, match="라벨 없는 타일"):
        judge(TRUTH, {"MB99_TOP_L1_CAM1_01_x0_y0": []}, conf=0.25)


def _dataset(tmp: Path) -> Path:
    for split_dir in ("images/val", "labels/val"):
        (tmp / split_dir).mkdir(parents=True)
    for tile, classes in TRUTH.items():
        (tmp / "images" / "val" / f"{tile}.jpg").write_bytes(b"jpg")
        (tmp / "labels" / "val" / f"{tile}.txt").write_text("".join(f"{c} 0.1 0.1 0.2 0.2 0.1 0.2\n" for c in classes))
    (tmp / "best.pt").write_bytes(b"w")
    return tmp


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


def test_evaluate는_가중치가_없거나_설정이_틀리면_실패한다(tmp_path, capsys):
    data = _dataset(tmp_path)
    with pytest.raises(EvaluateError, match="가중치"):
        run_evaluate(tmp_path / "none.pt", data, predict=lambda *a: {})
    assert main(["evaluate", "--weights", str(data / "best.pt"), "--data", str(data), "--conf", "1.5"]) == 2


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
    assert run_doc["trainArgs"]["freeze"] == 10
    model = json.loads(w.store.get(SETTINGS.models_bucket, w.registry.runs()[0].model_key.replace(
        "best.pt", "model.json")))
    run_name = w.registry.runs()[0].model_key.split("/")[1]
    assert model["modelVersion"] == f"yolo11m-aihub-v001-{run_name}-trt-fp16"


def test_modelVersion은_실행마다_다르고_inspect_worker_접미사를_붙여도_64자를_넘지_않는다():
    a = model_version("yolo11m-aihub", "v002", "20261108T120000Z")
    b = model_version("yolo11m-aihub", "v002", "20261108T150000Z")
    assert a == "yolo11m-aihub-v002-20261108T120000Z-trt-fp16" and a != b
    long = model_version("x" * 80, "v002", "20261108T120000Z")
    assert len(long + "+contour-v1") == 64 and long.endswith("-v002-20261108T120000Z-trt-fp16")


def test_옵션을_주지_않으면_기존_학습과_같다(tmp_path):
    w = built()
    trainer = FakeTrainer()
    train(w.store, w.registry, SETTINGS, "v001", base_weights="yolo11s-seg.pt", epochs=1, imgsz=640,
          cache_dir=tmp_path, trainer=trainer)
    assert trainer.specs[0].extra == {}
