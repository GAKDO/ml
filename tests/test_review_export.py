# 이 파일이 확인하는 것: 관리자 검수 결과를 재학습 데이터로 내보내기(11/8).
# "흠집 맞음"은 흠집 라벨로, "흠집 아님"(먼지·테이프·반사 등)은 "이건 흠집이 아니다"라는 음성 예시로
# 다음 학습의 학습 칸(train)에만 더한다. 학교 서버 DB에서 검수 결과를 CSV로 뽑는 SQL도 확인한다.

"""사람 검수 결과 → 재학습(11/8): 흠집 맞음은 라벨, 흠집 아님은 음성 예시로 다음 학습 train에 더한다."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest
from test_training import FakeTrainer
from test_training_tiles import _jpeg, _train, _world

from trace_ml.__main__ import main
from trace_ml.dataset.review_export import (
    CsvReviewSource,
    ReviewedDefect,
    ReviewedPhoto,
    ReviewExportError,
    export,
    group_rows,
    psql_command,
)
from trace_ml.labelstudio.config import ConfigError

WHEN = datetime(2026, 11, 8, 9, 0, tzinfo=UTC)
SQUARE = [(100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)]
FAR = [(1500.0, 800.0), (1600.0, 800.0), (1600.0, 900.0), (1500.0, 900.0)]


# 테스트 도구: 검수된 흠집 후보 하나(결정·사유 포함)를 만든다.
def _defect(did: str, decision: str, poly=SQUARE, cls: str = "dent", reason: str | None = None) -> ReviewedDefect:
    return ReviewedDefect(did, cls, list(poly), 0.31, "aug-v002", decision, reason)


# 테스트 도구: 검수된 사진 한 장을 만든다.
def _photo(pid: str, face: str, *defects: ReviewedDefect) -> ReviewedPhoto:
    return ReviewedPhoto(pid, "SET0000000000000000001", "RNT0000000000000000001", face, "L1", "CAM1",
                         f"sets/SET1/{face}_L1_CAM1.jpg", WHEN, list(defects))


# 테스트 도구: 가짜 검수 결과 출처(정해 둔 사진 목록을 돌려줌).
class FakeSource:
    def __init__(self, photos: list[ReviewedPhoto]) -> None:
        self._photos = photos
        self.since: datetime | None = None

    def photos(self, since: datetime | None) -> list[ReviewedPhoto]:
        self.since = since
        return self._photos


def _fetch(bucket: str, key: str) -> bytes:
    assert bucket == "trace-captures"
    return _jpeg()


# 테스트 도구: 흠집 맞음+아님이 섞인 윗면 사진과, 흠집 아님만 있는 바닥 사진 두 장.
def _two_photos() -> list[ReviewedPhoto]:
    return [
        _photo("PH1", "TOP", _defect("D1", "CONFIRMED"), _defect("D2", "DISMISSED", FAR, reason="DUST")),
        _photo("PH2", "BOT", _defect("D3", "DISMISSED", reason="TAPE")),
    ]


# [상황] 위 두 장을 내보낸다.
# [기대] 윗면의 흠집 맞음(dent)은 YOLO 라벨 한 줄이 되고, 흠집 아님만 있던 바닥 사진은 빈 라벨(깨끗한 예시)이 된다.
#        맞음 1·아님 사유(DUST·TAPE)별 개수가 집계되고 사진 2장이 학습 칸에 들어간다.
def test_흠집_맞음은_YOLO_라벨이_되고_흠집_아님만_있던_사진은_빈_라벨로_남는다(tmp_path):
    out = tmp_path / "rv"
    result = export(FakeSource(_two_photos()), _fetch, "trace-captures", out, now=lambda: WHEN)
    top = (out / "labels" / "train" / "rv_SET0000000000000000001_TOP_L1_CAM1.txt").read_text(encoding="utf-8")
    bot = (out / "labels" / "train" / "rv_SET0000000000000000001_BOT_L1_CAM1.txt").read_text(encoding="utf-8")
    vals = [float(v) for v in top.split()]
    assert vals[0] == 1  # dent = codes.yaml 순서 1
    assert vals[1:3] == pytest.approx([100 / 1920, 100 / 1080], abs=1e-6)
    assert len(top.splitlines()) == 1 and bot == ""
    assert (result.photos, result.positives, result.negatives) == (2, 1, 1)
    assert result.confirmed == {"dent": 1} and result.dismissed == {"DUST": 1, "TAPE": 1}
    assert len(list((out / "images" / "train").glob("*.jpg"))) == 2


# [상황] 내보내기 결과 폴더를 살펴본다.
# [기대] "흠집 아님" 자리는 따로(hard) 기록되고, reviews.jsonl에 흠집마다 결정과 사유가, summary.json에 형식 이름과 종류가 있다.
def test_흠집_아님으로_지운_자리는_hard에_따로_남고_reviews_jsonl에_결정과_사유가_있다(tmp_path):
    out = tmp_path / "rv"
    export(FakeSource(_two_photos()), _fetch, "trace-captures", out, now=lambda: WHEN)
    hard = sorted(p.name for p in (out / "hard" / "train").glob("*.txt"))
    assert hard == ["rv_SET0000000000000000001_BOT_L1_CAM1.txt", "rv_SET0000000000000000001_TOP_L1_CAM1.txt"]
    rows = [json.loads(line) for line in (out / "reviews.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [d["decision"] for d in rows[0]["defects"]] == ["CONFIRMED", "DISMISSED"]
    assert rows[0]["defects"][1]["reason"] == "DUST" and rows[0]["image"]["width"] == 1920
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema"] == "trace.review-export.v1" and summary["classes"][1] == "dent"


# [상황] 이미 파일이 있는 폴더로 내보내거나, 검수 결과가 하나도 없다.
# [기대] "비어 있지 않음"·"없음" 오류로 내보내지 않는다.
def test_비어_있지_않은_폴더나_검수_결과가_없으면_내보내지_않는다(tmp_path):
    out = tmp_path / "rv"
    out.mkdir()
    (out / "old.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ReviewExportError, match="비어 있지 않음"):
        export(FakeSource(_two_photos()), _fetch, "trace-captures", out)
    with pytest.raises(ReviewExportError, match="없음"):
        export(FakeSource([]), _fetch, "trace-captures", tmp_path / "empty")


# [상황] 사진을 읽지 못했거나, 흠집 맞음의 다각형 점이 3개 미만이거나 모르는 종류다.
# [기대] 그 사진은 통째로 빼고 사유를 남긴다(라벨로 못 바꾼 흠집이 "깨끗함"으로 학습되는 것 방지).
def test_읽지_못한_사진과_라벨로_못_바꾼_흠집_맞음이_있는_사진은_빼고_사유를_남긴다(tmp_path):
    # 흠집 맞음을 라벨로 못 바꾼 사진을 남기면 그 흠집이 배경으로 학습되므로 사진째 뺀다
    photos = [_photo("PH1", "TOP", _defect("D1", "CONFIRMED", [(1.0, 1.0), (2.0, 2.0)])),
              _photo("PH2", "BOT", _defect("D2", "CONFIRMED")),
              _photo("PH3", "KB", _defect("D3", "CONFIRMED", cls="smudge")),
              _photo("PH4", "SIDE", _defect("D4", "CONFIRMED"))]
    result = export(FakeSource(photos), lambda b, k: b"not-jpeg" if "BOT" in k else _jpeg(), "trace-captures",
                    tmp_path / "rv")
    assert (result.photos, result.positives, result.negatives) == (1, 1, 0)
    assert result.confirmed == {"dent": 1}
    assert any("3개 미만" in s for s in result.skipped) and any("읽지 못함" in s for s in result.skipped)
    assert any("모르는 클래스" in s for s in result.skipped)
    assert sum("사진을 뺌" in s for s in result.skipped) == 2
    assert [p.stem.split("_")[2] for p in (tmp_path / "rv" / "images" / "train").glob("*.jpg")] == ["SIDE"]


# [상황] 쓸 수 있는 사진이 하나도 없다.
# [기대] "0장" 오류로 멈추고 폴더를 비워 둔다.
def test_내보낸_사진이_0장이면_멈추고_폴더를_비워_둔다(tmp_path):
    out = tmp_path / "rv"
    with pytest.raises(ReviewExportError, match="0장"):
        export(FakeSource([_photo("PH1", "TOP", _defect("D1", "CONFIRMED"))]), lambda b, k: b"bad",
               "trace-captures", out)
    assert not any(out.iterdir())


# 테스트 도구: 학교 서버에서 뽑은 것과 같은 17칸 CSV 파일을 만든다.
def _csv(path: Path, rows: list[list[str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["public_id"] * 17)
        writer.writerows(rows)
    return path


def _row(photo: str, defect: str, when: str, decision: str = "CONFIRMED", reason: str = "",
         asset: str = "") -> list[str]:
    return [photo, "SET1", "RNT1", "TOP", "L1", "CAM1", f"sets/{photo}.jpg", when, defect, "scratch",
            json.dumps([[1, 1], [50, 1], [50, 50]]), "0.42", "aug", decision, reason, "C02SERIAL", asset]


# [상황] 여러 사진·흠집 줄이 있는 CSV를 기준 시각 이후만 골라 읽는다.
# [기대] 기준 이후 사진만 남고, 줄들이 사진별로 묶이며, 결정·사유·좌표·기기 시리얼을 읽는다.
def test_학교_서버_CSV를_사진별로_묶고_since_뒤만_고른다(tmp_path):
    path = _csv(tmp_path / "reviews.csv", [
        _row("PH1", "D1", "2026-11-07T10:00:00+00:00"),
        _row("PH2", "D2", "2026-11-08T10:00:00.123+00:00"),
        _row("PH2", "D3", "2026-11-08T10:00:00.123+00:00", "DISMISSED", "REFLECTION", asset="mb04"),
    ])
    photos = CsvReviewSource(path).photos(datetime(2026, 11, 8, tzinfo=UTC))
    assert [p.photo_id for p in photos] == ["PH2"]
    assert [(d.decision, d.reason) for d in photos[0].defects] == [("CONFIRMED", None), ("DISMISSED", "REFLECTION")]
    assert photos[0].defects[0].polygon[1] == (50.0, 1.0)
    assert photos[0].device_ids() == {"C02SERIAL"}  # 사진의 첫 줄 값을 쓴다


# [상황] 칸 수가 17개가 아닌 CSV.
# [기대] "17개" 오류로 멈춘다.
def test_열_수가_다른_CSV는_멈춘다(tmp_path):
    path = _csv(tmp_path / "bad.csv", [["PH1", "SET1"]])
    with pytest.raises(ReviewExportError, match="17개"):
        CsvReviewSource(path).photos(None)


# [상황] 학교 서버 DB에서 검수 결과를 뽑는 명령을 만든다.
# [기대] 부스 촬영이면서 검수 완료된 세트, 결정된 흠집만 읽고, 기기 자산 번호를 함께 가져온다. 기준 시각이 없어도 동작한다.
def test_SQL은_검수_완료된_부스_빠른_검사와_결정된_흠집만_읽는다():
    cmd = psql_command(datetime(2026, 11, 8, tzinfo=UTC))
    assert cmd.startswith("docker compose exec -T postgres psql") and "\\copy (" in cmd
    assert "cs.source = 'BOOTH'" in cmd and "profile" not in cmd and "review_completed_at is not null" in cmd
    assert "'2026-11-08T00:00:00+00:00'::timestamptz" in cmd and "#>> '{}'" in cmd
    assert "left join rental.device dv" in cmd and "dv.asset_no" in cmd
    assert "null::timestamptz" in psql_command(None)
    rows = [("PH1", "S", "R", "TOP", "L1", "CAM1", "k", WHEN, "D1", "dent", [[0, 0], [1, 0], [1, 1]], 0.5, "m",
             "CONFIRMED", None, "C02X", None)]
    assert group_rows(rows)[0].reviewed_at == WHEN


# [상황] 사진을 받다가 연결이 끊긴다.
# [기대] 만들던 것을 지워, 같은 폴더로 다시 돌리면 정상으로 끝난다.
def test_사진_받기가_중간에_끊기면_만든_것을_지워_같은_폴더로_다시_돌릴_수_있다(tmp_path):
    out = tmp_path / "rv"
    calls = []

    def flaky(bucket: str, key: str) -> bytes:
        calls.append(key)
        if len(calls) == 2:
            raise ConnectionError("터널 끊김")
        return _jpeg()

    with pytest.raises(ConnectionError):
        export(FakeSource(_two_photos()), flaky, "trace-captures", out)
    assert not out.exists() or not any(out.iterdir())
    assert export(FakeSource(_two_photos()), _fetch, "trace-captures", out).photos == 2


# [상황] 내보내기 폴더의 summary.json이 깨졌거나 모르는 형식이다.
# [기대] "읽지 못함"·"모르는 내보내기 형식" 오류로 멈춘다.
def test_깨진_summary_json은_내보내기_오류로_멈춘다(tmp_path):
    from trace_ml.dataset.review_export import read_summary

    out = _export(tmp_path)
    (out / "summary.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ReviewExportError, match="읽지 못함"):
        read_summary(out)
    (out / "summary.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ReviewExportError, match="모르는 내보내기 형식"):
        read_summary(out)


# [상황] 검수 사진 중 하나가 검증(VAL)용 기기(REG02)의 사진이다.
# [기대] 그 사진은 학습 칸에 넣지 않고 "제외됨"으로 기록한다(검증 점수가 부풀지 않게).
def test_검증_기기_사진은_검수분에서_빠져_train에_들어가지_않는다(tmp_path):
    w = _world([(10, 10), (20, 10), (20, 20), (10, 20)])
    out = tmp_path / "rv"
    photos = _two_photos()
    photos[1].asset_no = "reg02"  # BOT 사진은 VAL 기기(REG02) 것
    export(FakeSource(photos), _fetch, "trace-captures", out, now=lambda: WHEN)
    rows = [json.loads(line) for line in (out / "reviews.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[1]["device"] == {"serial": None, "assetNo": "reg02"}
    result = _train(w, tmp_path / "cache", FakeTrainer(), extra_train=out)
    run = json.loads(w.s3.objects[("trace-models", result.prefix + "run.json")][0])
    assert run["extraTrain"]["excludedHeldOut"] == ["rv_SET0000000000000000001_BOT_L1_CAM1"]
    tile_dir = next((tmp_path / "cache" / "datasets").glob("v001-tile640-rv*"))
    names = {p.name for p in (tile_dir / "images" / "train").glob("rv_*.jpg")}
    assert names and not any("_BOT_" in n for n in names)


# 테스트 도구: 흠집 맞음과 흠집 아님이 서로 다른 조각에 오도록 내보내기 폴더를 만든다.
def _export(tmp_path: Path) -> Path:
    out = tmp_path / "rv"
    # 흠집 맞음은 (100,100)~(300,300), 흠집 아님은 오른쪽 아래 끝(1500,800)~(1600,900) → 서로 다른 타일
    export(FakeSource(_two_photos()), _fetch, "trace-captures", out, now=lambda: WHEN)
    return out


# [상황] 데이터셋 + 검수분으로 조각 학습을 한다(빈 조각 비율 0).
# [기대] 검수분은 학습 칸에만 들어가고, "흠집 아님" 자리 조각은 음성 예시로 꼭 남는다.
#        학습 기록에 검수분 장수·사유별 개수·요약 해시가 남는다.
def test_검수분은_train에만_더하고_흠집_아님_자리가_걸친_타일은_빈_타일_고르기에서_빠지지_않는다(tmp_path):
    w = _world([(10, 10), (20, 10), (20, 20), (10, 20)])
    rv = _export(tmp_path)
    result = _train(w, tmp_path / "cache", FakeTrainer(), background_ratio=0, extra_train=rv)
    run = json.loads(w.s3.objects[("trace-models", result.prefix + "run.json")][0])
    counts = run["tiling"]["counts"]
    # train 흠집 타일: 데이터셋 1 + 검수 TOP 1. 빈 타일 비율 0이어도 흠집 아님 자리 타일은 남는다
    assert counts["train"]["defect"] == 2 and counts["train"]["dismissed"] >= 2
    assert counts["train"]["background"] == counts["train"]["dismissed"]
    tile_dir = next((tmp_path / "cache" / "datasets").glob("v001-tile640-rv*"))
    assert any(p.name.startswith("rv_") for p in (tile_dir / "images" / "train").glob("*.jpg"))
    assert not any(p.name.startswith("rv_") for p in (tile_dir / "images" / "val").glob("*.jpg"))
    assert run["extraTrain"]["photos"] == 2 and run["extraTrain"]["dismissed"] == {"DUST": 1, "TAPE": 1}
    assert len(run["extraTrain"]["summarySha256"]) == 64


# [상황] 검수분 없이 학습한다.
# [기대] 학습 기록의 검수분 칸이 비어 있다.
def test_검수분_없이_학습하면_run_json의_extraTrain은_비어_있다(tmp_path):
    w = _world([(10, 10), (20, 10), (20, 20), (10, 20)])
    result = _train(w, tmp_path, FakeTrainer())
    run = json.loads(w.s3.objects[("trace-models", result.prefix + "run.json")][0])
    assert run["extraTrain"] is None


# [상황] 검수분의 종류 목록이 다르거나, 검수 내보내기 폴더가 아닌 곳을 준다.
# [기대] 설정 오류로 학습하지 않는다.
def test_클래스가_다르거나_내보내기_폴더가_아니면_학습하지_않는다(tmp_path):
    w = _world([(10, 10), (20, 10), (20, 20), (10, 20)])
    rv = _export(tmp_path)
    summary = json.loads((rv / "summary.json").read_text(encoding="utf-8"))
    summary["classes"] = ["dent", "scratch", "crack", "stain"]
    (rv / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    trainer = FakeTrainer()
    with pytest.raises(ConfigError, match="클래스"):
        _train(w, tmp_path / "cache", trainer, extra_train=rv)
    with pytest.raises(ConfigError, match="내보내기 폴더가 아님"):
        _train(w, tmp_path / "cache", trainer, extra_train=tmp_path / "nothing")
    assert trainer.specs == []


# [상황] "trace-ml review sql"과 "review export"를 실행한다. 저장소 설정이 없을 때도 실행한다.
# [기대] SQL 명령이 찍히고, 설정이 없으면 빠진 이름을 알리고 2로 끝난다.
#        설정이 있으면 폴더를 만들고 다음 단계(--extra-train) 안내를 찍는다.
def test_review_sql은_명령을_보이고_export는_CSV와_가짜_저장소로_폴더를_만든다(tmp_path, capsys):
    assert main(["review", "sql", "--since", "2026-11-08"], env={}) == 0
    assert "\\copy (" in capsys.readouterr().out
    path = _csv(tmp_path / "reviews.csv", [_row("PH1", "D1", "2026-11-08T10:00:00+00:00")])
    assert main(["review", "export", "--rows", str(path), "--out", str(tmp_path / "rv")], env={}) == 2
    assert "CAPTURE_MINIO_ENDPOINT" in capsys.readouterr().err
    from trace_ml.dataset import review_cli

    args = __import__("argparse").Namespace(command="export", rows=path, out=tmp_path / "rv", since=None)
    assert review_cli.run(args, {}, fetch=lambda env: (_fetch, "trace-captures")) == 0
    printed = capsys.readouterr().out
    assert "흠집 라벨 1" in printed and "--extra-train" in printed
    img = cv2.imread(str(next((tmp_path / "rv" / "images" / "train").glob("*.jpg"))))
    assert isinstance(img, np.ndarray) and img.shape[:2] == (1080, 1920)


# [상황] 검수 결과를 뽑는 SQL을 만든다.
# [기대] 원본 사진 좌표(photo_polygon)를 읽고(11/9), AI가 놓쳐 관리자가 직접 표시한 흠집도 빠뜨리지 않는다.
def test_SQL은_표준_좌표가_아니라_원본_사진_px_흠집을_읽고_사람이_직접_표시한_흠집도_빼지_않는다():
    # 빠른 검사 흠집은 defect.polygon(표준 좌표)이 비어 있거나 다른 좌표계다. 사진 라벨은 photo_polygon(원본 px)으로 만든다(11/9)
    cmd = psql_command(None)
    assert "d.photo_polygon::text" in cmd and "d.polygon::text" not in cmd
    assert "d.source" not in cmd  # AI가 놓쳐 관리자가 직접 표시한 흠집(HUMAN·CONFIRMED)도 흠집 맞음으로 내보낸다
