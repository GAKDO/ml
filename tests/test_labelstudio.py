"""ML3 수용 시나리오 (specs/001-labelstudio-sync/spec.md)."""

from __future__ import annotations

import re

import pytest
from fake_ls import BASE, FakeLabelStudio

from trace_ml import __main__ as cli
from trace_ml.codes import DEFECT_CLASSES
from trace_ml.labelstudio.clean import NO_DEFECT_RESULT, mark_clean
from trace_ml.labelstudio.client import LabelStudio, LabelStudioError
from trace_ml.labelstudio.config import KEY_REGEX, ConfigError, Settings, label_config, load_settings
from trace_ml.labelstudio.review import review_sheets
from trace_ml.labelstudio.setup import setup, sync

SECRET = "minio-secret-value"
ENV = {
    "LABEL_STUDIO_URL": BASE,
    "LABEL_STUDIO_API_KEY": "test-key",
    "MINIO_ENDPOINT": "http://minio.test:9000",
    "LABEL_STUDIO_MINIO_ACCESS_KEY": "labelstudio",
    "LABEL_STUDIO_MINIO_SECRET_KEY": SECRET,
}


@pytest.fixture
def fake() -> FakeLabelStudio:
    return FakeLabelStudio()


@pytest.fixture
def settings() -> Settings:
    return load_settings(ENV)


def client(fake: FakeLabelStudio, key: str = "test-key") -> LabelStudio:
    return LabelStudio(BASE, key, session=fake)


# 스토리 1
def test_프로젝트가_없을_때_setup하면_폴리곤_4클래스와_흠집없음_체크_프로젝트에_소스와_타깃이_연결된다(fake, settings):
    result = setup(client(fake), settings)

    assert result.created == ["프로젝트", "소스 trace-raw", "타깃 trace-labels"]
    config = fake.projects[0]["label_config"]
    assert "<PolygonLabels" in config and 'value="$image"' in config
    assert re.findall(r'<Label value="(\w+)"', config) == list(DEFECT_CLASSES)
    assert '<Choice value="no_defect"' in config
    assert [s["bucket"] for s in fake.storages["s3"]] == ["trace-raw"]
    assert [s["bucket"] for s in fake.storages["export/s3"]] == ["trace-labels"]


def test_이미_설정된_상태에서_setup을_다시_하면_프로젝트와_스토리지가_하나씩만_있다(fake, settings):
    setup(client(fake), settings)
    again = setup(client(fake), settings)

    assert again.created == [] and again.warnings == []
    assert len(fake.projects) == 1
    assert len(fake.storages["s3"]) == 1 and len(fake.storages["export/s3"]) == 1


def test_새_사진이_올라온_뒤_sync를_한_번_하면_소스_동기화가_호출되고_늘어난_작업_수가_보고된다(fake, settings):
    ls = client(fake)
    setup(ls, settings)
    fake.objects["trace-raw"] = ["MB01/MB01_TOP_L2_CAM1_01.jpg", "MB01/MB01_TOP_L2_CAM2_01.jpg"]

    result = sync(ls, settings)

    source_id = fake.storages["s3"][0]["id"]
    assert ("POST", f"/api/storages/s3/{source_id}/sync") in fake.calls
    assert (result.tasks_before, result.tasks_after, result.added) == (0, 2, 2)
    assert sync(ls, settings).added == 0  # 같은 사진은 다시 작업이 되지 않음


def test_소스는_프록시_모드이고_면_사진_파일명만_가져오며_SLATE는_뺀다(fake, settings):
    ls = client(fake)
    setup(ls, settings)
    source = fake.storages["s3"][0]
    assert source["presign"] is False and source["use_blob_urls"] is True

    fake.objects["trace-raw"] = [
        "MB01/MB01_SCR1_OFF_CAM1_01.jpg",
        "MB01/slate/MB01_SLATE_CAM1_01.jpg",
        "MB01/manifest.json",
        "MB01/MB01_XX_L2_CAM1_01.jpg",
    ]
    assert sync(ls, settings).added == 1
    assert re.search(KEY_REGEX, "GR03/GR03_KB_L4_CAM2_12.jpg")


# 스토리 2
def test_주석이_제출된_뒤_sync하면_타깃_동기화가_호출된다(fake, settings):
    ls = client(fake)
    setup(ls, settings)
    sync(ls, settings)

    target_id = fake.storages["export/s3"][0]["id"]
    assert ("POST", f"/api/storages/export/s3/{target_id}/sync") in fake.calls


def test_소스와_타깃_버킷을_같게_설정하면_setup이_시작하지_않고_오류를_낸다():
    with pytest.raises(ConfigError, match="같음"):
        load_settings({**ENV, "TRACE_LABELS_BUCKET": "trace-raw"})


def test_타깃은_객체_삭제가_꺼져_있고_trace_raw에_쓰는_설정이_없다(fake, settings):
    setup(client(fake), settings)

    target = fake.storages["export/s3"][0]
    assert target["can_delete_objects"] is False
    assert all(s["bucket"] != "trace-raw" for s in fake.storages["export/s3"])


# 경계 상황
def test_토큰이_틀리면_무엇이_실패했는지_알리고_토큰_값은_보이지_않는다(fake, settings):
    with pytest.raises(LabelStudioError, match="토큰") as err:
        setup(client(fake, key="wrong-key"), settings)
    assert "wrong-key" not in str(err.value)


def test_같은_이름_프로젝트가_둘이면_고르지_않고_멈춘다(fake, settings):
    fake.add_project(settings.project_title, label_config())
    fake.add_project(settings.project_title, label_config())

    with pytest.raises(LabelStudioError, match="2개"):
        setup(client(fake), settings)
    assert fake.storages["s3"] == []


def test_기존_프로젝트의_라벨_설정이_다르면_덮어쓰지_않고_차이를_알린다(fake, settings):
    old = "<View><Image name='image' value='$image'/></View>"
    fake.add_project(settings.project_title, old)

    result = setup(client(fake), settings)

    assert any("라벨 설정" in w for w in result.warnings)
    assert fake.projects[0]["label_config"] == old
    assert not any(method == "PATCH" for method, _ in fake.calls)


def test_setup_전에_sync하면_먼저_setup하라고_알린다(fake, settings):
    with pytest.raises(LabelStudioError, match="setup"):
        sync(client(fake), settings)


def test_비동기_동기화는_끝날_때까지_기다리고_실패하면_사유를_알린다(fake, settings):
    ls = client(fake)
    setup(ls, settings)
    source = fake.storages["s3"][0]
    responses = iter([{**source, "status": "queued"}, {**source, "status": "failed", "traceback": "a\nNoSuchBucket"}])
    ls._call = lambda *a, **k: next(responses)  # type: ignore[method-assign]

    with pytest.raises(LabelStudioError, match="NoSuchBucket"):
        ls.sync_storage("s3", source["id"], sleep=lambda _s: None)


# 명령줄
def test_환경_변수가_빠지면_이름만_알리고_2로_끝나며_비밀은_출력하지_않는다(capsys):
    env = {k: v for k, v in ENV.items() if k != "LABEL_STUDIO_API_KEY"}
    assert cli.main(["labelstudio", "setup"], env=env) == 2
    out = capsys.readouterr()
    assert "LABEL_STUDIO_API_KEY" in out.err
    assert SECRET not in out.out + out.err


def test_명령줄_setup과_sync는_결과를_보이고_Label_Studio에_닿지_않으면_1로_끝난다(fake, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_client", lambda s: LabelStudio(s.url, s.api_key, session=fake))
    assert cli.main(["labelstudio", "setup"], env=ENV) == 0
    fake.objects["trace-raw"] = ["MB01/MB01_TOP_L2_CAM1_01.jpg"]
    assert cli.main(["labelstudio", "sync"], env=ENV) == 0
    out = capsys.readouterr().out
    assert "작업 0 → 1 (+1)" in out
    assert SECRET not in out and "test-key" not in out

    fake.unreachable = True
    assert cli.main(["labelstudio", "sync"], env=ENV) == 1
    assert "닿지 않음" in capsys.readouterr().err


# 흠집 없는 기기 일괄 no_defect
def _project_with_tasks(fake: FakeLabelStudio) -> int:
    pid = fake.add_project("TRACE 흠집 라벨링", label_config())["id"]
    for n in range(1, 4):
        fake.add_task(pid, f"MB02/MB02_TOP_L1_CAM1_0{n}.jpg")
    fake.add_task(pid, "MB02/MB02_BOT_L1_CAM1_01.jpg", annotated=True)
    fake.add_task(pid, "MB05/MB05_TOP_L1_CAM1_01.jpg")
    return pid


def test_mark_clean은_고른_기기의_주석_없는_사진에만_no_defect를_제출하고_다른_기기와_그린_사진은_건드리지_않는다(fake, settings):
    _project_with_tasks(fake)

    result = mark_clean(client(fake), settings, ["MB02"])

    assert result.counts == {"MB02": (3, 1)}
    by_key = {t["data"]["image"].rsplit("/", 1)[1]: t for t in fake.tasks}
    assert by_key["MB02_TOP_L1_CAM1_01.jpg"]["annotations"] == [{"result": NO_DEFECT_RESULT}]
    assert by_key["MB02_BOT_L1_CAM1_01.jpg"]["annotations"] == [{"result": []}]
    assert by_key["MB05_TOP_L1_CAM1_01.jpg"]["annotations"] == []


def test_mark_clean을_두_번_해도_주석이_하나씩만_생긴다(fake, settings):
    _project_with_tasks(fake)
    mark_clean(client(fake), settings, ["MB02"])

    again = mark_clean(client(fake), settings, ["MB02"])

    assert again.counts == {"MB02": (0, 4)}


def test_mark_clean_dry_run은_장수만_세고_제출하지_않는다(fake, settings):
    _project_with_tasks(fake)

    result = mark_clean(client(fake), settings, ["MB02"], dry_run=True)

    assert result.counts == {"MB02": (3, 1)}
    assert not any(m == "POST" and "annotations" in p for m, p in fake.calls)


def test_mark_clean에_기기_ID_형식이_아닌_값을_주면_아무것도_제출하지_않는다(fake, settings):
    _project_with_tasks(fake)

    with pytest.raises(LabelStudioError, match="형식"):
        mark_clean(client(fake), settings, ["MB0"])
    assert all(t["total_annotations"] == 0 for t in fake.tasks if "MB05" in t["data"]["image"])


def test_mark_clean_작업은_페이지를_끝까지_넘겨_모두_읽는다(fake, settings):
    pid = fake.add_project("TRACE 흠집 라벨링", label_config())["id"]
    for n in range(7):
        fake.add_task(pid, f"MB03/MB03_TOP_L{n % 4 + 1}_CAM1_0{n}.jpg")

    assert len(client(fake).tasks(pid, page_size=3)) == 7
    # 서버가 page_size를 5로 잘라도 total까지 모두 읽는다
    assert len(client(fake).tasks(pid, page_size=500)) == 7


def test_명령줄_mark_clean은_기기별_장수를_보이고_기기를_안_주면_2로_끝난다(fake, monkeypatch, capsys):
    _project_with_tasks(fake)
    monkeypatch.setattr(cli, "make_client", lambda s: client(fake))

    assert cli.main(["labelstudio", "mark-clean"], env=ENV) == 2
    assert cli.main(["labelstudio", "mark-clean", "MB02", "MB09"], env=ENV) == 0
    out = capsys.readouterr().out
    assert "MB02: no_defect 제출 3장" in out and "MB09" in out and "사진이 없음" in out


def test_setup과_sync에_기기_ID나_dry_run을_주면_아무것도_하지_않고_2로_끝난다(fake, monkeypatch):
    monkeypatch.setattr(cli, "make_client", lambda s: client(fake))

    assert cli.main(["labelstudio", "sync", "--dry-run"], env=ENV) == 2
    assert cli.main(["labelstudio", "setup", "MB02"], env=ENV) == 2
    assert fake.calls == []


# 검수용 모아보기
def _jpeg(w: int = 640, h: int = 360) -> bytes:
    import cv2
    import numpy as np

    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), 90, np.uint8))
    assert ok
    return buf.tobytes()


def _polygon(label: str, x: float, y: float) -> dict:
    pts = [[x, y], [x + 2, y], [x + 2, y + 3], [x, y + 3]]
    return {"type": "polygonlabels", "from_name": "defect", "to_name": "image",
            "value": {"points": pts, "polygonlabels": [label]}}


def test_review는_폴리곤_사진만_모아보기로_만들고_상태표와_zip을_남긴다(fake, settings, tmp_path):
    pid = fake.add_project("TRACE 흠집 라벨링", label_config())["id"]
    drawn = fake.add_task(pid, "MB05/MB05_TOP_L2_CAM1_01.jpg")
    drawn["annotations"] = [{"id": 1, "result": [_polygon("scratch", 80, 10), _polygon("dent", 10, 80)]}]
    clean = fake.add_task(pid, "MB05/MB05_BOT_L2_CAM1_01.jpg")
    clean["annotations"] = [{"id": 2, "result": NO_DEFECT_RESULT}]
    fake.add_task(pid, "MB05/MB05_TOP_L3_CAM1_01.jpg")
    fake.add_task(pid, "MB07/MB07_TOP_L2_CAM1_01.jpg")
    fetched: list[tuple[str, str]] = []

    def fetch(bucket: str, key: str) -> bytes:
        fetched.append((bucket, key))
        return _jpeg()

    result = review_sheets(client(fake), settings, ["MB05"], tmp_path / "review", fetch)

    assert result.counts == {"MB05": [1, 1, 1]}
    assert fetched == [("trace-raw", "MB05/MB05_TOP_L2_CAM1_01.jpg")]
    sheets = list((tmp_path / "review" / "MB05").glob("*.jpg"))
    assert [p.name for p in sheets] == [f"MB05_TOP_L2_CAM1_01__task{drawn['id']}.jpg"]
    summary = (tmp_path / "review" / "summary.csv").read_text(encoding="utf-8-sig")
    assert "scratch dent" in summary and "no_defect" in summary and "안 함" in summary
    assert result.zip_path is not None and result.zip_path.exists()


def test_명령줄_review는_기기를_안_주면_2로_끝나고_dry_run을_받지_않는다(fake, monkeypatch):
    monkeypatch.setattr(cli, "make_client", lambda s: client(fake))

    assert cli.main(["labelstudio", "review"], env=ENV) == 2
    assert cli.main(["labelstudio", "review", "MB05", "--dry-run"], env=ENV) == 2
    assert fake.calls == []


def test_review_with_clean은_no_defect_사진도_넣어_빠진_흠집을_볼_수_있게_한다(fake, settings, tmp_path):
    pid = fake.add_project("TRACE 흠집 라벨링", label_config())["id"]
    clean = fake.add_task(pid, "MB05/MB05_BOT_L2_CAM1_01.jpg")
    clean["annotations"] = [{"id": 2, "result": NO_DEFECT_RESULT}]

    result = review_sheets(client(fake), settings, ["MB05"], tmp_path / "r", lambda b, k: _jpeg(),
                           include_clean=True)

    assert result.counts == {"MB05": [0, 1, 0]} and result.sheets == 1
    assert [p.name for p in (tmp_path / "r" / "MB05").glob("*.jpg")] == [f"MB05_BOT_L2_CAM1_01__task{clean['id']}__clean.jpg"]

