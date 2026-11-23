# 이 파일이 확인하는 것: Label Studio 연동(ML3, specs/001-labelstudio-sync/spec.md의 수용 시나리오).
# Label Studio: 사람이 사진에 흠집 다각형을 그려 라벨을 다는 웹 도구.
#   - setup: 프로젝트(흠집 4종 다각형 + "흠집 없음" 체크)를 만들고 원본 사진 저장소(trace-raw)·라벨 저장소(trace-labels)를 연결
#   - sync: 새 사진을 작업으로 가져오고, 제출된 라벨을 라벨 저장소로 내보내기
#   - mark-clean: 고른 기기의 아직 라벨 없는 사진에 "흠집 없음"을 한꺼번에 제출
#   - review: 그린 라벨을 사진 위에 겹친 확인용 모아보기 만들기
# 가짜 Label Studio(fake_ls.py)를 쓰므로 진짜 서버 없이 돈다. 토큰 값은 모두 가짜다.

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
# [상황] 프로젝트가 없는 상태에서 setup을 한다.
# [기대] 프로젝트·소스(trace-raw)·타깃(trace-labels)이 만들어지고, 라벨 설정에 흠집 4종과 "흠집 없음" 선택지가 있다.
def test_프로젝트가_없을_때_setup하면_폴리곤_4클래스와_흠집없음_체크_프로젝트에_소스와_타깃이_연결된다(fake, settings):
    result = setup(client(fake), settings)

    assert result.created == ["프로젝트", "소스 trace-raw", "타깃 trace-labels"]
    config = fake.projects[0]["label_config"]
    assert "<PolygonLabels" in config and 'value="$image"' in config
    assert re.findall(r'<Label value="(\w+)"', config) == list(DEFECT_CLASSES)
    assert '<Choice value="no_defect"' in config
    assert [s["bucket"] for s in fake.storages["s3"]] == ["trace-raw"]
    assert [s["bucket"] for s in fake.storages["export/s3"]] == ["trace-labels"]


# [상황] setup을 두 번 한다.
# [기대] 두 번째는 아무것도 새로 만들지 않는다(중복 프로젝트 없음).
def test_이미_설정된_상태에서_setup을_다시_하면_프로젝트와_스토리지가_하나씩만_있다(fake, settings):
    setup(client(fake), settings)
    again = setup(client(fake), settings)

    assert again.created == [] and again.warnings == []
    assert len(fake.projects) == 1
    assert len(fake.storages["s3"]) == 1 and len(fake.storages["export/s3"]) == 1


# [상황] 새 사진 2장이 올라온 뒤 sync를 하고, 한 번 더 한다.
# [기대] 작업이 0 → 2로 늘었다고 알려 주고, 두 번째는 같은 사진을 다시 작업으로 만들지 않는다.
def test_새_사진이_올라온_뒤_sync를_한_번_하면_소스_동기화가_호출되고_늘어난_작업_수가_보고된다(fake, settings):
    ls = client(fake)
    setup(ls, settings)
    fake.objects["trace-raw"] = ["MB01/MB01_TOP_L2_CAM1_01.jpg", "MB01/MB01_TOP_L2_CAM2_01.jpg"]

    result = sync(ls, settings)

    source_id = fake.storages["s3"][0]["id"]
    assert ("POST", f"/api/storages/s3/{source_id}/sync") in fake.calls
    assert (result.tasks_before, result.tasks_after, result.added) == (0, 2, 2)
    assert sync(ls, settings).added == 0  # 같은 사진은 다시 작업이 되지 않음


# [상황] 소스 연결 설정과 파일 이름 걸러내기 규칙을 본다.
# [기대] 사진은 Label Studio를 거쳐 보여 주고(사전 서명 URL 안 씀), 면 사진 이름 규칙에 맞는 것만 가져오며,
#        촬영 정보판(SLATE) 사진은 뺀다.
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
# [상황] 라벨이 제출된 뒤 sync를 한다.
# [기대] 라벨 저장소로 내보내기 동기화가 호출된다.
def test_주석이_제출된_뒤_sync하면_타깃_동기화가_호출된다(fake, settings):
    ls = client(fake)
    setup(ls, settings)
    sync(ls, settings)

    target_id = fake.storages["export/s3"][0]["id"]
    assert ("POST", f"/api/storages/export/s3/{target_id}/sync") in fake.calls


# [상황] 원본 사진과 라벨을 같은 저장소로 설정한다.
# [기대] "같음" 오류로 시작하지 않는다(원본 사진을 덮어쓰는 사고 방지).
def test_소스와_타깃_버킷을_같게_설정하면_setup이_시작하지_않고_오류를_낸다():
    with pytest.raises(ConfigError, match="같음"):
        load_settings({**ENV, "TRACE_LABELS_BUCKET": "trace-raw"})


# [상황] 라벨 저장소 연결 설정을 본다.
# [기대] Label Studio가 저장소 파일을 지울 수 없고, 원본 사진 저장소에 쓰는 연결은 없다.
def test_타깃은_객체_삭제가_꺼져_있고_trace_raw에_쓰는_설정이_없다(fake, settings):
    setup(client(fake), settings)

    target = fake.storages["export/s3"][0]
    assert target["can_delete_objects"] is False
    assert all(s["bucket"] != "trace-raw" for s in fake.storages["export/s3"])


# 경계 상황
# [상황] 틀린 API 토큰으로 접속한다.
# [기대] "토큰" 문제라고 알리되 토큰 값은 오류 문구에 나오지 않는다.
def test_토큰이_틀리면_무엇이_실패했는지_알리고_토큰_값은_보이지_않는다(fake, settings):
    with pytest.raises(LabelStudioError, match="토큰") as err:
        setup(client(fake, key="wrong-key"), settings)
    assert "wrong-key" not in str(err.value)


# [상황] 같은 이름의 프로젝트가 2개 있다.
# [기대] 아무거나 고르지 않고 "2개" 오류로 멈춘다. 저장소 연결도 하지 않는다.
def test_같은_이름_프로젝트가_둘이면_고르지_않고_멈춘다(fake, settings):
    fake.add_project(settings.project_title, label_config())
    fake.add_project(settings.project_title, label_config())

    with pytest.raises(LabelStudioError, match="2개"):
        setup(client(fake), settings)
    assert fake.storages["s3"] == []


# [상황] 이미 있는 프로젝트의 라벨 설정이 우리 설정과 다르다.
# [기대] 덮어쓰지 않고 경고로 차이를 알린다(사람이 그린 라벨이 망가지지 않게).
def test_기존_프로젝트의_라벨_설정이_다르면_덮어쓰지_않고_차이를_알린다(fake, settings):
    old = "<View><Image name='image' value='$image'/></View>"
    fake.add_project(settings.project_title, old)

    result = setup(client(fake), settings)

    assert any("라벨 설정" in w for w in result.warnings)
    assert fake.projects[0]["label_config"] == old
    assert not any(method == "PATCH" for method, _ in fake.calls)


# [상황] setup 없이 sync를 한다.
# [기대] 먼저 setup하라고 알린다.
def test_setup_전에_sync하면_먼저_setup하라고_알린다(fake, settings):
    with pytest.raises(LabelStudioError, match="setup"):
        sync(client(fake), settings)


# [상황] 동기화가 백그라운드로 돌다가 "버킷 없음"으로 실패한다.
# [기대] 끝날 때까지 기다린 뒤 실패 사유(NoSuchBucket)를 알린다.
def test_비동기_동기화는_끝날_때까지_기다리고_실패하면_사유를_알린다(fake, settings):
    ls = client(fake)
    setup(ls, settings)
    source = fake.storages["s3"][0]
    responses = iter([{**source, "status": "queued"}, {**source, "status": "failed", "traceback": "a\nNoSuchBucket"}])
    ls._call = lambda *a, **k: next(responses)  # type: ignore[method-assign]

    with pytest.raises(LabelStudioError, match="NoSuchBucket"):
        ls.sync_storage("s3", source["id"], sleep=lambda _s: None)


# 명령줄
# [상황] API 키 설정이 빠진 채 setup 명령을 실행한다.
# [기대] 빠진 이름(LABEL_STUDIO_API_KEY)만 알리고 2로 끝나며, 비밀 값은 출력하지 않는다.
def test_환경_변수가_빠지면_이름만_알리고_2로_끝나며_비밀은_출력하지_않는다(capsys):
    env = {k: v for k, v in ENV.items() if k != "LABEL_STUDIO_API_KEY"}
    assert cli.main(["labelstudio", "setup"], env=env) == 2
    out = capsys.readouterr()
    assert "LABEL_STUDIO_API_KEY" in out.err
    assert SECRET not in out.out + out.err


# [상황] 명령줄로 setup·sync를 하고, Label Studio가 꺼진 상태에서 sync를 한다.
# [기대] 결과(작업 0 → 1)를 찍고 비밀은 안 보인다. 접속이 안 되면 "닿지 않음"과 함께 1로 끝난다.
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
# 테스트 도구: 기기 MB02·MB05의 사진 작업이 든 프로젝트를 만든다(일부는 이미 라벨 있음).
def _project_with_tasks(fake: FakeLabelStudio) -> int:
    pid = fake.add_project("TRACE 흠집 라벨링", label_config())["id"]
    for n in range(1, 4):
        fake.add_task(pid, f"MB02/MB02_TOP_L1_CAM1_0{n}.jpg")
    fake.add_task(pid, "MB02/MB02_BOT_L1_CAM1_01.jpg", annotated=True)
    fake.add_task(pid, "MB05/MB05_TOP_L1_CAM1_01.jpg")
    return pid


# [상황] MB02만 골라 "흠집 없음" 일괄 제출을 한다.
# [기대] MB02의 라벨 없는 사진 3장에만 제출되고(이미 있던 1장은 건너뜀), 다른 기기(MB05) 사진은 그대로다.
def test_mark_clean은_고른_기기의_주석_없는_사진에만_no_defect를_제출하고_다른_기기와_그린_사진은_건드리지_않는다(fake, settings):
    _project_with_tasks(fake)

    result = mark_clean(client(fake), settings, ["MB02"])

    assert result.counts == {"MB02": (3, 1)}
    by_key = {t["data"]["image"].rsplit("/", 1)[1]: t for t in fake.tasks}
    assert by_key["MB02_TOP_L1_CAM1_01.jpg"]["annotations"] == [{"result": NO_DEFECT_RESULT}]
    assert by_key["MB02_BOT_L1_CAM1_01.jpg"]["annotations"] == [{"result": []}]
    assert by_key["MB05_TOP_L1_CAM1_01.jpg"]["annotations"] == []


# [상황] 같은 일괄 제출을 두 번 한다.
# [기대] 두 번째는 제출 0장(모두 건너뜀). 라벨이 중복으로 생기지 않는다.
def test_mark_clean을_두_번_해도_주석이_하나씩만_생긴다(fake, settings):
    _project_with_tasks(fake)
    mark_clean(client(fake), settings, ["MB02"])

    again = mark_clean(client(fake), settings, ["MB02"])

    assert again.counts == {"MB02": (0, 4)}


# [상황] 미리 보기(dry-run)로 실행한다.
# [기대] 몇 장인지만 세고 실제로 제출하지 않는다.
def test_mark_clean_dry_run은_장수만_세고_제출하지_않는다(fake, settings):
    _project_with_tasks(fake)

    result = mark_clean(client(fake), settings, ["MB02"], dry_run=True)

    assert result.counts == {"MB02": (3, 1)}
    assert not any(m == "POST" and "annotations" in p for m, p in fake.calls)


# [상황] 기기 ID 형식이 아닌 값을 준다.
# [기대] "형식" 오류로 아무것도 제출하지 않는다.
def test_mark_clean에_기기_ID_형식이_아닌_값을_주면_아무것도_제출하지_않는다(fake, settings):
    _project_with_tasks(fake)

    with pytest.raises(LabelStudioError, match="형식"):
        mark_clean(client(fake), settings, ["MB0"])
    assert all(t["total_annotations"] == 0 for t in fake.tasks if "MB05" in t["data"]["image"])


# [상황] 작업 7개를 3개씩, 또는 서버가 5개로 잘라 주는 상황에서 읽는다.
# [기대] 쪽을 끝까지 넘겨 7개를 모두 읽는다(일부 사진을 빠뜨리지 않게).
def test_mark_clean_작업은_페이지를_끝까지_넘겨_모두_읽는다(fake, settings):
    pid = fake.add_project("TRACE 흠집 라벨링", label_config())["id"]
    for n in range(7):
        fake.add_task(pid, f"MB03/MB03_TOP_L{n % 4 + 1}_CAM1_0{n}.jpg")

    assert len(client(fake).tasks(pid, page_size=3)) == 7
    # 서버가 page_size를 5로 잘라도 total까지 모두 읽는다
    assert len(client(fake).tasks(pid, page_size=500)) == 7


# [상황] 기기 없이, 그리고 MB02·MB09(사진 없음)를 주고 명령을 실행한다.
# [기대] 기기 없으면 2로 끝나고, 주면 기기별 제출 장수와 "사진이 없음"을 찍는다.
def test_명령줄_mark_clean은_기기별_장수를_보이고_기기를_안_주면_2로_끝난다(fake, monkeypatch, capsys):
    _project_with_tasks(fake)
    monkeypatch.setattr(cli, "make_client", lambda s: client(fake))

    assert cli.main(["labelstudio", "mark-clean"], env=ENV) == 2
    assert cli.main(["labelstudio", "mark-clean", "MB02", "MB09"], env=ENV) == 0
    out = capsys.readouterr().out
    assert "MB02: no_defect 제출 3장" in out and "MB09" in out and "사진이 없음" in out


# [상황] setup·sync에 쓰지 않는 옵션(기기 ID, dry-run)을 준다.
# [기대] 아무것도 하지 않고 2로 끝난다(잘못 친 명령이 실제로 실행되지 않게).
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


# 테스트 도구: Label Studio 형식의 흠집 다각형 하나를 만든다.
def _polygon(label: str, x: float, y: float) -> dict:
    pts = [[x, y], [x + 2, y], [x + 2, y + 3], [x, y + 3]]
    return {"type": "polygonlabels", "from_name": "defect", "to_name": "image",
            "value": {"points": pts, "polygonlabels": [label]}}


# [상황] MB05의 사진 중 흠집을 그린 것·흠집 없음·안 한 것이 하나씩 있다.
# [기대] 그린 사진만 라벨을 겹친 확인용 이미지로 만들고, 상태표(그린 종류·흠집 없음·안 함)와 zip 묶음을 남긴다.
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


# [상황] 기기 없이, 또는 dry-run을 붙여 review 명령을 실행한다.
# [기대] 둘 다 2로 끝나고 Label Studio를 부르지 않는다.
def test_명령줄_review는_기기를_안_주면_2로_끝나고_dry_run을_받지_않는다(fake, monkeypatch):
    monkeypatch.setattr(cli, "make_client", lambda s: client(fake))

    assert cli.main(["labelstudio", "review"], env=ENV) == 2
    assert cli.main(["labelstudio", "review", "MB05", "--dry-run"], env=ENV) == 2
    assert fake.calls == []


# [상황] "흠집 없음" 사진도 포함하는 옵션으로 review한다.
# [기대] 흠집 없음 사진도 확인용 이미지로 만든다(작업자가 놓친 흠집을 찾아볼 수 있게).
def test_review_with_clean은_no_defect_사진도_넣어_빠진_흠집을_볼_수_있게_한다(fake, settings, tmp_path):
    pid = fake.add_project("TRACE 흠집 라벨링", label_config())["id"]
    clean = fake.add_task(pid, "MB05/MB05_BOT_L2_CAM1_01.jpg")
    clean["annotations"] = [{"id": 2, "result": NO_DEFECT_RESULT}]

    result = review_sheets(client(fake), settings, ["MB05"], tmp_path / "r", lambda b, k: _jpeg(),
                           include_clean=True)

    assert result.counts == {"MB05": [0, 1, 0]} and result.sheets == 1
    assert [p.name for p in (tmp_path / "r" / "MB05").glob("*.jpg")] == [f"MB05_BOT_L2_CAM1_01__task{clean['id']}__clean.jpg"]


# [상황] 점이 모자란 잘못된 다각형이 있고, 폴더에 이전 실행 파일이 남아 있다.
# [기대] 잘못된 다각형은 그리지 않고 상태표에 "잘못된 폴리곤 1개"로 남기며, zip에는 이번 결과만 넣는다.
def test_review는_점이_모자란_폴리곤을_그리지_않고_상태표에_남기며_이전_실행_파일은_zip에_넣지_않는다(fake, settings, tmp_path):
    pid = fake.add_project("TRACE 흠집 라벨링", label_config())["id"]
    bad = fake.add_task(pid, "MB05/MB05_TOP_L2_CAM1_01.jpg")
    short = {"type": "polygonlabels", "value": {"points": [[1, 1], [2, 2]], "polygonlabels": ["dent"]}}
    bad["annotations"] = [{"id": 1, "result": [short]}]
    out = tmp_path / "review"
    (out / "MB05").mkdir(parents=True)
    (out / "MB05" / "old.jpg").write_bytes(b"x")

    result = review_sheets(client(fake), settings, ["MB05"], out, lambda b, k: _jpeg())

    assert result.sheets == 0
    assert "잘못된 폴리곤 1개" in (out / "summary.csv").read_text(encoding="utf-8-sig")
    import zipfile
    assert zipfile.ZipFile(result.zip_path).namelist() == ["review/summary.csv"]


# [상황] 작업의 사진 주소가 원본 사진 저장소(trace-raw)가 아니다.
# [기대] "trace-raw" 오류로 거부한다(엉뚱한 저장소의 파일을 읽지 않게).
def test_review는_trace_raw가_아닌_사진_주소를_거부한다(fake, settings, tmp_path):
    pid = fake.add_project("TRACE 흠집 라벨링", label_config())["id"]
    task = fake.add_task(pid, "MB05/MB05_TOP_L2_CAM1_01.jpg")
    task["data"]["image"] = "s3://other/MB05/MB05_TOP_L2_CAM1_01.jpg"
    task["annotations"] = [{"id": 1, "result": [_polygon("dent", 10, 10)]}]

    with pytest.raises(LabelStudioError, match="trace-raw"):
        review_sheets(client(fake), settings, ["MB05"], tmp_path / "r", lambda b, k: _jpeg())

