# 이 파일이 확인하는 것: 데이터셋 빌더(ML1) 스토리 1. Label Studio에서 제출한 흠집 라벨로
# "기기 단위" 데이터셋 버전(v001, v002…)을 만든다(FR-DAT-04).
# 기기 단위: 한 노트북의 사진은 학습(TRAIN)·검증(VAL)·시험(TEST) 중 한 곳에만 넣는다.
# 같은 노트북이 학습과 시험에 섞이면 점수가 실제보다 좋게 나오기 때문.
# manifest: 데이터셋에 든 사진·라벨 목록과 해시. 나중에 데이터가 바뀌었는지 확인하는 근거.

"""ML1 스토리 1: 제출된 주석으로 기기 단위 데이터셋 버전을 만든다 (FR-DAT-04)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
import yaml
from fakes import NO_DEFECT, SETTINGS, TRI, World, ann, polygon

from trace_ml.__main__ import main
from trace_ml.dataset.annotations import AnnotationErrors, collect
from trace_ml.dataset.builder import BuildError, Commit, build, next_version
from trace_ml.dataset.cli import Deps
from trace_ml.dataset.splits import SplitError, assign, check_change
from trace_ml.registry import Device, VersionExists
from trace_ml.store import AlreadyExists

CLEAN = Commit("a" * 40, False)
NOW = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)


def _build(w: World, **kw):
    return build(w.store, w.registry, SETTINGS, kw.pop("commit", CLEAN), now=NOW, **kw)


def _manifest(w: World, version: str = "v001") -> dict:
    return json.loads(w.s3.objects[("trace-datasets", f"{version}/manifest.json")][0])


# 테스트 도구: 기기 3대(REG03은 미리 시험용으로 정함)에 라벨 사진 3장씩을 만든다.
def three_devices() -> World:
    w = World()
    w.registry.save_device(Device("REG03", "MacBook Air", "TEST"))
    for code in ("REG01", "REG02", "REG03"):
        w.labeled(code, 3)
    return w


# [상황] 기기 3대로 데이터셋을 만든다.
# [기대] 기기마다 사진이 한 칸에만 들어가고 manifest의 기기표와 같다. 사람이 먼저 정한 칸(REG03=TEST)이 우선한다.
def test_한_기기의_사진은_모두_한_분할에만_있고_manifest의_기기표와_같다():
    w = three_devices()
    result = _build(w)
    m = _manifest(w)
    split_of: dict[str, set[str]] = {}
    for item in m["items"]:
        split_of.setdefault(item["device"], set()).add(item["split"])
    assert all(len(s) == 1 for s in split_of.values())
    assert {d: s.pop().upper() for d, s in split_of.items()} == m["devices"]
    assert m["devices"]["REG03"] == "TEST"  # 사람이 먼저 정한 분할이 이긴다
    placed = [c for codes in result.counts["devices"].values() for c in codes]
    assert sorted(placed) == ["REG01", "REG02", "REG03"]


# [상황] 찌그러짐(dent) 다각형 하나가 있는 사진으로 만든다.
# [기대] 라벨 파일은 "종류 번호 + 0~1 좌표" YOLO 형식이고, data.yaml에 종류 이름 4개가 번호와 함께 적힌다.
def test_라벨_파일은_YOLO_분할_형식이고_클래스_번호와_0에서_1_좌표다():
    w = World()
    w.photo("REG01/REG01_TOP_L1_CAM1_01.jpg", ann(1, polygon("dent", [(0, 0), (50, 0), (50, 120)])))
    _build(w)
    body = w.s3.objects[("trace-datasets", "v001/labels/train/REG01_TOP_L1_CAM1_01.txt")][0].decode()
    assert body == "1 0.000000 0.000000 0.500000 0.000000 0.500000 1.000000\n"
    data = yaml.safe_load(w.s3.objects[("trace-datasets", "v001/data.yaml")][0])
    assert data["names"] == {0: "scratch", 1: "dent", 2: "crack", 3: "stain"}
    assert data["train"] == "images/train"


# [상황] 데이터셋을 만든다.
# [기대] manifest에 사진·라벨 해시, 빌더 코드 커밋, codes 버전, 만든 시각이 있고 기록부에도 버전이 남는다.
#        manifest는 맨 마지막에 쓴다(중간에 끊기면 완성된 버전으로 보이지 않게).
def test_manifest에_사진_해시_빌더_커밋_codes_버전이_있고_기록부에_버전이_남는다():
    w = three_devices()
    _build(w)
    m = _manifest(w)
    item = m["items"][0]
    assert len(item["image"]["sha256"]) == 64 and len(item["label"]["sha256"]) == 64
    assert m["builderCommit"] == "a" * 40 and m["codesVersion"] and m["builtAt"] == "2026-11-05T12:00:00Z"
    row = w.registry.versions()[0]
    assert (row.version, row.manifest_key, row.builder_commit) == ("v001", "v001/manifest.json", "a" * 40)
    assert len(row.counts["manifestSha256"]) == 64
    # manifest는 마지막에 쓴다
    assert w.s3.puts[-1] == ("trace-datasets", "v001/manifest.json")


# [상황] "흠집 없음"으로 제출한 사진.
# [기대] 빈 라벨 파일로 들어가고 배경(깨끗한 사진) 수로 센다.
def test_흠집_없음으로_제출한_사진은_빈_라벨_파일_배경이다():
    w = World()
    w.photo("REG01/REG01_TOP_L1_CAM1_01.jpg", ann(1, NO_DEFECT))
    result = _build(w)
    assert w.s3.objects[("trace-datasets", "v001/labels/train/REG01_TOP_L1_CAM1_01.txt")][0] == b""
    assert result.counts["background"]["train"] == 1


# [상황] 라벨 작업자가 건너뛴(Skip) 사진이 섞여 있다.
# [기대] 그 사진은 데이터셋에 넣지 않는다.
def test_건너뛴_주석이_섞여_있으면_그_사진은_데이터셋에_없다():
    w = World()
    w.labeled("REG01", 1)
    w.photo("REG01/REG01_BOT_L1_CAM1_01.jpg", ann(7, cancelled=True))
    _build(w)
    keys = [i["image"]["key"] for i in _manifest(w)["items"]]
    assert keys == ["REG01/REG01_TOP_L1_CAM1_01.jpg"]


# [상황] 한 사진에 라벨이 여러 번 달렸다. 하나는 "정답(ground truth)" 표시가 있다.
# [기대] 정답 표시가 있으면 그것을, 없으면 가장 최근 것을 쓴다.
def test_주석이_여럿이면_ground_truth_없으면_가장_최근_것을_쓴다():
    old = ann(1, polygon("scratch", TRI), updated="2026-11-03T00:00:00Z")
    new = ann(2, polygon("dent", TRI), polygon("dent", TRI), updated="2026-11-04T00:00:00Z")
    gt = ann(3, NO_DEFECT, updated="2026-11-01T00:00:00Z", ground_truth=True)
    doc = {"id": 5, "data": {"image": "s3://trace-raw/REG01/REG01_TOP_L1_CAM1_01.jpg"}}
    assert [p.cls for p in collect([{**doc, "annotations": [old, new]}])[0].polygons] == [1, 1]
    assert collect([{**doc, "annotations": [old, new, gt]}])[0].annotation_id == 3


# [상황] 라벨을 지우고 건너뛰기로 바꿨는데 예전 라벨 기록이 남아 있다. 그 뒤 다시 그리기도 한다.
# [기대] 건너뛰기가 최신이면 빠지고, 그 뒤 다시 그린 라벨이 있으면 그것을 쓴다.
def test_주석을_지우고_Skip으로_바꾸면_남은_옛_내보내기가_있어도_그_사진은_빠진다():
    stale = ann(939, polygon("dent", TRI), updated="2026-11-08T12:00:00Z")
    skip = ann(1400, cancelled=True, updated="2026-11-08T12:50:00Z")
    doc = {"id": 1152, "data": {"image": "s3://trace-raw/MB09/MB09_SB_L1_CAM1_01.jpg"}}
    assert collect([{**doc, "annotations": [stale, skip]}]) == []
    # Skip 뒤에 다시 그린 주석이 있으면 그것을 쓴다
    redo = ann(1401, polygon("scratch", TRI), updated="2026-11-08T13:00:00Z")
    assert collect([{**doc, "annotations": [stale, skip, redo]}])[0].annotation_id == 1401


# [상황] 예전 Label Studio 형식으로 내보낸 라벨.
# [기대] 작업 번호·라벨 번호·기기·종류를 제대로 읽는다.
def test_옛_주석_형식_내보내기도_읽는다():
    doc = {"id": 9, "result": [polygon("crack", TRI)], "was_cancelled": False,
           "task": {"id": 4, "data": {"image": "s3://trace-raw/REG01/REG01_SL_L2_CAM1_01.jpg"}}}
    [img] = collect([doc])
    assert (img.task_id, img.annotation_id, img.device, img.polygons[0].cls) == (4, 9, "REG01", 2)


# [상황] v001을 만든 뒤 기기를 추가하고 같은 v001 이름으로 다시 만든다.
# [기대] "이미 있는 버전"으로 거절되고 기존 v001 파일은 하나도 바뀌지 않는다(한 번 만든 데이터셋은 고정).
def test_v001이_이미_있으면_v001로_다시_만들기를_거절하고_기존_객체는_그대로다():
    w = three_devices()
    _build(w)
    before = dict(w.s3.objects)
    w.labeled("REG04", 1)
    with pytest.raises(VersionExists):
        _build(w, version="v001")
    assert {k: v for k, v in w.s3.objects.items() if k[0] == "trace-datasets"} == \
        {k: v for k, v in before.items() if k[0] == "trace-datasets"}


# [상황] 기록부에는 없지만 저장소에 v001 파일이 남아 있다.
# [기대] 덮어쓰지 않고 "이미 있음"으로 멈춘다.
def test_버킷에만_남은_버전도_덮어쓰지_않는다():
    w = three_devices()
    w.s3.put_object(Bucket="trace-datasets", Key="v001/manifest.json", Body=b"{}")
    with pytest.raises(AlreadyExists):
        _build(w, version="v001")


# [상황] 같은 위치에 파일을 두 번 쓴다.
# [기대] 두 번째는 "이미 있음"으로 멈추고 처음 내용이 남는다.
def test_조건부_쓰기가_거절되면_이미_있음으로_멈춘다():
    w = World()
    w.store.put_new("trace-datasets", "x", b"1")
    with pytest.raises(AlreadyExists):
        w.store.put_new("trace-datasets", "x", b"2")
    assert w.s3.objects[("trace-datasets", "x")][0] == b"1"


# [상황] 버전 이름 없이 두 번째로 만든다. 잘못된 이름("2")도 준다.
# [기대] v002로 만들어지고(v009 다음은 v010), 잘못된 이름은 거절된다.
def test_버전을_정하지_않으면_다음_번호로_만든다():
    w = three_devices()
    _build(w)
    assert _build(w).version == "v002"
    assert next_version(["v001", "v009", "x"]) == "v010"
    with pytest.raises(BuildError):
        _build(w, version="2")


# [상황] 아무것도 그리지 않은 제출, "흠집 없음"과 흠집을 동시에 표시한 제출이 섞여 있다.
# [기대] 문제 있는 작업 번호를 알려 주며 실패하고, 저장소·기록부에 아무것도 쓰지 않는다.
def test_빈_제출이_있으면_작업_번호를_알리며_실패하고_아무것도_쓰지_않는다():
    w = World()
    w.labeled("REG01", 2)
    tid = w.photo("REG01/REG01_BOT_L1_CAM1_01.jpg", ann(50))
    both = w.photo("REG01/REG01_SF_L1_CAM1_01.jpg", ann(51, NO_DEFECT, polygon("scratch", TRI)))
    with pytest.raises(AnnotationErrors) as exc:
        _build(w)
    text = " ".join(exc.value.problems)
    assert f"작업 {tid}" in text and f"작업 {both}" in text
    assert w.s3.keys("trace-datasets") == [] and w.registry.versions() == []


# [상황] 약속에 없는 종류(rust), 점이 2개뿐인 다각형, 파일 이름 규칙에 안 맞는 사진.
# [기대] 모두 빌드를 멈춘다.
@pytest.mark.parametrize("result, image", [
    ([polygon("rust", TRI)], "REG01/REG01_TOP_L1_CAM1_01.jpg"),
    ([polygon("scratch", TRI[:2])], "REG01/REG01_TOP_L1_CAM1_01.jpg"),
    ([NO_DEFECT], "REG01/slate/REG01_SLATE.jpg"),
])
def test_codes_밖_클래스_점_2개_파일명_규칙_위반은_빌드를_멈춘다(result, image):
    doc = {"id": 1, "data": {"image": f"s3://trace-raw/{image}"}, "annotations": [ann(1, *result)]}
    with pytest.raises(AnnotationErrors):
        collect([doc])


# [상황] 라벨이 하나도 없거나, 라벨 파일이 JSON이 아니다.
# [기대] 시작하지 않는다.
def test_주석이_없거나_JSON이_아니면_시작하지_않는다():
    w = World()
    with pytest.raises(BuildError):
        _build(w)
    w.s3.put_object(Bucket="trace-labels", Key="1", Body=b"not json")
    with pytest.raises(AnnotationErrors):
        _build(w)


# [상황] 빌더 코드에 커밋하지 않은 수정이 있다.
# [기대] 멈춘다. 허용 옵션을 주면 만들되 manifest에 "수정 있음"을 기록한다(어떤 코드로 만들었는지 추적).
def test_커밋하지_않은_변경이_있으면_allow_dirty_없이는_멈춘다():
    w = three_devices()
    with pytest.raises(BuildError):
        _build(w, commit=Commit("b" * 40, True))
    _build(w, commit=Commit("b" * 40, True), allow_dirty=True)
    assert _manifest(w)["builderDirty"] is True


# [상황] 칸이 정해지지 않은 기기 7대를 자동 배정한다.
# [기대] 학습 5·검증 1·시험 1(약 70:15:15)로 나뉘고, 다시 돌려도 똑같다. 새 기기의 기종은 "미등록".
def test_자동_배정은_기기_수_기준_70_15_15에_가장_모자란_쪽이고_결정적이다():
    table, new = assign({f"G{i:02d}" for i in range(1, 8)}, {})
    assert list(table.values()).count("TRAIN") == 5
    assert list(table.values()).count("VAL") == 1 and list(table.values()).count("TEST") == 1
    assert assign({f"G{i:02d}" for i in range(1, 8)}, {})[0] == table
    assert all(d.model_name == "미등록" for d in new)


# [상황] v001에서 시험용으로 쓴 기기를 학습용으로 바꾸려 한다.
# [기대] 거절된다(시험 데이터가 학습에 새면 점수를 믿을 수 없음). 새 기기 추가나 기종 이름 고치기는 된다.
def test_v001에_TEST로_쓰인_기기를_TRAIN으로_바꾸려_하면_거절된다():
    w = three_devices()
    _build(w)
    known = w.registry.devices()
    with pytest.raises(SplitError):
        check_change(Device("REG03", "MacBook Air", "TRAIN"), known, w.registry.used_devices())
    check_change(Device("REG09", "MacBook Air", "TRAIN"), known, w.registry.used_devices())
    check_change(Device("REG03", "MacBook Air M2", "TEST"), known, w.registry.used_devices())


# [상황] 명령줄로 기기 추가 → 데이터셋 만들기 → 목록 보기, 그리고 쓰인 기기의 칸 바꾸기를 한다.
# [기대] 만든 결과(칸별 사진 수·종류별 다각형 수)가 찍히고, 칸 바꾸기는 "분할을 바꿀 수 없음"으로 1을 돌려준다.
#        출력 어디에도 비밀 키가 없다.
def test_명령줄_device_add_build_list가_돌고_분할_변경_거절은_1로_끝난다(capsys):
    w = World()
    w.labeled("REG01", 2)
    deps = Deps({}, registry=lambda: w.registry, store=lambda: w.store, commit=lambda: CLEAN)
    env = {"MINIO_ENDPOINT": "http://minio.test", "MINIO_ACCESS_KEY": "ak", "MINIO_SECRET_KEY": "sk"}
    deps.env = env
    assert main(["dataset", "device", "add", "REG01", "--split", "VAL", "--model", "MacBook Air"], deps=deps) == 0
    assert main(["dataset", "build"], deps=deps) == 0
    assert main(["dataset", "list"], deps=deps) == 0
    assert main(["dataset", "device", "list"], deps=deps) == 0
    assert main(["dataset", "device", "add", "REG01", "--split", "TRAIN"], deps=deps) == 1
    assert main(["dataset", "device", "add", "REG05"], deps=deps) == 1
    out = capsys.readouterr()
    assert "v001 만듦: train 0 · val 2 · test 0" in out.out
    # 라벨 수정 뒤 Label Studio 저장 수와 대조할 클래스별 폴리곤 수
    assert "폴리곤 " in out.out and "흠집 없는 사진 " in out.out
    assert "REG01  VAL    MacBook Air" in out.out
    assert "분할을 바꿀 수 없음" in out.err
    assert "sk" not in out.out + out.err.replace("REG", "")


# [상황] 저장소 설정 없이 데이터셋 만들기를 실행한다.
# [기대] 빠진 설정 이름(MINIO_ENDPOINT)만 알리고 2로 끝난다.
def test_명령줄_설정이_빠지면_이름만_알리고_2로_끝난다(capsys):
    w = World()
    w.labeled("REG01", 1)
    deps = Deps({}, registry=lambda: w.registry, store=lambda: w.store, commit=lambda: CLEAN)
    assert main(["dataset", "build"], deps=deps) == 2
    assert "MINIO_ENDPOINT" in capsys.readouterr().err


# [상황] 기기가 모두 학습용이라 검증 사진이 없다.
# [기대] "val 사진이 없음"으로 멈추고 아무것도 쓰지 않는다. 허용 옵션을 주면 만든다.
def test_명령줄_build는_val_기기가_없으면_아무것도_쓰지_않고_멈춘다(capsys):
    w = World()
    w.labeled("REG01", 2)
    deps = Deps({"MINIO_ENDPOINT": "http://minio.test", "MINIO_ACCESS_KEY": "ak", "MINIO_SECRET_KEY": "sk"},
                registry=lambda: w.registry, store=lambda: w.store, commit=lambda: CLEAN)
    assert main(["dataset", "device", "add", "REG01", "--split", "TRAIN", "--model", "MacBook Air"], deps=deps) == 0
    assert main(["dataset", "build"], deps=deps) == 1
    assert "val 사진이 없음" in capsys.readouterr().err
    assert w.s3.keys("trace-datasets") == [] and w.registry.versions() == []
    assert main(["dataset", "build", "--allow-no-val"], deps=deps) == 0
