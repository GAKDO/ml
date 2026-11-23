# 이 파일이 확인하는 것: 폰으로 찍은 사진의 흠집 탐지(서버 쪽 YOLO). 부스와 같은 조각 나누기·중복 제거를 쓰고,
# 모델 설명 파일(model.json)을 검사하며, 모델 파일이 없으면 탐지만 건너뛴다(11/2 결정).

"""폰 흠집 탐지: 부스와 같은 타일링·중복 제거, model.json 검사, 가중치 없으면 건너뜀 (11/2 결정, A 담당)."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from trace_ml.inspect.detect import NullDetector
from trace_ml.inspect.yolo_detect import ModelInfo, RawDetection, TiledDetector, find_weights, load_detector, tiles
from trace_ml.labelstudio.config import ConfigError

INFO = ModelInfo("yolo11s-seg-v001", ("scratch", "dent", "crack", "stain"))


# [상황] 1600×1200 사진을 조각으로 나눈다.
# [기대] 사진 전체가 덮이고 이웃 조각끼리 충분히 겹친다(경계에 걸친 흠집도 놓치지 않게).
def test_타일은_사진_전체를_덮고_이웃끼리_20퍼센트_이상_겹친다() -> None:
    ts = tiles(1600, 1200)
    covered = np.zeros((1200, 1600), bool)
    for x, y, w, h in ts:
        covered[y:y + h, x:x + w] = True
    assert covered.all()
    xs = sorted({x for x, *_ in ts})
    assert all(b - a <= 512 for a, b in itertools.pairwise(xs))


# [상황] 겹친 두 조각에서 같은 긁힘을 찾았고, 모델이 모르는 종류 번호(9)도 냈다.
# [기대] 모르는 종류는 버리고, 같은 긁힘은 신뢰도 높은 것 하나만 남기며, 위치는 원본 사진 좌표로 바뀐다.
def test_타일_좌표는_원본_px로_옮기고_겹친_타일의_같은_흠집은_하나만_남긴다() -> None:
    def infer(tile: np.ndarray) -> list[RawDetection]:
        # 모든 타일에서 원본 (700, 100)~(760, 110) 근처 긁힘을 찾는 척: 타일 원점을 tile 값으로 표시해 둔다
        ox, oy = int(tile[0, 0, 0]) * 8, int(tile[0, 0, 1]) * 8
        if not (ox <= 700 and 760 <= ox + tile.shape[1] and oy <= 100 and 110 <= oy + tile.shape[0]):
            return []
        poly = [(700 - ox, 100 - oy), (760 - ox, 100 - oy), (760 - ox, 110 - oy), (700 - ox, 110 - oy)]
        return [RawDetection(0, 0.9 if ox == 512 else 0.8, poly), RawDetection(9, 0.99, poly)]

    image = np.zeros((900, 1300, 3), np.uint8)
    for x, y, w, h in tiles(1300, 900):
        image[y, x] = (x // 8, y // 8, 0)  # 타일 왼쪽 위 픽셀에 원점 표시
    found = TiledDetector(infer, INFO).detect(image)
    assert len(found) == 1  # 없는 클래스 번호(9)는 버리고, 두 타일의 같은 흠집은 하나로
    assert found[0].cls == "scratch" and found[0].confidence == 0.9
    assert found[0].polygon[0] == [700.0, 100.0]
    assert TiledDetector(infer, INFO).model_version == "yolo11s-seg-v001"


# 테스트 도구: 가짜 모델 파일(best.pt)과 model.json을 임시 폴더에 만든다.
def _weights(tmp_path: Path, model: dict) -> Path:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")
    (tmp_path / "model.json").write_text(json.dumps(model), encoding="utf-8")
    return weights


# [상황] 모델 설정 없이 탐지기를 만든다.
# [기대] 아무것도 찾지 않는 "빈 탐지기"가 된다(워커는 정상으로 돈다).
def test_가중치가_없으면_폰_탐지를_건너뛴다() -> None:
    assert isinstance(load_detector({}), NullDetector)


# [상황] 모델 파일과 model.json을 준다.
# [기대] 그 모델 버전으로 탐지한다.
def test_가중치와_model_json이_있으면_그_모델로_탐지한다(tmp_path: Path) -> None:
    weights = _weights(tmp_path, {"modelVersion": "yolo11s-seg-v001", "classes": list(INFO.classes)})
    detector = load_detector({"TRACE_DETECT_WEIGHTS": str(weights)}, make_infer=lambda p: lambda tile: [])
    assert detector.model_version == "yolo11s-seg-v001"
    assert detector.detect(np.zeros((100, 100, 3), np.uint8)) == []


# [상황] model.json에 버전이 없거나, 흠집 종류가 codes.yaml과 다르다.
# [기대] 설정 오류로 시작하지 않는다(엉뚱한 종류 이름으로 결과를 내지 않게).
@pytest.mark.parametrize("model", [{"classes": ["scratch", "dent", "crack", "stain"]},
                                   {"modelVersion": "v", "classes": ["scratch", "dent"]}])
def test_model_json에_버전이_없거나_클래스가_codes와_다르면_시작하지_않는다(tmp_path: Path, model: dict) -> None:
    weights = _weights(tmp_path, model)
    with pytest.raises(ConfigError):
        load_detector({"TRACE_DETECT_WEIGHTS": str(weights)}, make_infer=lambda p: lambda tile: [])


# [상황] 없는 모델 파일 경로를 준다.
# [기대] 설정 오류로 시작하지 않는다.
def test_가중치_경로가_틀리면_시작하지_않는다(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_detector({"TRACE_DETECT_WEIGHTS": str(tmp_path / "none.pt")})


# [상황] 모델 폴더에 best.onnx와 model.json을 넣는다.
# [기대] 그 ONNX 파일로 탐지하고 모델 버전도 model.json 값을 쓴다.
def test_모델_폴더에_best_onnx와_model_json을_넣으면_그_모델로_탐지한다(tmp_path: Path) -> None:
    (tmp_path / "best.onnx").write_bytes(b"x")
    (tmp_path / "model.json").write_text(json.dumps({"modelVersion": "best-v001", "classes": list(INFO.classes)}),
                                         encoding="utf-8")
    seen: list[Path] = []
    detector = load_detector({"TRACE_MODEL_DIR": str(tmp_path)},
                             make_infer=lambda p: seen.append(p) or (lambda tile: []))
    assert detector.model_version == "best-v001"
    assert seen == [tmp_path / "best.onnx"]


# [상황] 모델 폴더가 비어 있다.
# [기대] 탐지만 건너뛰고 워커는 시작한다.
def test_모델_폴더가_비어_있으면_탐지만_건너뛰고_워커는_시작한다(tmp_path: Path) -> None:
    assert isinstance(load_detector({"TRACE_MODEL_DIR": str(tmp_path)}), NullDetector)


# [상황] 학습 라이브러리(ultralytics)가 없는 서버 컨테이너에 best.pt만 있다.
# [기대] 쓸 수 없으니 탐지를 건너뛰고 시작한다. 라이브러리가 있으면 best.pt를 쓴다.
def test_ultralytics가_없는_컨테이너는_best_pt만_있으면_탐지를_건너뛰고_시작한다(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "best.pt").write_bytes(b"pt")
    monkeypatch.setattr("trace_ml.inspect.yolo_detect._has_ultralytics", lambda: False)
    assert find_weights({"TRACE_MODEL_DIR": str(tmp_path)}) is None
    assert isinstance(load_detector({"TRACE_MODEL_DIR": str(tmp_path)}), NullDetector)
    monkeypatch.setattr("trace_ml.inspect.yolo_detect._has_ultralytics", lambda: True)
    assert find_weights({"TRACE_MODEL_DIR": str(tmp_path)}) == tmp_path / "best.pt"


# [상황] 탐지 신뢰도 문턱을 "abc"나 1.5처럼 0~1 밖으로 준다.
# [기대] 설정 오류로 시작하지 않는다.
@pytest.mark.parametrize("conf", ["abc", "1.5"])
def test_탐지_문턱이_0과_1_사이가_아니면_시작하지_않는다(tmp_path: Path, conf: str) -> None:
    weights = _weights(tmp_path, {"modelVersion": "v", "classes": list(INFO.classes)})
    with pytest.raises(ConfigError):
        load_detector({"TRACE_DETECT_WEIGHTS": str(weights), "TRACE_DETECT_CONF": conf})


# [상황] ONNX 모델의 원시 출력(상자 하나는 신뢰도 0.9, 하나는 0.1)을 해석한다.
# [기대] 문턱 아래는 버리고, 남은 하나는 찌그러짐(dent) 0.9로, 위치는 조각 좌표의 다각형으로 바뀐다.
def test_ONNX_출력은_부스와_같은_후처리로_타일_px_폴리곤이_된다() -> None:
    from trace_ml.inspect.onnx_seg import decode, letterbox

    # 타일 320×320 → 입력 640(2배, 여백 없음). 입력 px (100,100)~(300,200) 상자 하나, 클래스 1(dent)
    blob, box = letterbox(np.zeros((320, 320, 3), np.uint8), 640)
    assert blob.shape == (1, 3, 640, 640) and box.scale == 2.0
    preds = np.zeros((1, 4 + 4 + 32, 8400), np.float32)
    preds[0, :4, 0] = [200, 150, 200, 100]
    preds[0, 5, 0] = 0.9
    preds[0, 8, 0] = 1.0  # 마스크 계수 첫 번째만 1
    preds[0, 5, 1] = 0.1  # 문턱 아래는 버린다
    protos = np.ones((1, 32, 160, 160), np.float32)
    found = decode(preds, protos, box, min_confidence=0.25)
    assert len(found) == 1
    assert found[0].class_index == 1 and found[0].confidence == pytest.approx(0.9)
    xs = [p[0] for p in found[0].polygon]
    ys = [p[1] for p in found[0].polygon]
    assert min(xs) == pytest.approx(50, abs=1) and max(xs) == pytest.approx(150, abs=1)
    assert min(ys) == pytest.approx(50, abs=1) and max(ys) == pytest.approx(100, abs=1)
