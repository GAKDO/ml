"""폰 흠집 탐지: 부스와 같은 타일링·중복 제거, model.json 검사, 가중치 없으면 건너뜀 (11/2 결정, A 담당)."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from trace_ml.inspect.detect import NullDetector
from trace_ml.inspect.yolo_detect import ModelInfo, RawDetection, TiledDetector, load_detector, tiles
from trace_ml.labelstudio.config import ConfigError

INFO = ModelInfo("yolo11s-seg-v001", ("scratch", "dent", "crack", "stain"))


def test_타일은_사진_전체를_덮고_이웃끼리_20퍼센트_이상_겹친다() -> None:
    ts = tiles(1600, 1200)
    covered = np.zeros((1200, 1600), bool)
    for x, y, w, h in ts:
        covered[y:y + h, x:x + w] = True
    assert covered.all()
    xs = sorted({x for x, *_ in ts})
    assert all(b - a <= 512 for a, b in itertools.pairwise(xs))


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


def _weights(tmp_path: Path, model: dict) -> Path:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")
    (tmp_path / "model.json").write_text(json.dumps(model), encoding="utf-8")
    return weights


def test_가중치가_없으면_폰_탐지를_건너뛴다() -> None:
    assert isinstance(load_detector({}), NullDetector)


def test_가중치와_model_json이_있으면_그_모델로_탐지한다(tmp_path: Path) -> None:
    weights = _weights(tmp_path, {"modelVersion": "yolo11s-seg-v001", "classes": list(INFO.classes)})
    detector = load_detector({"TRACE_DETECT_WEIGHTS": str(weights)}, make_infer=lambda p: lambda tile: [])
    assert detector.model_version == "yolo11s-seg-v001"
    assert detector.detect(np.zeros((100, 100, 3), np.uint8)) == []


@pytest.mark.parametrize("model", [{"classes": ["scratch", "dent", "crack", "stain"]},
                                   {"modelVersion": "v", "classes": ["scratch", "dent"]}])
def test_model_json에_버전이_없거나_클래스가_codes와_다르면_시작하지_않는다(tmp_path: Path, model: dict) -> None:
    weights = _weights(tmp_path, model)
    with pytest.raises(ConfigError):
        load_detector({"TRACE_DETECT_WEIGHTS": str(weights)}, make_infer=lambda p: lambda tile: [])


def test_가중치_경로가_틀리면_시작하지_않는다(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_detector({"TRACE_DETECT_WEIGHTS": str(tmp_path / "none.pt")})
