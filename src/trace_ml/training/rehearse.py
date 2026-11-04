"""사진 없이 학습 파이프라인 끝까지 돌려 보기(리허설).

실제 부스 사진이 오기 전에 집 PC에서 `합성 사진 → 640 타일 → YOLO11-seg 학습 → 고정 640 ONNX → model.json`을
한 번 끝까지 돌려, 실사진이 들어왔을 때 막힐 곳(CUDA·ultralytics·ONNX 내보내기·Jetson 엔진 빌드)을 미리 찾는다.

- MinIO·PostgreSQL·Label Studio를 쓰지 않는다. 결과는 로컬 폴더에만 남는다(trace-models에 올리지 않음).
- model.json은 `testOnly: true`라 부스가 이 모델의 탐지를 서버에 올리지 않는다. 정확도가 아니라 배선 확인용이다.
- 합성 사진 규칙: 어두운 부스 바닥 위 알루미늄 판(노트북 면) + 긁힘(가는 곡선)·찍힘(작은 타원). 라벨은 그린 픽셀의 윤곽이라
  폴리곤과 그림이 정확히 맞는다.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..codes import DEFECT_CLASSES
from .tiles import tile_dataset
from .yolo import TrainOutput, TrainSpec, train_yolo

SCRATCH, DENT = DEFECT_CLASSES.index("scratch"), DEFECT_CLASSES.index("dent")


@dataclass(frozen=True)
class Rehearsal:
    out_dir: Path
    onnx: Path
    model_json: Path
    counts: dict[str, dict[str, int]]
    metrics: dict[str, Any]
    onnx_check: str


def _panel(rng: np.random.Generator, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """어두운 바닥 위 기울어진 알루미늄 판. (사진, 판 마스크)."""
    image = np.full((height, width, 3), int(rng.integers(20, 45)), np.uint8)
    cx, cy = width / 2 + rng.uniform(-0.05, 0.05) * width, height / 2 + rng.uniform(-0.05, 0.05) * height
    pw, ph = width * rng.uniform(0.6, 0.8), height * rng.uniform(0.6, 0.8)
    box = cv2.boxPoints(((cx, cy), (pw, ph), float(rng.uniform(-6, 6)))).astype(np.int32)
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [box], 255)
    # 옆 조명처럼 한쪽이 밝은 기울기 + 은색 기본값
    base = rng.uniform(150, 190)
    ramp = np.linspace(-1, 1, width)[None, :] * rng.uniform(5, 20) + np.linspace(-1, 1, height)[:, None] * 8
    metal = np.clip(base + ramp + rng.normal(0, 2.5, (height, width)), 0, 255).astype(np.uint8)
    image[mask > 0] = np.repeat(metal[..., None], 3, axis=2)[mask > 0]
    return image, mask


def _scratch(rng: np.random.Generator, image: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """가는 곡선 긁힘. 그린 픽셀 마스크를 돌려준다."""
    ys, xs = np.nonzero(inside)
    i = int(rng.integers(len(xs)))
    x, y, angle = float(xs[i]), float(ys[i]), rng.uniform(0, np.pi)
    pts = []
    for _ in range(int(rng.integers(6, 14))):
        pts.append((x, y))
        angle += rng.normal(0, 0.15)
        step = rng.uniform(10, 30)
        x, y = x + np.cos(angle) * step, y + np.sin(angle) * step
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.polylines(mask, [np.array(pts, np.int32)], False, 255, int(rng.integers(3, 7)), cv2.LINE_AA)
    mask = ((mask > 127) & (inside > 0)).astype(np.uint8) * 255
    shade = int(rng.choice([-1, 1]) * rng.integers(35, 70))  # 스침광에서 밝게 또는 어둡게
    image[mask > 0] = np.clip(image[mask > 0].astype(np.int16) + shade, 0, 255).astype(np.uint8)
    return mask


def _dent(rng: np.random.Generator, image: np.ndarray, inside: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(inside)
    i = int(rng.integers(len(xs)))
    axes = (int(rng.integers(10, 26)), int(rng.integers(8, 18)))
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.ellipse(mask, (int(xs[i]), int(ys[i])), axes, float(rng.uniform(0, 180)), 0, 360, 255, -1)
    mask = ((mask > 0) & (inside > 0)).astype(np.uint8) * 255
    shadow = cv2.GaussianBlur(mask, (0, 0), 3).astype(np.float32) / 255
    image[:] = np.clip(image.astype(np.float32) - shadow[..., None] * rng.uniform(40, 70), 0, 255).astype(np.uint8)
    return mask


def _label_lines(cls: int, mask: np.ndarray) -> list[str]:
    """그린 마스크 → YOLO-seg 줄(정규화 폴리곤). 판 가장자리에 잘려 조각이 나면 조각마다 한 줄."""
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in contours:
        if len(c) < 3 or cv2.contourArea(c) < 16:
            continue
        pts = c.reshape(-1, 2)
        lines.append(f"{cls} " + " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in pts))
    return lines


def synth_photo(seed: int, width: int = 1920, height: int = 1080,
                clean: bool = False) -> tuple[np.ndarray, list[str]]:
    """합성 부스 사진 한 장과 YOLO-seg 라벨 줄. clean이면 흠집 없는 사진(빈 라벨)."""
    rng = np.random.default_rng(seed)
    image, panel = _panel(rng, width, height)
    inside = cv2.erode(panel, np.ones((41, 41), np.uint8))
    lines: list[str] = []
    if not clean:
        for _ in range(int(rng.integers(2, 6))):
            lines += _label_lines(SCRATCH, _scratch(rng, image, inside))
        for _ in range(int(rng.integers(0, 3))):
            lines += _label_lines(DENT, _dent(rng, image, inside))
    image = np.clip(image + rng.normal(0, 2.0, image.shape), 0, 255).astype(np.uint8)  # 센서 잡음
    return image, lines


def write_raw_dataset(dest: Path, photos: int, width: int = 1920, height: int = 1080,
                      seed: int = 7) -> dict[str, int]:
    """dest/images|labels/<분할> (dataset build가 받는 것과 같은 모양). 분할마다 흠집 없는 사진을 섞는다."""
    # 합이 photos와 같게: val은 흠집·깨끗한 사진이 하나씩 있게 2장 이상, test 1장 이상, 나머지 train
    val, test = max(2, round(photos * 0.2)), max(1, round(photos * 0.1))
    sizes = {"train": photos - val - test, "val": val, "test": test}
    if sizes["train"] < 1:
        raise ValueError(f"사진 {photos}장으로는 train을 만들 수 없음")
    counts = {}
    n = 0
    for split, k in sizes.items():
        (dest / "images" / split).mkdir(parents=True, exist_ok=True)
        (dest / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(k):
            # 분할마다 첫 장은 흠집 사진, 마지막 한 장은 흠집 없는 사진(오탐 확인용), 그 사이도 다섯 장에 한 장은 깨끗하게
            clean = k > 1 and (i == k - 1 or (i > 0 and n % 5 == 4))
            image, lines = synth_photo(seed + n, width, height, clean=clean)
            stem = f"SYN{n:03d}_TOP_L1_CAM1"
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise RuntimeError("합성 사진을 쓰지 못함")
            (dest / "images" / split / f"{stem}.jpg").write_bytes(buf.tobytes())
            (dest / "labels" / split / f"{stem}.txt").write_text("".join(f"{x}\n" for x in lines), encoding="utf-8")
            n += 1
        counts[split] = k
    return counts


def check_onnx(onnx: Path, size: int, classes: int) -> str:
    """부스(booth `detect/yolo_seg.py`)가 기대하는 모양인지: 입력 1×3×size×size, 출력 (1,4+nc+32,N)·(1,32,size/4,size/4).

    onnxruntime이 없거나 모양이 맞지 않으면 RuntimeError(확인 없이 통과로 보이지 않게).
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError('onnxruntime이 없어 ONNX를 확인하지 못함: python -m pip install -e ".[train]"') from exc
    session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    if list(inp.shape) != [1, 3, size, size]:
        raise RuntimeError(f"ONNX 입력이 고정 1x3x{size}x{size}가 아님: {inp.shape}")
    blob = np.full((1, 3, size, size), 0.5, np.float32)
    outs = sorted(session.run(None, {inp.name: blob}), key=lambda a: a.ndim)
    if len(outs) != 2 or outs[0].shape[1] != 4 + classes + 32 or outs[1].shape[1:] != (32, size // 4, size // 4):
        raise RuntimeError(f"YOLO11-seg 출력 모양이 부스와 다름: {[o.shape for o in outs]}")
    return f"통과(입력 {inp.shape}, 출력 {list(outs[0].shape)}·{list(outs[1].shape)})"


def rehearse(out_dir: Path, *, photos: int = 40, epochs: int = 30, base_weights: str = "yolo11s-seg.pt",
             tile: int = 640, overlap: float = 0.2, batch: int = -1, width: int = 1920, height: int = 1080,
             trainer: Callable[[TrainSpec], TrainOutput] = train_yolo,
             onnx_checker: Callable[[Path, int, int], str] = check_onnx) -> Rehearsal:
    """합성 데이터로 실제 `trace-ml train`과 같은 타일·학습·내보내기 함수를 그대로 부른다."""
    if photos < 10 or epochs < 1 or tile <= 0:
        raise ValueError("photos는 10 이상, epochs는 1 이상, tile은 양수")
    out_dir = out_dir.resolve()  # ultralytics는 상대 경로 project를 runs/segment 아래로 옮긴다
    raw = out_dir / "raw"
    for old in (raw, out_dir / "model"):  # 다시 돌릴 때 옛 합성 사진·결과가 섞이지 않게
        if old.exists():
            shutil.rmtree(old)
    write_raw_dataset(raw, photos, width, height)
    names = dict(enumerate(DEFECT_CLASSES))
    tiled = tile_dataset(raw, out_dir / f"tile{tile}", names, size=tile, overlap=overlap)
    has_test = bool(tiled.counts["test"]["tiles"])
    out = trainer(TrainSpec(tiled.data_yaml, base_weights, epochs, tile, out_dir / "runs", "rehearsal", has_test,
                            batch))
    onnx_check = onnx_checker(out.onnx, tile, len(DEFECT_CLASSES))
    model_dir = out_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    onnx = model_dir / "best.onnx"
    onnx.write_bytes(out.onnx.read_bytes())
    (model_dir / "best.pt").write_bytes(out.weights.read_bytes())
    # 부스 ModelInfo 형식. testOnly라 부스가 서버에 올리지 않는다(진짜 v001은 trace-ml train이 만든다)
    info = {"modelVersion": f"{Path(base_weights).stem}-rehearsal-trt-fp16", "classes": list(DEFECT_CLASSES),
            "testOnly": True, "datasetVersion": "rehearsal", "imgsz": tile, "tileSize": tile, "overlap": overlap}
    model_json = model_dir / "model.json"
    model_json.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    (model_dir / "metrics.json").write_text(json.dumps(out.metrics, indent=2), encoding="utf-8")
    # 부스 bench·눈 확인용 원본 해상도 사진: 흠집 있는 것과 없는 것 하나씩
    for photo in sorted((raw / "images" / "val").glob("*.jpg")):
        label = raw / "labels" / "val" / f"{photo.stem}.txt"
        name = "sample_defect.jpg" if label.read_text(encoding="utf-8").strip() else "sample_clean.jpg"
        if not (model_dir / name).exists():
            (model_dir / name).write_bytes(photo.read_bytes())
    return Rehearsal(model_dir, onnx, model_json, tiled.counts, out.metrics, onnx_check)
