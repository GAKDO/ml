"""YOLO11-seg ONNX를 onnxruntime(CPU)로 돌리는 폰 탐지(학교 서버 inspect-worker 컨테이너용, torch 없음).

후처리(레터박스·클래스별 NMS·상자 안쪽 마스크 → 폴리곤)는 부스 booth/src/trace_booth/detect/yolo_seg.py와 같은 식이다.
부스(Jetson TensorRT)와 같은 best.onnx·model.json을 그대로 쓰므로 모델을 바꿀 때 두 곳에 같은 파일만 넣으면 된다.

ultralytics 기본 ONNX 내보내기(nms·end2end 끔) 기준:
- 입력 `images` (1, 3, 640, 640) RGB 0~1, 비율 유지 + 회색(114) 여백
- `output0` (1, 4 + 클래스 수 + 32, 8400): cx·cy·w·h(입력 px), 클래스 점수, 마스크 계수
- `output1` (1, 32, 160, 160): 마스크 원형
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..labelstudio.config import ConfigError
from .yolo_detect import Infer, RawDetection

MASK_COEFFS = 32
MAX_DETECTIONS = 100


@dataclass(frozen=True)
class Letterbox:
    scale: float  # 타일 px × scale = 입력 px
    pad_x: float
    pad_y: float
    width: int  # 원래 타일 크기
    height: int


def letterbox(tile: Any, size: int = 640) -> tuple[np.ndarray, Letterbox]:
    """BGR(또는 회색) 타일 → (1, 3, size, size) float32 RGB 0~1. 비율을 지키고 남는 곳은 114로 채운다."""
    import cv2

    image = np.asarray(tile)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_w, new_h = round(width * scale), round(height * scale)
    if (new_w, new_h) != (width, height):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_w) / 2, (size - new_h) / 2
    top, left = round(pad_y - 0.1), round(pad_x - 0.1)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = image
    blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(blob), Letterbox(scale, float(left), float(top), width, height)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    """xyxy 상자의 탐욕 NMS. 점수 높은 순으로 남긴 번호."""
    order = scores.argsort()[::-1]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        ix = np.clip(np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0]), 0, None)
        iy = np.clip(np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1]), 0, None)
        inter = ix * iy
        overlap = inter / np.maximum(area[i] + area[rest] - inter, 1e-9)
        order = rest[overlap < iou]
    return keep


def _polygon(mask: np.ndarray) -> np.ndarray | None:
    """이진 마스크에서 가장 큰 바깥 윤곽. 점이 3개 안 되면 None."""
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if len(biggest) < 3:
        return None
    return biggest.reshape(-1, 2).astype(np.float32)


def decode(output0: np.ndarray, output1: np.ndarray, box: Letterbox, min_confidence: float = 0.25,
           iou: float = 0.45, size: int = 640) -> list[RawDetection]:
    """엔진 출력 → 타일 px 폴리곤. 같은 클래스끼리만 NMS(ultralytics 기본과 같음)."""
    import cv2

    preds = np.asarray(output0, dtype=np.float32)[0].T  # (8400, 4 + nc + 32)
    protos = np.asarray(output1, dtype=np.float32)[0]  # (32, 160, 160)
    num_classes = preds.shape[1] - 4 - MASK_COEFFS
    if num_classes < 1:
        raise ValueError(f"YOLO-seg 출력이 아님: output0 {output0.shape}")
    class_scores = preds[:, 4:4 + num_classes]
    cls = class_scores.argmax(1)
    conf = class_scores[np.arange(len(cls)), cls]
    picked = conf >= min_confidence
    if not picked.any():
        return []
    preds, cls, conf = preds[picked], cls[picked], conf[picked]
    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    offset = cls[:, None].astype(np.float32) * (size * 4)  # 클래스마다 떨어뜨려 클래스별 NMS
    keep = nms(boxes + offset, conf, iou)[:MAX_DETECTIONS]

    # 마스크는 상자 안쪽만 계산한다(11/6 추론 시간 줄이기). 예전에는 잡힌 것마다 160×160 마스크 전체를 640×640으로
    # 늘려 상자 밖을 버렸는데, 덜 학습된 모델이 상자를 많이 내면 이 후처리가 엔진 실행보다 오래 걸렸다.
    proto_h, proto_w = protos.shape[1:]
    flat = protos.reshape(MASK_COEFFS, proto_h, proto_w)
    ratio = proto_w / size  # 입력 px × ratio = 원형 px
    out: list[RawDetection] = []
    for i in keep:
        x0, y0, x1, y1 = (float(v) for v in np.clip(boxes[i], 0, size))
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        px0, py0 = int(np.floor(x0 * ratio)), int(np.floor(y0 * ratio))
        px1, py1 = min(proto_w, int(np.ceil(x1 * ratio))), min(proto_h, int(np.ceil(y1 * ratio)))
        if px1 <= px0 or py1 <= py0:
            continue
        coeffs = preds[i, 4 + num_classes:]
        crop = flat[:, py0:py1, px0:px1]
        logits = (coeffs @ crop.reshape(MASK_COEFFS, -1)).reshape(py1 - py0, px1 - px0)
        left, top = px0 / ratio, py0 / ratio  # 잘라 낸 조각의 입력 px 왼쪽 위
        width, height = round((px1 - px0) / ratio), round((py1 - py0) / ratio)
        # sigmoid(x) > 0.5 ⇔ x > 0 이라 늘린 로짓을 0으로 자른다(exp 계산 생략)
        up = cv2.resize(logits, (width, height), interpolation=cv2.INTER_LINEAR) > 0
        bx0, by0 = max(0, round(x0 - left)), max(0, round(y0 - top))
        bx1, by1 = min(width, round(x1 - left)), min(height, round(y1 - top))
        mask = np.zeros_like(up)
        mask[by0:by1, bx0:bx1] = up[by0:by1, bx0:bx1]  # 상자 밖 마스크는 버린다
        poly = _polygon(mask)
        if poly is None:
            continue
        px = np.clip((poly[:, 0] + left - box.pad_x) / box.scale, 0, box.width)
        py = np.clip((poly[:, 1] + top - box.pad_y) / box.scale, 0, box.height)
        out.append(RawDetection(int(cls[i]), float(conf[i]),
                                [(round(float(x), 1), round(float(y), 1)) for x, y in zip(px, py)]))
    return out


def onnx_infer(weights: Path, min_confidence: float = 0.25, iou: float = 0.45, threads: int = 0) -> Infer:
    """best.onnx를 onnxruntime CPU로 부르는 타일 추론. threads가 0이면 onnxruntime 기본(코어 수)."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ConfigError("onnxruntime이 설치되지 않음(pip install -e '.[worker]')") from exc
    options = ort.SessionOptions()
    if threads > 0:
        options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(weights), sess_options=options, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    size = int(session.get_inputs()[0].shape[-1])

    def infer(tile: np.ndarray) -> list[RawDetection]:
        blob, box = letterbox(tile, size)
        out0, out1 = sorted(session.run(None, {name: blob}), key=lambda a: a.ndim)
        return decode(out0, out1, box, min_confidence, iou, size)

    return infer
