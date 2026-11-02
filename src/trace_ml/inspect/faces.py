"""면별 표준 좌표 크기(contracts capture/faces.md). 1 px = 0.25 mm."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..labelstudio.config import ConfigError

PX_PER_MM = 4.0
FACES: tuple[str, ...] = ("TOP", "BOT", "SF", "SB", "SL", "SR", "KB", "SCR0", "SCR1")


@dataclass(frozen=True)
class DeviceDims:
    """기종 치수(mm). rental.device_model의 width·depth·base_h·lid_h와 같은 뜻."""

    code: str
    width: float
    depth: float
    base_h: float
    lid_h: float


# rental-server V2 시드와 같은 값(decisions.md 11/2: 폭·깊이는 Apple 사양, 두께는 B 실측).
# 이벤트에 기종이 실리기 전까지 워커가 쓰는 표. 기종을 더하면 rental 시드와 함께 고친다.
DEVICE_MODELS: dict[str, DeviceDims] = {
    "MBA13-M2": DeviceDims("MBA13-M2", width=304.1, depth=215.0, base_h=7.0, lid_h=4.0),
}


def face_size_mm(face: str, dims: DeviceDims) -> tuple[float, float]:
    """면의 (가로, 세로) mm. faces.md "크기" 칸."""
    side_h = dims.base_h + dims.lid_h
    if face in ("TOP", "BOT", "KB", "SCR0", "SCR1"):
        return dims.width, dims.depth
    if face in ("SF", "SB"):
        return dims.width, side_h
    if face in ("SL", "SR"):
        return dims.depth, side_h
    raise ValueError(f"모르는 면: {face}")


def face_size_px(face: str, dims: DeviceDims) -> tuple[int, int]:
    """펼친 이미지 (가로, 세로) px."""
    w, h = face_size_mm(face, dims)
    return round(w * PX_PER_MM), round(h * PX_PER_MM)


def load_dims(env: Mapping[str, str]) -> DeviceDims:
    code = env.get("TRACE_DEVICE_MODEL") or "MBA13-M2"
    if code not in DEVICE_MODELS:
        raise ConfigError(f"TRACE_DEVICE_MODEL={code}: 워커 기종 표에 없음 ({', '.join(DEVICE_MODELS)})")
    return DEVICE_MODELS[code]
