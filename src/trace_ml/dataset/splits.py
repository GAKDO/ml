"""기기 단위 분할. 한 기기 = 한 분할이고, 버전에 쓰인 뒤에는 바꿀 수 없다."""

from __future__ import annotations

from ..registry import SPLITS, Device

# 자동 배정 목표 비율(기기 수 기준). 테스트 기기는 사람이 먼저 정하는 것을 권한다
TARGET = {"TRAIN": 0.70, "VAL": 0.15, "TEST": 0.15}
UNKNOWN_MODEL = "미등록"


class SplitError(Exception):
    """분할 규칙 위반."""


def assign(device_codes: set[str], known: dict[str, Device]) -> tuple[dict[str, str], list[Device]]:
    """모든 기기의 분할과 새로 배정한 기기 목록. 배정은 기기 코드 순서로 결정적이다."""
    table = {code: d.split for code, d in known.items()}
    counts = {s: sum(1 for v in table.values() if v == s) for s in SPLITS}
    new: list[Device] = []
    for code in sorted(device_codes - table.keys()):
        total = sum(counts.values()) + 1
        # 목표 대비 가장 모자란 분할. 같으면 TRAIN·VAL·TEST 순
        split = max(SPLITS, key=lambda s: (TARGET[s] * total - counts[s], -SPLITS.index(s)))
        table[code] = split
        counts[split] += 1
        new.append(Device(code, UNKNOWN_MODEL, split))
    return {c: table[c] for c in sorted(device_codes)}, new


def check_change(device: Device, known: dict[str, Device], used: set[str]) -> None:
    """버전에 쓰인 기기의 분할을 바꾸면 한 기기가 두 분할에 걸치므로 막는다."""
    if device.split not in SPLITS:
        raise SplitError(f"분할은 {', '.join(SPLITS)} 중 하나")
    before = known.get(device.code)
    if before and before.split != device.split and device.code in used:
        raise SplitError(f"{device.code}는 이미 데이터셋 버전에 {before.split}로 쓰여 분할을 바꿀 수 없음")
