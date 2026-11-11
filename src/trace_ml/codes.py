"""contracts `codes.yaml` 0.5.0 값(이 레포가 쓰는 값은 0.3.0 이후 그대로). 이 레포의 코드 값은 여기에만 둔다."""

CODES_VERSION = "0.5.0"

# inspection.defect_class: 긁힘·찍힘·균열·오염
DEFECT_CLASSES: tuple[str, ...] = ("scratch", "dent", "crack", "stain")

# 부스 촬영 파일명에 들어가는 값 (booth contracts/storage-layout.md)
BOOTH_FACES: tuple[str, ...] = ("TOP", "BOT", "SF", "SB", "SL", "SR", "KB", "SCR0", "SCR1")
BOOTH_LIGHTS: tuple[str, ...] = ("L1", "L2", "L3", "L4", "OFF")
BOOTH_CAMERAS: tuple[str, ...] = ("CAM1", "CAM2")
