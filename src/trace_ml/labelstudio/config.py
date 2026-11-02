"""라벨 설정 XML, trace-raw 키 필터, 환경 변수 설정."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from ..codes import BOOTH_CAMERAS, BOOTH_FACES, BOOTH_LIGHTS, DEFECT_CLASSES

PROJECT_TITLE = "TRACE 흠집 라벨링"
NO_DEFECT = "no_defect"

# {기기ID}/{기기ID}_{면}_{조명}_{카메라}_{번호}.jpg 만 가져온다. {기기ID}/slate/ 아래는 걸러진다.
KEY_REGEX = (
    r"^[A-Z]{2,4}[0-9]{2}/[A-Z]{2,4}[0-9]{2}"
    f"_({'|'.join(BOOTH_FACES)})_({'|'.join(BOOTH_LIGHTS)})_({'|'.join(BOOTH_CAMERAS)})"
    r"_[0-9]{2}\.jpg$"
)

_COLORS = ("#e6194b", "#f58231", "#4363d8", "#911eb4")


def label_config() -> str:
    """폴리곤 4클래스 + '흠집 없음' 체크. 클래스 값은 codes.yaml 그대로 쓴다."""
    labels = "\n".join(
        f'    <Label value="{name}" background="{color}"/>' for name, color in zip(DEFECT_CLASSES, _COLORS, strict=True)
    )
    return (
        "<View>\n"
        '  <Header value="흠집마다 폴리곤을 그리고, 흠집이 없으면 no_defect를 체크하세요"/>\n'
        '  <Image name="image" value="$image" zoom="true"/>\n'
        '  <PolygonLabels name="defect" toName="image" strokeWidth="2" pointSize="small">\n'
        f"{labels}\n"
        "  </PolygonLabels>\n"
        f'  <Choices name="{NO_DEFECT}" toName="image" choice="single">\n'
        f'    <Choice value="{NO_DEFECT}" hint="흠집 없음"/>\n'
        "  </Choices>\n"
        "</View>"
    )


def same_config(a: str, b: str) -> bool:
    """태그 사이 공백만 다른 설정은 같다고 본다."""

    def norm(s: str) -> str:
        return re.sub(r">\s+<", "><", s.strip())

    return norm(a) == norm(b)


class ConfigError(Exception):
    """환경 변수가 빠졌거나 잘못됨."""


@dataclass(frozen=True)
class Settings:
    url: str
    minio_endpoint: str
    raw_bucket: str
    labels_bucket: str
    project_title: str = PROJECT_TITLE
    # 비밀 값은 repr에 나오지 않게 한다
    api_key: str = field(default="", repr=False)
    minio_access_key: str = field(default="", repr=False)
    minio_secret_key: str = field(default="", repr=False)


def load_settings(env: Mapping[str, str]) -> Settings:
    """환경 변수에서 설정을 읽는다. 빠진 이름만 알리고 값은 보여 주지 않는다."""

    def get(*names: str) -> str:
        for name in names:
            if env.get(name):
                return env[name]
        return ""

    values = {
        "LABEL_STUDIO_URL": get("LABEL_STUDIO_URL"),
        "LABEL_STUDIO_API_KEY": get("LABEL_STUDIO_API_KEY"),
        "MINIO_ENDPOINT": get("MINIO_ENDPOINT"),
        "LABEL_STUDIO_MINIO_ACCESS_KEY": get("LABEL_STUDIO_MINIO_ACCESS_KEY", "MINIO_ACCESS_KEY"),
        "LABEL_STUDIO_MINIO_SECRET_KEY": get("LABEL_STUDIO_MINIO_SECRET_KEY", "MINIO_SECRET_KEY"),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ConfigError("환경 변수가 없음: " + ", ".join(missing))
    settings = Settings(
        url=values["LABEL_STUDIO_URL"].rstrip("/"),
        minio_endpoint=values["MINIO_ENDPOINT"],
        raw_bucket=get("TRACE_RAW_BUCKET") or "trace-raw",
        labels_bucket=get("TRACE_LABELS_BUCKET") or "trace-labels",
        project_title=get("LABEL_STUDIO_PROJECT_TITLE") or PROJECT_TITLE,
        api_key=values["LABEL_STUDIO_API_KEY"],
        minio_access_key=values["LABEL_STUDIO_MINIO_ACCESS_KEY"],
        minio_secret_key=values["LABEL_STUDIO_MINIO_SECRET_KEY"],
    )
    if settings.raw_bucket == settings.labels_bucket:
        raise ConfigError(f"원본 버킷과 주석 버킷이 같음({settings.raw_bucket}): 원본을 덮어쓸 수 있어 시작하지 않음")
    return settings
