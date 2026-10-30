"""명령줄 진입점. 실제 기능은 이 패키지 아래 모듈로 더한다."""

from . import __version__


def main() -> int:
    print(f"trace-ml {__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
