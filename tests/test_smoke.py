"""이 파일이 확인하는 것: trace-ml 패키지가 설치되어 버전을 알려 주는지(가장 기본적인 동작 확인)."""

from trace_ml import __version__
from trace_ml.__main__ import main


# [상황] 패키지 버전 값을 읽는다.
# [기대] 0.1.0이다.
def test_패키지_버전_문자열을_읽을_수_있다():
    assert __version__ == "0.1.0"


# [상황] 명령줄 도구(trace-ml)를 인자 없이 실행한다.
# [기대] 정상 종료(0)하고 화면에 버전 0.1.0을 찍는다.
def test_main_prints_version(capsys):
    assert main([]) == 0
    assert "0.1.0" in capsys.readouterr().out
