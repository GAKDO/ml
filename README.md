# ml

TRACE(노트북 렌탈 상태 기록 서비스)의 **ml**: 집 PC 작업: inspect-worker·recon-worker(실험)·데이터셋 빌더·YOLO11-seg 학습

- 담당: A
- 설계: [GAKDO/docs](https://github.com/GAKDO/docs) — 데이터셋·학습 흐름은 design/kickoff-plan.md
- 기술: Python 3.12

## 실행

```bash
python -m pip install -e ".[dev]"
```

```bash
python -m trace_ml
```

## 테스트

```bash
python -m ruff check . && python -m pytest
```

## CI

| 워크플로 | 언제 | 하는 일 |
|---|---|---|
| `python-ci` | PR, develop·main push | ruff + pytest (ubuntu) |
| `비밀값 검사`·`release`·알림 | | 다른 레포와 같음 |
