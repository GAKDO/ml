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

| 명령 | 하는 일 | 기능 |
|---|---|---|
| `trace-ml labelstudio setup\|sync` | Label Studio 프로젝트·스토리지 설정, 작업 가져오기·주석 내보내기 | ML3 |
| `trace-ml db init` | PostgreSQL `ml` 표 3개(없을 때만) | ML1 |
| `trace-ml dataset device add\|list` | 기기 → 분할(TRAIN·VAL·TEST) | ML1 |
| `trace-ml dataset build\|list` | 제출된 주석으로 기기 단위 데이터셋 버전(덮어쓰기 없음) | ML1 |
| `trace-ml train <버전>` | YOLO11-seg 학습, 가중치·ONNX·지표를 버전과 함께 `trace-models`·`ml.training_run`에 | ML1 |
| `trace-ml models list` | 모델 → 데이터셋 버전 | ML1 |
| `trace-ml align photo\|eval\|pair` | 사진 한 장 펼치기, 합성·실제 사진 정합 오차(px·mm) | ML2 |

설정은 환경 변수로 받습니다(`specs/002-dataset-builder/quickstart.md`).

## 테스트

```bash
python -m ruff check . && python -m pytest
```

## CI

| 워크플로 | 언제 | 하는 일 |
|---|---|---|
| `python-ci` | PR, develop·main push | ruff + pytest (ubuntu) |
| `비밀값 검사`·`release`·알림 | | 다른 레포와 같음 |
