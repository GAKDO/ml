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
| `trace-ml train <버전> --from-run <폴더>` | 저장소가 멈췄을 때 `yolo`로 따로 학습한 폴더(`weights/best.pt`·`best.onnx`)를 학습 없이 같은 방식으로 기록(지표는 이 버전 val로 다시 잼). 올린 `best.onnx`·`model.json`은 `<폴더>/registered/`에도 남음 | ML1 |
| `trace-ml models list` | 모델 → 데이터셋 버전 | ML1 |
| `trace-ml review sql\|export` | 사람 검수 결과(흠집 맞음=라벨, 흠집 아님=음성 예시)를 YOLO 폴더로. `train --extra-train <폴더>`로 train에만 더함 | ML1 |
| `trace-ml evaluate --weights … --data …` | 기기 면 판정(문제 있음/없음) 정확도·문턱별 표, JSON | ML1 |
| `trace-ml inspect worker` | 큐 `trace.inspect.requested`의 세트를 윤곽 4점으로 펼쳐 `trace-derived`에 올리고 `trace.inspect.completed`(집 PC 자동 시작: `deploy/inspect-worker/`) | ML2 |
| (정합 보정) | `TRACE_ALIGN=features`(기본 `contour`)면 윤곽 결과를 같은 대여의 첫 시점 펼친 이미지에 특징점으로 맞춰 다듬음. 모델 폴더에 `superpoint.onnx`·`lightglue.onnx`가 있으면 SuperPoint+LightGlue, 없으면 ORB | ML2 |
| `trace-ml align photo\|eval\|pair` | 사진 한 장 펼치기, 합성·실제 사진 정합 오차(px·mm) | ML2 |

설정은 환경 변수로 받습니다(`specs/002-dataset-builder/quickstart.md`).

### 사람 검수 결과로 다시 학습하기(11/8)

관리자가 검수 화면에서 확정·기각한 결과를 다음 학습에 씁니다. 부스 빠른 검사(QUICK) 세트만 내보냅니다.

1. 학교 서버 `~/gitops/compose`에서 `trace-ml review sql --since 2026-11-08`이 보여 주는 명령을 돌려 `reviews.csv`를 만들고 집 PC로 가져옵니다(DB 포트는 밖에 열지 않음).
2. 집 PC에서 사진 저장소(gakdo.site Compose MinIO, 버킷 `trace-captures`)에 닿게 SSH 터널을 열고 `CAPTURE_MINIO_ENDPOINT`·`CAPTURE_MINIO_ACCESS_KEY`·`CAPTURE_MINIO_SECRET_KEY`를 env에 둡니다(값은 서버 `.env`, 채팅·Git에 넣지 않음).
3. `trace-ml review export --rows reviews.csv --out D:\trace\cache\review-<날짜>` → `images|labels/train`, 흠집 아님 자리 `hard/train`, `reviews.jsonl`, `summary.json`.
4. `trace-ml train <버전> --tile 640 --extra-train D:\trace\cache\review-<날짜>`. 검수분은 train에만 들어가고, 흠집 아님 자리가 걸친 타일은 빈 타일 고르기에서 빠지지 않습니다. `run.json`의 `extraTrain`에 요약 해시가 남습니다. 기기 시리얼이나 자산번호가 데이터셋 VAL·TEST 기기 코드(예: `MB04`)와 같은 사진은 점수가 부풀지 않게 빼고 `extraTrain.excludedHeldOut`에 적으므로, 관리자 화면에서 기기를 등록할 때 자산번호를 학습 기기 코드로 적어 둡니다.

AI가 아예 못 잡은 흠집은 이 사진들에 라벨이 없으므로 val로는 쓰지 않습니다. 평가는 지금처럼 검증 기기 3대로 합니다.

## 탐지 모델 실험 기록

대회 모델은 **v004**(YOLO11m-seg, AI Hub 사전학습 → 부스 사진 미세조정 + 증강)이고, 11/19 본선까지 바꾸지 않습니다. 모델은 mAP가 아니라 부스가 실제로 내는 판정, 곧 "이 면에 흠집이 있나"로 고릅니다. 자세한 측정 조건과 원자료는 [docs contest/metrics/M3.md](https://github.com/GAKDO/docs/blob/main/contest/metrics/M3.md)에 있습니다.

| 판 | 바꾼 것 | 결과(검증 데이터) | 판단 |
|---|---|---|---|
| AI Hub 사전학습만 | – | 깨끗한 면까지 모두 오탐(기기 면 2/6, v002) | 탈락: 부스 사진으로 미세조정 필요 |
| v001 | 부스 사진으로 2에포크 미세조정 | 기기 면 6/6, 오탐 0(v002) | 첫 대회 모델 |
| 100에포크 | 오래 학습 | 3/6·4/6, 깨끗한 면 오탐(v002) | 탈락: 학습 기기 12대를 외움 |
| v002 `--freeze 10 --lr0 0.0005` | 백본 고정 | 모든 자국을 scratch로 부름(v002) | 탈락 |
| v002 증강 6종 | 회전·밝기·mixup·copy_paste 등 | 흠집은 더 찾지만 깨끗한 면 4개 모두 오탐(v002) | "AI 후보 → 사람 검수" 방식으로 전환 |
| **v004** | 키보드·옆면 라벨, Skip 버그 수정(#32), 증강 | 흠집 면 4/4, 오탐 사진 15(v002 증강 48의 약 1/3), 사진 정밀도 0.71·재현율 0.60(v004) | **대회 모델** |

검증 데이터가 두 가지입니다. v004 이전 줄은 v002 검증 데이터(기기 면 6개), v004 줄과 비교 대상 "v002 증강 48"은 v004 검증 데이터(학습에 쓰지 않은 기기, 인스턴스 164개)로 쟀으므로 두 묶음의 숫자는 서로 바로 비교하지 않습니다. conf는 모두 0.25입니다. 부스 Jetson TensorRT FP16 엔진은 12장을 2.53초에 처리합니다(2호기 실측).

AI는 후보만 넓게 잡고, 관리자가 확정한 흠집만 등급과 봉인 기록에 들어갑니다. 관리자가 확정·기각한 결과와 직접 그린 놓친 흠집은 `trace-ml review export`로 다음 학습에 들어가고, 매일 아침 `retrain auto`가 기준(지난 재학습 이후 검수한 사진 120장, 그중 관리자가 확정한 흠집 20개, 기각한 후보는 세지 않음)을 채웠는지 확인합니다(#36·#37).

## 테스트

```bash
python -m ruff check . && python -m pytest
```

## CI

| 워크플로 | 언제 | 하는 일 |
|---|---|---|
| `python-ci` | PR, develop·main push | ruff + pytest (ubuntu) |
| `비밀값 검사`·`release`·알림 | | 다른 레포와 같음 |
