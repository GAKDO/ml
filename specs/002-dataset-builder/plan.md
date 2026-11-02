# 구현 계획: ML1 데이터셋 빌더·학습 기록

**브랜치**: `claude/project-thread-rlen71` | **작성일**: 2026-11-02 | **명세**: [spec.md](spec.md)

## 요약

`trace-ml db init`, `trace-ml dataset device add|list`, `trace-ml dataset build|list`, `trace-ml train`, `trace-ml models list`를 더한다. 저장소는 boto3 S3 클라이언트(MinIO), 기록부는 psycopg 3(PostgreSQL `ml` 스키마)이고 둘 다 주입받아 테스트에서 메모리 구현으로 바꾼다. 학습은 Ultralytics를 함수 하나로 감싸 선택 의존성(`.[train]`)으로 둔다.

코드가 400줄을 넘으므로 PR 세 개로 나눈다: PR 1 기반(설정·저장소·기록부), PR 2 빌더(FR-DAT-04), PR 3 학습 기록(FR-DAT-05).

## 기술 맥락

- **언어**: Python 3.12(집 PC RTX 4070 Ti), 3.11 이상
- **의존성**: `boto3>=1.35`(IfNoneMatch), `psycopg[binary]>=3.2`, `PyYAML`(data.yaml). 학습만 `ultralytics>=8.3`(선택)
- **테스트**: pytest + 메모리 S3·메모리 기록부·가짜 학습 함수. CI의 PostgreSQL 서비스가 있으면(`TRACE_ML_TEST_DB`) 실제 기록부 테스트도 돈다
- **설정**: `MINIO_ENDPOINT`, `TRAINER_MINIO_ACCESS_KEY`·`TRAINER_MINIO_SECRET_KEY`(없으면 `MINIO_ACCESS_KEY`·`MINIO_SECRET_KEY`), `TRACE_RAW_BUCKET`·`TRACE_LABELS_BUCKET`·`TRACE_DATASETS_BUCKET`·`TRACE_MODELS_BUCKET`(기본 trace-*), `POSTGRES_HOST`(기본 localhost)·`POSTGRES_PORT`·`POSTGRES_DB`(기본 trace)·`DB_USERNAME`·`DB_PASSWORD`(gitops 이름 그대로)

## 헌법 점검

| 원칙 | 점검 | 결과 |
| --- | --- | --- |
| I. 계약 | 클래스는 `codes.py`(codes.yaml `inspection.defect_class`)만 씀. 계약 변경 없음 | 통과 |
| II. 경계 | `ml` 스키마만 씀. Label Studio DB는 보지 않고 `trace-labels` 내보내기만 읽음 | 통과 |
| III. 기록 | 버전·모델 객체는 조건부 쓰기, 원본은 읽기만, 해시 대조 | 통과 |
| IV. 테스트 | 수용 시나리오 문장을 테스트 이름으로 | 통과 |
| V. 비밀 | 키·비밀번호는 환경 변수, repr·출력에 없음 | 통과 |
| VI. 단순 | 메타데이터 서버 없이 DB 표 3개 + manifest 한 파일 | 통과 |

## 구조

```text
src/trace_ml/
├── codes.py                  # defect_class (기존)
├── settings.py               # ML 설정(MinIO·버킷·DB)
├── store.py                  # S3 래퍼: get·put_new(If-None-Match)·list·exists
├── registry.py               # Registry 프로토콜 + PgRegistry + V1 SQL
├── dataset/
│   ├── annotations.py        # LS 내보내기 → LabeledImage (검증 포함)
│   ├── splits.py             # 기기 분할 배정
│   └── builder.py            # build 흐름, manifest
└── training/
    ├── fetch.py              # manifest → 로컬 YOLO 폴더(해시 대조)
    ├── yolo.py               # Ultralytics 학습·ONNX·지표 (함수 하나)
    └── record.py             # train 흐름, 올리기·기록
tests/
├── fakes.py                  # 메모리 S3, 메모리 기록부, LS 내보내기 만들기
├── test_dataset_builder.py
├── test_training.py
└── test_registry_pg.py       # TRACE_ML_TEST_DB 있을 때만
```

## 조사 메모

- Label Studio 1.19 타깃 스토리지는 작업 형식(`{"id", "data", "annotations": [...]}`, 키 = 작업 id)으로 쓰고, 옛 설정에서는 주석 형식(`{"id", "result", "task": {...}}`, 키 = 주석 id)이다. 둘 다 읽는다
- 폴리곤 `value.points`는 이미지 크기 대비 % 좌표다 → /100 하면 YOLO 정규 좌표
- Ultralytics는 `images/` → `labels/`로 바꿔 라벨을 찾는다. 받는 폴더를 `images/<분할>`, `labels/<분할>`로 둔다
- `put_object(IfNoneMatch="*")`: 이미 있으면 412. MinIO 2024-08 이후 지원. 옛 MinIO면 무시되므로 접두사 확인을 먼저 한다
