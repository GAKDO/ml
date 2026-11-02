# 작업: ML1 데이터셋 빌더·학습 기록

## PR 1 기반: 설정·저장소·기록부 (FR-DAT-04·05 공통)

- [x] T001 `pyproject.toml`: boto3·psycopg·PyYAML, 선택 `train`(ultralytics)
- [x] T002 [P] `settings.py`: MinIO·버킷·DB 환경 변수, 비밀은 repr 숨김
- [x] T003 [P] `store.py`: S3 래퍼(put_new = If-None-Match, 412 → 이미 있음)
- [x] T004 [P] `registry.py`: Registry 프로토콜, PgRegistry, V1 SQL, `db init`
- [x] T010 [P] `tests/fakes.py`, `tests/test_store_settings.py`, `tests/test_registry_pg.py` + CI PostgreSQL 서비스

## PR 2 빌더 (FR-DAT-04)

- [x] T005 `dataset/annotations.py`: 작업·주석 형식 읽기, 건너뛰기 제외, 주석 고르기, 검증(빈 제출·둘 다·클래스·키·점 수)
- [x] T006 `dataset/splits.py`: 기기 배정(수동 우선, 자동 70/15/15), 쓰인 기기 분할 변경 금지
- [x] T007 `dataset/builder.py`: 버전 이름, 중복 거절, 라벨·data.yaml·manifest 쓰기, 기록
- [x] T008 `__main__.py`: `db init`, `dataset device add|list`, `dataset build|list`
- [x] T009 [P] `tests/test_dataset_builder.py`: 스토리 1 수용 시나리오·경계 상황

## PR 3 학습 기록 (FR-DAT-05)

- [x] T011 `training/fetch.py`: manifest → 로컬 폴더, 해시 대조
- [x] T012 `training/yolo.py`: Ultralytics 학습·TEST 검증·ONNX 내보내기(함수 하나)
- [x] T013 `training/record.py`: 올리기(조건부·메타데이터), training_run 기록
- [x] T014 `__main__.py`: `train`, `models list`
- [x] T015 [P] `tests/test_training.py`: 스토리 2 수용 시나리오
- [ ] T016 quickstart.md, README 명령 표

## 실제 환경 확인 (사용자 장비)

- [ ] T017 집 PC에서 port-forward + `db init` + v001 빌드(11/5 라벨링 마감 뒤)
- [ ] T018 집 PC에서 `train v001` → ONNX를 Jetson(BO2)에 전달
