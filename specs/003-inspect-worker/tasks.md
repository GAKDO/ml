# 작업: ML2 inspect-worker (간소화)

## PR 1 정합 (FR-INS-02)

- [x] T001 `pyproject.toml`: opencv-python-headless·numpy, dev에 jsonschema
- [x] T002 [P] `inspect/faces.py`: faces.md 면 크기, 기종 표(MBA13-M2), `TRACE_DEVICE_MODEL`
- [x] T003 [P] `inspect/contour.py`: 윤곽 4점, 정렬, 직선 보정(0.5 px), 한 변 잘린 면 늘리기
- [x] T004 `inspect/align.py`: 호모그래피·펼치기·errorPx·coverage·to_standard
- [x] T008 `inspect/synthetic.py`·`measure.py`·`cli.py`: `align photo|eval|pair`
- [x] T009 [P] 테스트: `test_inspect_align.py`
- [x] T012 정합 오차 합성 수치 → docs `contest/`

## PR 2 워커 (FR-OPS-01, FR-RPT-01 텍스처, FR-INS-01 인터페이스)

- [x] T001a `pyproject.toml`: pika·redis
- [x] T005 [P] `inspect/detect.py`: Detector·NullDetector·길이·넓이
- [x] T006 `inspect/worker.py`: 해시 대조, 펼친 이미지·텍스처(이어 붙이기) 올리기, 결과 본문, stable_id
- [x] T007 `inspect/runner.py`: 선언(trace.dlx topic, dead 키), 처리→발행→ack, 3번 재시도, Redis 진행률
- [x] T010 CI: RabbitMQ 서비스, `TRACE_ML_TEST_AMQP`
- [x] T011 `deploy/inspect-worker/`: systemd·작업 스케줄러·worker.env 형식
- [x] T008a `cli.py`: `inspect worker`
- [x] T009a [P] 테스트: `test_inspect_worker.py`·`test_inspect_runner.py`(실제 RabbitMQ는 CI)

## 사용자 장비·후속

- [ ] T013 집 PC에 설치·자동 시작 등록(README), 실제 세트 한 번 처리
- [ ] T014 실제 부스·폰 사진으로 `align pair` → docs `contest/` 실측 수치
- [x] T015 폰 탐지 모델 연결(11/2 A 담당 결정): `yolo_detect.TiledDetector` + `TRACE_DETECT_WEIGHTS`
- [x] T016 이벤트 `device`(기종·치수)로 표준 좌표 크기 정하기(contracts #15)
- [ ] T016 (시간 남으면) LIGHTGLUE 보정, 180° 판별
