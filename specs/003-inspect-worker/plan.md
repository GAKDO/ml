# 구현 계획: ML2 inspect-worker (간소화)

**브랜치**: `claude/project-thread-gnyohx` | **작성일**: 2026-11-02 | **명세**: [spec.md](spec.md)

## 요약

`trace-ml inspect worker`가 RabbitMQ 큐 `trace.inspect.requested`에서 `trace.capture.uploaded`를 받아, 사진마다 윤곽 4점 호모그래피로 면을 표준 좌표(1 px = 0.25 mm)로 펼치고, 펼친 이미지·면 텍스처를 `trace-derived`에 올린 뒤 `trace.inspect.completed`를 낸다. 폰 탐지는 `Detector` 인터페이스와 좌표 변환·길이 재기까지, 모델은 B 답 뒤. 정합 오차는 `trace-ml align eval|pair`로 잰다.

부스 사진은 CAM1(앞 반)·CAM2(뒤 반)이라 한 변이 사진 밖으로 잘린다(booth README). 이 경우 보이는 세 변의 직선으로 참 꼭짓점 2개를 구하고, 보이는 변 길이로 배율을 잡아 잘린 꼭짓점 2개를 표준 길이만큼 늘려 만든다(카메라가 바닥을 수직으로 내려다봐 원근이 작다는 가정). 텍스처는 면마다 앞 반·뒤 반 펼친 이미지를 이어 붙인다.

## 기술 맥락

- **언어**: Python 3.12(집 PC), 3.11 이상
- **의존성**: `opencv-python-headless>=4.11`(approxPolyN), `numpy`, `pika`(RabbitMQ), `redis`(진행률). GPU 없음
- **테스트**: pytest + 합성 사진(정답 호모그래피) + 메모리 S3 + 가짜 채널. CI의 RabbitMQ 서비스(`TRACE_ML_TEST_AMQP`)로 쌓인 메시지 처리. 결과 이벤트는 contracts 스키마(옆 폴더 또는 GitHub develop)로 검사
- **설정**: `RABBITMQ_HOST`·`RABBITMQ_PORT`·`RABBITMQ_VHOST`·`RABBITMQ_USERNAME`·`RABBITMQ_PASSWORD`, `MINIO_ENDPOINT`, `WORKER_MINIO_ACCESS_KEY`·`WORKER_MINIO_SECRET_KEY`(없으면 `MINIO_*`), `TRACE_CAPTURES_BUCKET`·`TRACE_DERIVED_BUCKET`, `TRACE_DEVICE_MODEL`(기본 MBA13-M2), `REDIS_HOST`·`REDIS_PORT`·`REDIS_DATABASE`·`REDIS_PASSWORD`·`REDIS_KEY_PREFIX`(선택)

## 헌법 점검

| 원칙 | 점검 | 결과 |
| --- | --- | --- |
| I. 계약 | 받는·내는 이벤트를 contracts 스키마로 테스트. 키 형식은 contracts 예시. 계약 변경 없음 | 통과 |
| II. 경계 | DB를 보지 않고 이벤트와 MinIO만. 기종 치수는 워커 표(이벤트에 없음, 질문으로 올림) | 통과(질문 있음) |
| III. 기록 | 원본(trace-captures)은 읽기만, 해시 대조. 파생물만 덮어씀 | 통과 |
| IV. 테스트 | 수용 시나리오 문장을 테스트 이름으로 | 통과 |
| V. 비밀 | 환경 변수, repr 숨김, worker.env는 형식만 | 통과 |
| VI. 단순 | LightGlue 없이 윤곽 4점, 탐지 모델 없이 인터페이스 | 통과 |

## 구조

```text
src/trace_ml/inspect/
├── faces.py      # 면 크기(faces.md), 기종 치수 표
├── contour.py    # 윤곽 4점(온전한 면·한 변 잘린 면)
├── align.py      # 호모그래피·펼치기·errorPx·사진 px → 표준 px
├── detect.py     # Detector 인터페이스, NullDetector, 길이·넓이(mm)
├── worker.py     # 세트 하나 처리 → inspect.completed 본문
├── runner.py     # RabbitMQ 소비·발행·DLQ, Redis 진행률
├── synthetic.py  # 합성 사진(테스트·오차 재기)
├── measure.py    # 정합 오차 재기(합성·실제 두 사진)
└── cli.py        # inspect worker, align photo|eval|pair
deploy/inspect-worker/   # 집 PC 자동 시작(systemd·작업 스케줄러), worker.env 형식
```

## 조사 메모

- `getPerspectiveTransform`은 4점에서 정확히 맞으므로 4점 잔차는 0이다 → errorPx는 윤곽 점 전체의 테두리 거리 RMS로 잰다
- 픽셀 중심 정수 규칙: 표준 이미지 가장자리는 -0.5·w-0.5. 윤곽 점은 가장자리 픽셀 중심이라 0.5 px 안쪽 → 직선을 바깥으로 0.5 px 민다. 이 보정으로 합성 꼭짓점 오차가 1.6 → 0.3 px
- 가는 옆면(44 px 높이)은 x+y 최소로 왼쪽 위를 고르면 뒤집힌다 → "가장 위 변의 시작점"으로 정렬
- 사각형 근사가 안 되는 원근 사다리꼴은 `cv2.approxPolyN`(4.11+)
- 사진 테두리에 닿은 어두운 무늬는 마스크에 홈을 만든다 → 잘린 면의 오차는 보이는 세 변 가까이 점만으로
