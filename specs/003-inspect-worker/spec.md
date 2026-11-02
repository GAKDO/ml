# 기능 명세: ML2 inspect-worker (간소화)

**기능 브랜치**: `claude/project-thread-gnyohx` (Spec Kit 기본값은 `feature/003-inspect-worker`)

**작성일**: 2026-11-02

**상태**: 확정

**입력**: 백로그 ML2 설명 + 11/2 저녁 전체 공유의 "ML2 간소화": "정합은 윤곽 4점 호모그래피만 쓴다(LightGlue 보정은 시간 남으면). 사진별 alignments를 trace.inspect.completed로 낸다. 펼친 이미지를 trace-derived에 올리고, 3D 텍스처는 펼친 면 이미지를 그대로 쓴다. 폰 사진 탐지는 Jetson과 같은 모델로 하되, B가 맡을 수 있는지 확인 중이다(답 오기 전에는 인터페이스만 만들어 둔다)."

**요구사항 ID**: FR-INS-01(폰 탐지는 인터페이스만), FR-INS-02(정합), FR-OPS-01(꺼져 있던 동안 쌓인 작업 처리), FR-RPT-01(텍스처)

**계약**: contracts `events/trace.capture.uploaded.schema.json`(받음), `events/trace.inspect.completed.schema.json`(냄), `capture/faces.md`(면 방향·1 px = 0.25 mm), `codes.yaml` 0.5.0 `inspection.alignment_method`(CONTOUR). 계약 변경 없음.

## 명확화

### 세션 2026-11-02

(민서가 "묻지 말고 진행"을 요청해, 11/2 저녁 공유·decisions.md 11/2·faces.md에 맞는 답을 골라 적었다.)

- Q: 펼친 이미지 크기를 정하려면 기종 치수가 필요한데 `trace.capture.uploaded`에 기종이 없다. 어디서 얻나요? → A: 워커 설정 `TRACE_DEVICE_MODEL`(기본 `MBA13-M2`)과 워커 안의 기종 치수 표(rental V2 시드와 같은 값: 304.1 × 215.0 mm, 본체 7.0·상판 4.0 mm)를 쓴다. 시연 기종이 하나뿐이라 충분하다. 기종이 늘면 계약 고정 뒤 "더하는 변경"으로 이벤트에 기종 치수를 넣는다(사용자에게 질문으로 올림)
- Q: 윤곽 4점은 180° 돌아간 것과 구분이 안 된다(faces.md). 방향은? → A: 간소화판은 사진이 faces.md "이미지 위쪽"을 위로 찍혔다고 본다(부스는 카메라 고정, 폰은 윤곽 가이드가 이 방향으로 그림). 네 점은 사진 안 위치(왼쪽 위·오른쪽 위·오른쪽 아래·왼쪽 아래)로 정렬한다. 특징점 보정(LIGHTGLUE)은 시간이 남으면 더한다
- Q: 4점 호모그래피는 4점에서 오차가 항상 0이다. `errorPx`는 무엇인가요? → A: 윤곽 점 전체를 호모그래피로 옮긴 뒤 표준 직사각형 테두리까지의 거리 RMS(표준 좌표 px). 모서리가 둥글어 생기는 부분은 빼기 위해 꼭짓점 근처(짧은 변의 8%)는 재지 않는다. 윤곽을 못 찾으면 사진을 `failedPhotos`에 사유와 함께 넣는다
- Q: 텍스처는? → A: 면마다 정합에 성공한 사진 중 `errorPx`가 가장 작은 것의 펼친 이미지를 `textures/{captureSetId}/{face}.png`로 그대로 올린다. 같은 면 사진이 모두 실패하면 그 면은 `textureKeys`에서 빠진다
- Q: 키 형식은? → A: contracts 예시와 같게 `rectified/{photoId}.png`, `textures/{captureSetId}/{face}.png`(버킷 `trace-derived`). 같은 세트를 다시 처리하면 같은 키에 같은 결과를 덮어쓴다(워커 결과는 원본이 아니라 다시 만들 수 있는 파생물이다)
- Q: 폰 탐지는? → A: `Detector` 인터페이스(사진 → 원본 px 다각형·클래스·신뢰도)와, 호모그래피로 표준 px로 옮기고 길이·넓이(mm)를 재는 코드까지 만든다. 실제 모델은 B 답이 오기 전까지 `NullDetector`(빈 결과, 모델 버전 `detector-pending`)다. `needDetection`이 false(부스)면 탐지를 부르지 않는다
- Q: 같은 이벤트가 두 번 오면? → A: 결과 이벤트의 `eventId`와 `defectId`를 입력 `eventId`·사진·순번의 해시로 만들어 다시 와도 같다(README 멱등)
- Q: 진행률은? → A: `REDIS_HOST`가 있으면 Redis Pub/Sub 채널 `trace:progress:{rentalId}`에 `{"captureSetId","checkpointId","stage","done","total"}` JSON을 낸다(사진 하나 끝날 때마다). Redis가 없거나 실패해도 정합은 계속한다. report-server(RP2)가 이 채널을 구독한다
- Q: 실패한 메시지는? → A: 형식이 틀린 메시지는 바로, 처리 중 오류는 3번 다시 해 본 뒤 `trace.dlx`(topic)에 `trace.inspect.requested.dead`로 보낸다(decisions.md 11/2 DLX 규칙)

## 사용자 시나리오와 테스트 *(필수)*

### 사용자 스토리 1 - 업로드된 세트를 펼쳐 정합 결과를 낸다 (우선순위: P1)

capture-server가 `trace.capture.uploaded`를 내면, 워커가 사진마다 윤곽 4점을 찾아 표준 좌표로 펼치고, 펼친 이미지·면 텍스처를 `trace-derived`에 올린 뒤 사진별 호모그래피·오차를 `trace.inspect.completed`로 낸다.

**독립 테스트**: 합성 사진(배경 위에 원근 변형한 면)과 메모리 S3로 처리해 결과 이벤트·객체를 확인한다.

**수용 시나리오**:

1. **주어진 상황** 원근 변형한 TOP 사진일 때, **동작** 정합하면, **결과** 펼친 이미지가 1216 × 860 px(304.1 × 215.0 mm × 4)이고 꼭짓점 오차가 2 px 이하다.
2. **주어진 상황** 부스 세트(needDetection=false)일 때, **동작** 처리하면, **결과** 사진마다 alignments(method CONTOUR)가 있고 defects는 빈 배열, 면마다 textureKeys가 있다.
3. **주어진 상황** 윤곽이 없는 사진이 섞였을 때, **동작** 처리하면, **결과** 그 사진은 failedPhotos에 사유와 함께 있고 나머지는 정합된다.
4. **주어진 상황** 같은 이벤트가 두 번 올 때, **동작** 처리하면, **결과** 두 결과 이벤트의 eventId·내용이 같다.
5. **주어진 상황** 원본 사진 SHA-256이 이벤트와 다를 때, **동작** 처리하면, **결과** 그 사진은 failedPhotos("해시 불일치")다.

### 사용자 스토리 2 - 꺼져 있던 동안 쌓인 작업을 처리한다 (우선순위: P1)

집 PC가 꺼져 있어도 durable 큐 `trace.inspect.requested`에 이벤트가 쌓이고, 워커가 켜지면 모두 처리한다.

**수용 시나리오**:

1. **주어진 상황** 워커를 끈 채 세트 3개(사진 3장)를 올렸을 때, **동작** 워커를 켜면, **결과** 결과 이벤트 3개가 나오고 큐가 빈다. (실제 RabbitMQ, CI 서비스)
2. **주어진 상황** 형식이 틀린 메시지일 때, **동작** 받으면, **결과** DLQ로 가고 다음 메시지를 처리한다.

### 사용자 스토리 3 - 폰 사진 흠집을 표준 좌표로 낸다 (우선순위: P2, 모델은 B 답 뒤)

**수용 시나리오**:

1. **주어진 상황** 가짜 탐지기가 사진 px 다각형을 줄 때, **동작** 폰 세트(needDetection=true)를 처리하면, **결과** defects의 polygon은 표준 px이고 lengthMm·areaMm2가 1 px = 0.25 mm로 계산된다.
2. **주어진 상황** 부스 세트일 때, **동작** 처리하면, **결과** 탐지기를 부르지 않는다.

### 사용자 스토리 4 - 정합 오차를 잰다 (우선순위: P2, 경진대회 수치)

`trace-ml align eval`이 합성 사진으로 꼭짓점 오차(px·mm)를, `trace-ml align pair <부스> <폰>`이 실제 두 사진의 펼친 이미지 사이 오차를 잰다.

### 경계 상황

- 기종 설정이 표에 없으면 워커가 시작하지 않는다.
- 사진 디코딩 실패·윤곽 면적이 사진의 0.5% 미만이면 failedPhotos(가는 옆면도 받기 위해 낮게 잡음).
- 원근이 너무 커서 사각형이 볼록하지 않으면 failedPhotos.
- 결과 이벤트를 보낸 뒤에 ack한다(publisher confirm). 보내기 전에 죽으면 다시 처리한다.

## 요구사항

- **FR-001** `trace.capture.uploaded`를 큐 `trace.inspect.requested`(durable, DLX `trace.dlx`)에서 받는다.
- **FR-002** 사진마다 윤곽 4점 → faces.md 방향·4 px/mm 표준 크기로 호모그래피(사진 px → 표준 px)를 구하고 errorPx를 잰다.
- **FR-003** 펼친 이미지와 면 텍스처를 `trace-derived`에 올린다.
- **FR-004** `trace.inspect.completed`(스키마 1.0)를 `trace.events`에 persistent·`message_id = eventId`로 낸다.
- **FR-005** needDetection일 때 Detector 결과를 표준 px로 옮겨 defects로 낸다.
- **FR-006** 진행률을 Redis에 낸다(선택).
- **FR-007** 정합 오차 측정 명령을 둔다.

## 성공 기준

- 합성 사진 100장 꼭짓점 오차 평균 ≤ 1 px(0.25 mm), 최대 ≤ 3 px.
- CI 실제 RabbitMQ에서 쌓인 3건이 모두 처리된다.
