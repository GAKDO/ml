# TRACE 헌법 (Constitution)

TRACE는 노트북 렌탈의 출고·수령·반납 세 시점에 외관과 내부 상태를 같은 기준으로 기록하고, 무엇이 언제 달라졌는지 해시 체인으로 증명하는 서비스입니다. 이 문서는 모든 레포의 Spec Kit 명세·계획·구현이 지켜야 할 원칙입니다. 원본은 GAKDO/docs `speckit/constitution.md`이고, 각 레포의 `.specify/memory/constitution.md`는 그 사본입니다.

## 핵심 원칙

### I. 계약이 먼저다 (어길 수 없음)
- API(OpenAPI), 이벤트, 진단 JSON, 코드 값은 GAKDO/contracts가 원본이다. 코드 값(면·조명·카메라·클래스·등급·시점)은 `contracts/codes.yaml` 한 곳에서만 정의하고 DB에는 CHECK 제약으로 넣는다.
- 구현 중 계약을 바꿔야 하면 코드를 고치지 말고 멈춘 뒤 사람에게 알린다. 계약 PR을 contracts에 먼저 머지하고 그 버전을 쓰는 PR을 낸다.
- 11/3 계약 고정 뒤에는 깨지는 변경(필드 삭제·이름 변경·값 삭제)을 하지 않는다. 더하는 변경만 한다.

### II. 서비스 경계를 지킨다
- 서버마다 자기 PostgreSQL 스키마만 읽고 쓴다(auth·rental·capture·inspection·diagnostics·report·ml). 서비스 간 외래 키는 걸지 않고, 다른 서비스의 ID는 값으로만 든다.
- 서비스 사이 정합성은 RabbitMQ topic exchange `trace.events`의 이벤트 5종으로 맞춘다. 부스 탐지 결과만 예외로 gateway를 거쳐 REST로 올린다.
- 사진·데이터셋 같은 큰 파일은 서버를 거치지 않고 MinIO 사전 서명 URL로 직접 올린다.
- 외부에 여는 앱은 gateway(8314)·web(8315)·label-studio(8316)뿐이다.

### III. 기록은 고칠 수 없다
- 모든 사진과 진단 JSON에 SHA-256을 남기고, 해시를 이전 기록에 잇는다(해시 체인, 서버 Ed25519 서명).
- 원본 사진은 덮어쓰거나 지우지 않는다. 봉인된 시점의 기록은 바꾸지 않는다.
- 읽지 못한 값은 0이 아니라 null + 생략 사유로 남긴다. 결과에는 모델 버전·등급 규칙 버전을 함께 남긴다.

### IV. 테스트와 품질 게이트 (어길 수 없음)
- 수용 기준마다 자동 테스트가 있다. 명세의 수용 기준 문장을 테스트 이름이나 설명에 그대로 쓴다.
- 서버는 `./mvnw verify`(JaCoCo 라인 커버리지 40% 이상), 화면은 `npm run typecheck && npm test`, 파이썬은 `ruff check . && pytest`가 통과해야 PR을 머지한다.
- CI(테스트·비밀값 검사·이미지 빌드)가 초록이 아니면 머지하지 않는다.

### V. 비밀과 개인정보를 지킨다
- 비밀번호·토큰·키·학번은 코드·문서·커밋·채팅에 넣지 않는다. 비밀은 Rancher Secret과 GitHub Secret에만 둔다. gitleaks와 `scripts/check-no-secrets.sh`를 통과한다.
- 인증은 auth-server가 발급하는 JWT(RS256)이고, gateway와 각 서버는 JWKS로 검증한다. 역할·범위는 최소로 준다. `/api/v1/*/ping`은 배포 확인용이라 로그인 없이 연다.
- 진단 에이전트는 사용자 파일·브라우저 기록을 읽지 않고, 컴퓨터 이름·사용자 이름은 보내기 전에 지운다. 파일럿 실명과 측정값은 같은 테이블에 두지 않는다.
- 공유 인프라 자원은 모두 `trace` 접두사로 격리하고 다른 팀 자원을 읽거나 쓰지 않는다.

### VI. 단순하게, 일정 안에
- MVP 요구사항 50개(설계서)만 만든다. 명세에 없는 기능, 미래를 위한 추상화는 넣지 않는다. 확장 지점은 설계서 "확장성" 섹션이 정한 곳만 연다.
- 일정: 11/3 계약 고정, 11/10 기능 완성, 11/13 코드 프리즈. 일정이 밀리면 범위가 아니라 완성도를 먼저 지킨다.

## 기술과 형식

- 서버: Java 21, Spring Boot 3.5.7, Maven, Spring Data JPA, Flyway(서버마다 자기 스키마 마이그레이션), Spring Security 리소스 서버, Spring AMQP. gateway는 Spring Cloud Gateway 2025.0.3(WebFlux).
- 화면: React 18, TypeScript, Vite, React Three Fiber, Vitest. API 타입은 contracts의 OpenAPI로 만든다.
- 파이썬: 3.12(부스 Jetson은 JetPack 6의 3.10), pyproject, ruff, pytest.
- JSON 필드는 camelCase, 시각은 ISO 8601 + 오프셋, ID는 밖에 보이는 publicId(NanoID 21자)이고 DB 기본 키는 bigint `id`로 밖에 내보내지 않는다, REST 오류는 RFC 9457 Problem Details, 메시지마다 `schemaVersion`.
- 배포는 GitOps다. 이미지를 GHCR에 올리고 gitops 레포의 태그를 바꾸면 학교 Argo CD가 반영한다. 클러스터에 kubectl로 직접 배포하지 않는다.
- DB 마이그레이션은 이전 버전 코드에서도 동작하게 나눈다(컬럼 추가 먼저, 삭제는 다음 배포). 자동 롤백은 코드만 되돌리기 때문이다.

## 개발 흐름

- Git Flow: `develop`에서 `feature/<번호>-<이름>`을 따고 PR은 `develop`으로, 릴리스는 `release/x.y.z` → `main`. Spec Kit은 `feature/<명세 번호>-<이름>` 브랜치를 만든다.
- 명세 순서: `/speckit-specify` → `/speckit-clarify` → `/speckit-plan` → `/speckit-tasks` → `/speckit-analyze` → `/speckit-implement` → `/speckit-converge`. 명세·계획·작업 문서(`specs/<번호>-<이름>/`)도 같은 PR에 넣는다.
- 커밋은 `<종류>(<범위>): <내용>`(feat·fix·refactor·test·docs·chore·ci), PR 본문에 요구사항 ID(예: FR-RNT-05)를 적는다. 문서·PR·주석은 한국어로 쓴다.
- 팀 결정은 GAKDO/docs `decisions.md`에 남긴다. 설계 원본은 Claude 설계서와 Crowfoot ERD이고, 바뀌면 원본을 먼저 고친다.

## 운영 규칙

- 이 헌법은 다른 모든 관례보다 앞선다. 계획(`/speckit-plan`)의 헌법 점검에서 어긋나는 항목은 이유를 적고 사람의 승인을 받는다.
- 헌법을 바꾸려면 GAKDO/docs에 PR을 내고, 머지되면 모든 레포의 `.specify/memory/constitution.md`를 같은 내용으로 바꾼다. 버전은 원칙 삭제·의미 변경이면 주 버전, 원칙 추가면 부 버전, 문구 수정이면 수 버전을 올린다.

**Version**: 1.1.0 | **Ratified**: 2026-11-01 | **Last Amended**: 2026-11-02
