# ml

TRACE의 inspect-worker·recon-worker·데이터셋 빌더·학습 (집 PC) 레포입니다. 담당: A (ML3은 B). 이 레포의 기능: ML1, ML2, ML3 (GAKDO/docs `speckit/backlog.md`).

## 먼저 읽을 것

- 헌법: `.specify/memory/constitution.md` (모든 명세·계획·구현의 기준. 원본은 GAKDO/docs `speckit/constitution.md`)
- 기능 목록과 붙여 넣을 설명: GAKDO/docs [`speckit/backlog.md`](https://github.com/GAKDO/docs/blob/main/speckit/backlog.md)
- 설계서: [`design/mvp-design.md`](https://github.com/GAKDO/docs/blob/main/design/mvp-design.md), REST API: [`design/rest-api-v0.md`](https://github.com/GAKDO/docs/blob/main/design/rest-api-v0.md), 테이블: [`design/ddl.md`](https://github.com/GAKDO/docs/blob/main/design/ddl.md), 결정: [`decisions.md`](https://github.com/GAKDO/docs/blob/main/decisions.md)
- 계약 원본: [GAKDO/contracts](https://github.com/GAKDO/contracts) (OpenAPI·JSON Schema·`codes.yaml`)

## 명령

- 설치: `python -m pip install -e ".[dev]"`
- 검사: `ruff check . && pytest -q`

## Spec Kit으로 기능 하나 만들기

1. `git switch develop && git pull`
2. `/speckit-specify` + backlog.md의 설명 → `feature/<번호>-<이름>` 브랜치와 `specs/<번호>-<이름>/spec.md`
3. `/speckit-clarify` → 질문에 답하기
4. `/speckit-plan` + 아래 "계획에 줄 기술 설명"
5. `/speckit-tasks` → `/speckit-analyze` → `/speckit-implement` → `/speckit-converge` (Converged까지 반복)
6. 로컬 검사가 통과하면 PR을 `develop`으로. 본문에 요구사항 ID를 적고 `specs/` 문서도 함께 올린다

## 계획에 줄 기술 설명

`/speckit-plan` 뒤에 그대로 붙여 넣습니다.

```
Python 3.12(집 PC RTX 4070 Ti), PyTorch, Ultralytics YOLO11-seg, SuperPoint + LightGlue, OpenCV 호모그래피, pika(RabbitMQ), boto3(MinIO), redis. GPU 없는 CI에서도 도는 작은 고정 데이터로 테스트한다. ruff·pytest 통과.
```

## 꼭 지킬 것

- 계약(API·이벤트·진단 JSON·코드 값)을 바꿔야 하면 코드를 고치지 말고 멈추고 사람에게 알린다.
- 비밀번호·토큰·키·학번을 코드·문서·커밋에 넣지 않는다. 설정은 환경 변수로 받는다.
- 커밋은 `<종류>(<범위>): <내용>`(feat·fix·refactor·test·docs·chore·ci), 문서·PR·주석은 한국어.
- 레포 이름은 `_server`, 클러스터 안 서비스·이미지 이름은 `-server`(예: `auth-server`).
