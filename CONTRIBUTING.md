# 브랜치 전략과 릴리스 절차

TRACE의 모든 레포는 블로그(AIGJ-01-002-blog/devlog)와 같은 **목표 단위 Git Flow**로 개발한다.
설계와 결정은 [GAKDO/docs](https://github.com/GAKDO/docs)에 있다.

| 브랜치 | 어디서 만드나 | 어디로 합치나 | 버전 | 배포 |
|---|---|---|---|---|
| `main` | - | - | 릴리스할 때만 올림 | 학교 클러스터(trace 네임스페이스) 배포 + 태그 |
| `develop` | `main` | `main`(릴리스 PR) | 올리지 않음 | 없음. PR·푸시마다 테스트와 이미지 빌드(`:develop` 태그) |
| `feature/<이슈번호>-<이름>` | `develop` | `develop` | 올리지 않음 | 없음 |
| `hotfix/<버전>` | `main` | `main`, 그다음 `develop` | Patch | 학교 클러스터 배포 |

## 1. 기능 개발 (feature → develop)

1. `develop`에서 브랜치를 딴다: `git switch -c feature/12-device-list origin/develop`.
2. 작업하고 PR을 **`develop`으로** 연다. 본문에 `Closes #이슈번호`와 요구사항 ID(예: FR-RNT-05)를 적는다.
3. PR에서 CI·비밀값 검사·이미지 빌드(올리지 않음)가 돈다. 모두 초록이면 squash 머지한다.
4. `CHANGELOG.md`의 `## [Unreleased]` 아래 알맞은 구획(추가·바꿈·고침)에 한 줄을 더한다.

## 2. 계약을 바꿀 때

API·이벤트·진단 JSON·코드 값은 [contracts](../../../contracts) 레포가 원본이다. 계약을 바꾸는 PR을 contracts에 먼저 머지하고,
그 버전을 쓰는 PR을 서버·화면·에이전트 레포에 낸다. 11/3 계약 고정 이후에는 깨지는 변경(필드 삭제·이름 변경)을 하지 않는다.

## 3. 릴리스 (develop → main)

1. `develop`에서 `release/x.y.z` 브랜치를 딴다.
2. 버전을 올린다(서버는 `pom.xml`, 화면은 `package.json`, 파이썬은 `pyproject.toml`).
3. `CHANGELOG.md`의 `## [Unreleased]`를 `## [x.y.z] - 날짜`로 바꾸고 [릴리스 노트 틀](.github/RELEASE_TEMPLATE.md)에 맞춘다.
4. PR을 **`main`으로** 열고, 초록이면 **merge commit으로** 머지한다(squash하지 않는다).
5. main 푸시로 `배포`가 이미지를 올리고 gitops 레포의 이미지 태그를 바꾸면 Argo CD가 학교 클러스터에 반영한다.
   `release`가 태그와 GitHub Release를 만든다.

## 4. 급한 수정 (hotfix)

데모가 깨졌거나 보안 문제일 때만 쓴다. `main`에서 `hotfix/x.y.z`를 따서 고치고, `main`에 머지한 뒤 바로 `main`을 `develop`에 합친다.

## 커밋 메시지

`<종류>(<범위>): <내용>` — 종류는 feat·fix·refactor·test·docs·chore·ci. 예: `feat(device): 기기 목록 API (FR-RNT-05)`.
