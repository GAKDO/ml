<!-- base 브랜치: 기능·수정은 develop, 릴리스 PR과 hotfix만 main (CONTRIBUTING.md) -->
## Summary
<!-- 무엇이 바뀌었는지 한두 줄 -->

## Why
<!-- 왜 필요한지, 관련 요구사항 ID(예: FR-RNT-05)와 이슈(Closes #12) -->

## 검증
- [ ] `ruff check .`과 `pytest` 통과
- [ ] 계약(API·이벤트·진단 JSON·코드 값)을 바꿨다면 contracts 레포 PR이 먼저 머지됨
- [ ] 비밀값을 커밋하지 않음 (`scripts/check-no-secrets.sh`)

## 참고
<!-- 스크린샷, 남은 일, 리뷰어가 볼 곳 -->
