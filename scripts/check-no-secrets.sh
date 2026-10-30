#!/usr/bin/env bash
# 저장소에 실제 비밀값이 들어가지 않았는지 검사한다. (블로그 deploy/scripts/check-no-secrets.sh를 옮김)
# - secret.env·kubeconfig·.env 파일이 추적되고 있으면 실패 (.example은 허용)
# - 학교 공용 비밀번호 패턴(nhnacademy로 시작하는 비밀번호 등)이 보이면 실패
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

fail=0
if git ls-files | grep -E '(^|/)secret\.env$|(^|/)kubeconfig|(^|/)\.env$' ; then
  echo "비밀 파일이 커밋되어 있습니다" >&2; fail=1
fi
# 패턴은 값 자체를 적지 않고 형태로만 찾는다
if git grep -n -I -i -E '(password|pwd|secret)[^=:]*[=:][[:space:]]*["'"'"']?[Nn]hnacademy[0-9]' -- . ':!scripts/check-no-secrets.sh' ; then
  echo "학교 공용 비밀번호로 보이는 값이 있습니다" >&2; fail=1
fi
[ "$fail" -eq 0 ] && echo "비밀값 없음"
exit $fail
