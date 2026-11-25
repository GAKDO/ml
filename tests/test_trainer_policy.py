"""이 파일이 확인하는 것: 학습용(trainer) MinIO 키의 권한 정책 파일.
원본 사진·라벨은 읽기만, 데이터셋·모델은 올릴 수 있지만 지울 수는 없어야 한다(실수로 원본을 망치지 않게).

trainer 키 정책: 원본·주석은 읽기만, 데이터셋·모델은 쓰되 지우지 못한다.
"""

from __future__ import annotations

import json
from pathlib import Path

POLICY = json.loads((Path(__file__).parent.parent / "deploy/minio/trainer-policy.json").read_text(encoding="utf-8"))


def _actions_on(bucket: str) -> set[str]:
    return {a for s in POLICY["Statement"] if s["Effect"] == "Allow"
            for r in s["Resource"] if r.startswith(f"arn:aws:s3:::{bucket}") for a in s["Action"]}


# [상황] 원본 사진(trace-raw)·라벨(trace-labels) 저장소에 허용된 동작을 본다.
# [기대] 읽기와 목록 보기만 있다.
def test_원본과_주석_버킷에는_쓰지_못한다():
    for bucket in ("trace-raw", "trace-labels"):
        assert _actions_on(bucket) == {"s3:GetObject", "s3:ListBucket"}


# [상황] 데이터셋(trace-datasets)·모델(trace-models) 저장소에 허용된 동작을 본다.
# [기대] 올리기는 되지만 지우기 권한은 하나도 없다.
def test_데이터셋과_모델_버킷은_쓸_수_있지만_지울_수_없다():
    for bucket in ("trace-datasets", "trace-models"):
        acts = _actions_on(bucket)
        assert "s3:PutObject" in acts and not any("Delete" in a for a in acts)
