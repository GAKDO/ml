"""trainer 키 정책: 원본·주석은 읽기만, 데이터셋·모델은 쓰되 지우지 못한다."""

from __future__ import annotations

import json
from pathlib import Path

POLICY = json.loads((Path(__file__).parent.parent / "deploy/minio/trainer-policy.json").read_text())


def _actions_on(bucket: str) -> set[str]:
    return {a for s in POLICY["Statement"] if s["Effect"] == "Allow"
            for r in s["Resource"] if r.startswith(f"arn:aws:s3:::{bucket}") for a in s["Action"]}


def test_원본과_주석_버킷에는_쓰지_못한다():
    for bucket in ("trace-raw", "trace-labels"):
        assert _actions_on(bucket) == {"s3:GetObject", "s3:ListBucket"}


def test_데이터셋과_모델_버킷은_쓸_수_있지만_지울_수_없다():
    for bucket in ("trace-datasets", "trace-models"):
        acts = _actions_on(bucket)
        assert "s3:PutObject" in acts and not any("Delete" in a for a in acts)
