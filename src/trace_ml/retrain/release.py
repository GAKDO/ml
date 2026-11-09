"""릴리스 묶음: 서버와 Jetson에 같은 best.onnx·model.json을 보내고, 두 곳 모두 해시로 확인하게 한다.

releases/<modelVersion>/ 에 best.onnx·model.json·RELEASE.json·deploy-server.sh·deploy-jetson.sh를 쓴다.
두 스크립트에는 같은 기대 해시가 박혀 있어, 둘 다 '확인 끝'이 나오면 두 곳의 모델이 같다(모델 일치 원칙).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .pipeline import Entry, RetrainError

_CHECK = """set -euo pipefail
cd "$(dirname "$0")"
VERSION='{version}'
echo "{onnx}  best.onnx
{info}  model.json" | sha256sum -c --quiet || {{ echo "파일 해시가 RELEASE.json과 다름: 다시 복사하세요" >&2; exit 1; }}
"""

SERVER = """#!/usr/bin/env bash
# 학교 서버에서: 이 폴더를 scp로 올린 뒤 bash deploy-server.sh
{check}COMPOSE="${{TRACE_COMPOSE_DIR:-$HOME/gitops/compose}}"
BACKUP="$COMPOSE/models-previous"
mkdir -p "$BACKUP"
# 되돌릴 수 있게 지금 모델을 먼저 옮겨 둔다(models/ 안에 두면 워커가 같이 읽으므로 밖에)
for f in best.onnx model.json; do
  if [ -f "$COMPOSE/models/$f" ]; then cp "$COMPOSE/models/$f" "$BACKUP/$f"; fi
done
cp best.onnx model.json "$COMPOSE/models/"
cd "$COMPOSE"
docker compose restart inspect-worker
for _ in $(seq 1 30); do
  if docker compose logs --since 5m inspect-worker 2>&1 | grep -qF "폰 탐지 모델 $VERSION"; then
    echo "서버 확인 끝: $VERSION ({onnx_short})"
    echo "성능 화면 값은 .env MODEL_METRICS에 $VERSION=<mAP50 box>/<mAP50 mask>를 더하고 docker compose up -d capture-server"
    exit 0
  fi
  sleep 2
done
echo "inspect-worker 로그에 '폰 탐지 모델 $VERSION'이 없음: docker compose logs --tail 50 inspect-worker" >&2
exit 1
"""

JETSON = """#!/usr/bin/env bash
# 부스 Jetson(jetson2)에서: 이 폴더를 USB·scp로 옮긴 뒤 bash deploy-jetson.sh
{check}PY="${{TRACE_BOOTH_PYTHON:-$HOME/trace-venv/bin/python}}"
ROOT="${{TRACE_MODELS_ROOT:-$HOME/trace-models}}"
SRC="$ROOT/$VERSION-src"; OUT="$ROOT/$VERSION"
mkdir -p "$SRC"
cp best.onnx model.json "$SRC/"
if [ ! -f "$OUT/best.engine" ]; then
  "$PY" -m trace_booth build-engine --onnx "$SRC/best.onnx" --out "$OUT"
fi
echo "{info}  $OUT/model.json" | sha256sum -c --quiet || {{ echo "엔진 옆 model.json이 릴리스와 다름: $OUT" >&2; exit 1; }}
ln -sfn "$OUT" "$ROOT/current"
echo "Jetson 확인 끝: $VERSION ({onnx_short})"
echo "부스 앱을 다시 띄우세요: export TRACE_BOOTH_ENGINE=$ROOT/current/best.engine && $HOME/trace-venv/bin/trace-booth serve"
"""


def write_release(entry: Entry, model_dir: Path, releases: Path, metrics: str | None = None) -> Path:
    """레지스트리에 올린 모델의 배포 묶음. 이미 있으면 덮어쓰지 않는다."""
    out = releases / entry.modelVersion
    if out.exists():
        raise RetrainError(f"릴리스 폴더가 이미 있음(덮어쓰지 않음): {out}")
    out.mkdir(parents=True)
    for name in ("best.onnx", "model.json"):
        shutil.copyfile(model_dir / name, out / name)
    check = _CHECK.format(version=entry.modelVersion, onnx=entry.onnxSha256, info=entry.modelJsonSha256)
    values = {"check": check, "onnx_short": entry.onnxSha256[:12], "info": entry.modelJsonSha256}
    (out / "deploy-server.sh").write_text(SERVER.format(**values), encoding="utf-8", newline="\n")
    (out / "deploy-jetson.sh").write_text(JETSON.format(**values), encoding="utf-8", newline="\n")
    doc = {"schema": "trace.model-release.v1", "modelVersion": entry.modelVersion, "previous": entry.previous,
           "kind": entry.kind, "promotedAt": entry.promotedAt,
           "files": {"best.onnx": entry.onnxSha256, "model.json": entry.modelJsonSha256},
           "modelMetrics": metrics, "targets": ["school-server inspect-worker", "booth jetson2"]}
    (out / "RELEASE.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
