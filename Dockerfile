# inspect-worker(ML2) 컨테이너: 폰 사진 정합·흠집 탐지(best.onnx, CPU)·3D 텍스처. 학교 서버 Compose에서 돈다.
# torch 없이 onnxruntime만 넣어 이미지를 작게 둔다. 모델은 이미지에 넣지 않고 /models 폴더를 붙인다
# (best.onnx·model.json만 바꾸고 `docker compose restart inspect-worker`).
FROM python:3.12.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRACE_MODEL_DIR=/models

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[worker]" \
    && useradd --system --uid 10001 --home /app trace \
    && mkdir -p /models && chown trace /models

ARG APP_COMMIT=dev
ENV APP_COMMIT=${APP_COMMIT}
USER trace
VOLUME ["/models"]
ENTRYPOINT ["trace-ml", "inspect", "worker"]
