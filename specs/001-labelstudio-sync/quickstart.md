# 빠른 확인: ML3

```bash
python -m pip install -e ".[dev]"
ruff check . && pytest -q
```

실제 Label Studio(클러스터 NodePort 8316 또는 노트북 Docker)에서:

```bash
export LABEL_STUDIO_URL=http://<주소>:8316
export LABEL_STUDIO_API_KEY=...            # 화면 Account & Settings에서 발급, 채팅·코드에 쓰지 않음
export MINIO_ENDPOINT=https://storage.java21.net
export LABEL_STUDIO_MINIO_ACCESS_KEY=...  LABEL_STUDIO_MINIO_SECRET_KEY=...   # labelstudio 역할 키
trace-ml labelstudio setup
trace-ml labelstudio sync        # 촬영 뒤마다
```

확인: 촬영 도구로 REG01을 찍은 뒤 sync 한 번에 작업이 28장(+SCR1) 늘고 SLATE는 없는지, 주석 하나를 제출하고 sync하면 `trace-labels`에 파일이 생기는지, `trace-raw` 객체의 수정 시각이 그대로인지 본다.
