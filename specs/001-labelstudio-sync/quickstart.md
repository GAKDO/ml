# 빠른 확인: ML3

```bash
python -m pip install -e ".[dev]"
ruff check . && pytest -q
```

실제 Label Studio(클러스터 NodePort 8316 또는 노트북 Docker)에서:

```bash
export LABEL_STUDIO_URL=http://<주소>:8316
export LABEL_STUDIO_API_KEY=...            # 화면 Account & Settings에서 발급, 채팅·코드에 쓰지 않음
export MINIO_ENDPOINT=http://storage.java21.net:8000   # S3 API 주소. https://storage.java21.net/은 웹 관리 화면이라 안 됨
export LABEL_STUDIO_MINIO_ACCESS_KEY=...  LABEL_STUDIO_MINIO_SECRET_KEY=...   # trace-labelstudio 계정 키(공용 nhnacademy 계정은 쓰지 않음)
trace-ml labelstudio setup
trace-ml labelstudio sync        # 촬영 뒤마다
```

확인: 촬영 도구로 REG01을 찍은 뒤 sync 한 번에 작업이 28장(+SCR1) 늘고 SLATE는 없는지, 주석 하나를 제출하고 sync하면 `trace-labels`에 파일이 생기는지, `trace-raw` 객체의 수정 시각이 그대로인지 본다.

화면에서 손으로 확인하거나 고칠 때 맞아야 하는 값(A 11/2 공지, `setup`이 만드는 값과 같음):

| 항목 | 값 |
| --- | --- |
| S3 endpoint | `http://storage.java21.net:8000/` |
| 소스 스토리지 버킷 | `trace-raw` |
| 타깃 스토리지 버킷 | `trace-labels` |
| Use pre-signed URLs | 끔(프록시 모드) |
| 키 | trace-labelstudio 계정 |
