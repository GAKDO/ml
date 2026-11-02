# 빠른 확인: ML1

```bash
python -m pip install -e ".[dev]"          # 학습까지 하려면 ".[dev,train]"
ruff check . && pytest -q
```

## 집 PC에서 데이터셋 버전 만들기

```bash
kubectl -n trace port-forward svc/postgres 5432:5432     # 다른 창에 띄워 둔다
export DB_USERNAME=... DB_PASSWORD=...                      # trace-secrets의 값, 채팅·코드에 쓰지 않음
export MINIO_ENDPOINT=https://storage.java21.net
export TRAINER_MINIO_ACCESS_KEY=... TRAINER_MINIO_SECRET_KEY=...   # 없으면 MINIO_ACCESS_KEY·MINIO_SECRET_KEY
trace-ml db init                                            # 처음 한 번
trace-ml labelstudio sync                                   # 제출된 주석을 trace-labels로 내보내기(ML3)
trace-ml dataset device add REG03 --split TEST --model "MacBook Air M2"   # 테스트 기기는 먼저 정한다
trace-ml dataset build                                      # → v001
trace-ml dataset list
trace-ml dataset device list                                # 자동 배정된 기기의 기종을 add로 채운다
```

확인: `trace-datasets/v001/`에 `labels/`·`data.yaml`·`manifest.json`이 있고, 같은 명령을 `--version v001`로 다시 돌리면 거절되는지, manifest `devices`에서 기기마다 분할이 하나인지 본다.
