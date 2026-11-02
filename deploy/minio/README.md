# trainer MinIO 키 (집 PC 전용)

데이터셋 빌더·학습(`trace-ml dataset build`, `trace-ml train`)만 쓰는 키입니다. 클러스터 앱은 쓰지 않으므로 trace-secrets에는 넣지 않고, 집 PC의 환경 변수로만 둡니다.

| 버킷 | 권한 |
| --- | --- |
| `trace-raw`, `trace-labels` | 읽기·목록 |
| `trace-datasets`, `trace-models` | 읽기·쓰기·목록, **삭제 없음**(버전 덮어쓰기 방지의 바탕) |

## 만들기 (MinIO 관리자 권한이 있는 사람이 한 번)

```bash
mc alias set trace http://storage.java21.net:8000 <관리자 키> <관리자 비밀>
mc mb --ignore-existing trace/trace-datasets trace/trace-models
mc admin policy create trace trace-trainer deploy/minio/trainer-policy.json
mc admin user add trace trace-trainer '<새 비밀, 40자 이상 무작위>'
mc admin policy attach trace trace-trainer --user trace-trainer
```

관리자 권한이 없으면 학교 MinIO 담당자에게 이 파일(`trainer-policy.json`)을 주고 같은 키를 요청합니다. 그 전까지는 팀 키(`MINIO_ACCESS_KEY`·`MINIO_SECRET_KEY`)로 대신 돕니다.

## 집 PC 설정

```bash
export TRAINER_MINIO_ACCESS_KEY=trace-trainer
export TRAINER_MINIO_SECRET_KEY=...   # 채팅·코드·커밋에 쓰지 않음
```
