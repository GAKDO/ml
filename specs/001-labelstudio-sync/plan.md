# 구현 계획: ML3 Label Studio 연동

**브랜치**: `claude/project-thread-sy1p9l` | **작성일**: 2026-11-02 | **명세**: [spec.md](spec.md)

## 요약

`trace-ml labelstudio setup|sync` 명령을 만든다. Label Studio REST API(1.19)를 감싼 작은 클라이언트가 프로젝트·라벨 설정·S3 소스(trace-raw, 프록시 모드)·S3 타깃(trace-labels)을 없을 때만 만들고, sync에서 두 스토리지 동기화를 부른다.

## 기술 맥락

- **언어**: Python 3.12(집 PC·관리자 노트북), 3.11 이상
- **의존성**: `requests`(HTTP). Label Studio SDK는 쓰지 않는다(필요한 API가 6개뿐)
- **테스트**: pytest + 가짜 HTTP 세션(요청 기록·응답 지정). 네트워크 없음
- **설정**: `LABEL_STUDIO_URL`, `LABEL_STUDIO_API_KEY`, `MINIO_ENDPOINT`, `LABEL_STUDIO_MINIO_ACCESS_KEY`·`LABEL_STUDIO_MINIO_SECRET_KEY`(없으면 `MINIO_ACCESS_KEY`·`MINIO_SECRET_KEY`), `TRACE_RAW_BUCKET`(기본 trace-raw), `TRACE_LABELS_BUCKET`(기본 trace-labels)

## 헌법 점검

| 원칙 | 점검 | 결과 |
| --- | --- | --- |
| I. 계약 | 클래스는 codes.yaml 0.3.0 `inspection.defect_class` 값을 한 곳(`codes.py`)에만 둠 | 통과 |
| II. 경계 | Label Studio DB는 건드리지 않고 API만 씀. 사진은 MinIO에서 Label Studio가 직접 읽음 | 통과 |
| III. 기록 | 소스는 읽기만, 타깃은 삭제 금지, 소스=타깃 거절 | 통과 |
| IV. 테스트 | 수용 시나리오 문장을 테스트 이름으로 | 통과 |
| V. 비밀 | 토큰·키는 환경 변수, 출력하지 않음 | 통과 |
| VI. 단순 | SDK 없이 requests, 명령 두 개 | 통과 |

## 구조

```text
src/trace_ml/
├── codes.py               # defect_class 값
└── labelstudio/
    ├── __init__.py
    ├── config.py          # 라벨 설정 XML, 키 필터, 설정값
    ├── client.py          # Label Studio API 래퍼 (세션 주입)
    └── setup.py           # setup·sync 흐름
src/trace_ml/__main__.py   # trace-ml labelstudio setup|sync
tests/
├── fake_ls.py             # 가짜 Label Studio
└── test_labelstudio.py
```

## 조사 메모

- 소스: `POST /api/storages/s3` (`use_blob_urls: true`로 객체 하나가 작업 하나, `presign: false`가 프록시 모드, `regex_filter`, `recursive_scan: true`), 동기화 `POST /api/storages/s3/{id}/sync`
- 타깃: `POST /api/storages/export/s3` (`can_delete_objects: false`), 동기화 `POST /api/storages/export/s3/{id}/sync`
- 인증 헤더 `Authorization: Token <키>`
- 키 필터: `^[A-Z]{2,4}[0-9]{2}/[A-Z]{2,4}[0-9]{2}_(TOP|BOT|SF|SB|SL|SR|KB|SCR0|SCR1)_(L1|L2|L3|L4|OFF)_(CAM1|CAM2)_[0-9]{2}\.jpg$` (slate/는 걸러짐)
- 실제 Label Studio에서의 확인은 클러스터 배포(A) 뒤에 quickstart로 한다
