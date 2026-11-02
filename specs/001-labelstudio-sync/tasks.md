# 작업: ML3 Label Studio 연동

- [x] T001 `pyproject.toml`에 `requests` 의존성
- [x] T002 [P] `src/trace_ml/codes.py`: defect_class 4종
- [x] T003 [P] `src/trace_ml/labelstudio/config.py`: 라벨 설정 XML, 키 필터, 환경 변수 설정
- [x] T004 `src/trace_ml/labelstudio/client.py`: 프로젝트·S3 소스·S3 타깃 찾기·만들기·동기화, 작업 수
- [x] T005 `src/trace_ml/labelstudio/setup.py`: setup(없을 때만 만듦, 설정 차이 알림, 소스=타깃 거절), sync
- [x] T006 `src/trace_ml/__main__.py`: `trace-ml labelstudio setup|sync`
- [x] T007 [P] `tests/fake_ls.py`, `tests/test_labelstudio.py`: 스토리 1·2 수용 시나리오, 경계 상황
- [ ] T008 실제 Label Studio에서 quickstart 확인 (클러스터 배포 뒤)
