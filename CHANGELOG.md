# 변경 기록

이 레포의 버전별 변경을 적는다. 형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.1.0/)를 따른다.
기능 PR은 `## [Unreleased]` 아래에 한 줄을 더하고, 릴리스 PR에서 `## [x.y.z] - 날짜`로 바꾼다(CONTRIBUTING.md 3번).
main에 버전이 적힌 CHANGELOG가 올라가면 release.yml이 태그와 GitHub Release를 만든다.

## [Unreleased]

### 추가
- 정합 보정(FR-INS-02): `TRACE_ALIGN=features`면 윤곽 4점 결과를 대여의 기준 이미지(`trace-derived/align-ref/`)에 특징점(모델 폴더에 `superpoint.onnx`·`lightglue.onnx`가 있으면 SuperPoint+LightGlue ONNX, 없으면 ORB) + ECC로 다듬는다. 기본 `contour`는 그대로
- inspect-worker 컨테이너(`Dockerfile`, GHCR `inspect-worker`): best.onnx를 onnxruntime CPU로 돌리는 폰 탐지(부스와 같은 후처리), 모델 폴더 `TRACE_MODEL_DIR`·탐지 문턱 `TRACE_DETECT_CONF`
- `trace-ml pretrain scan|build|train`: AI Hub 라벨 JSON(이미지별·labelme·COCO)을 클래스 표로 YOLO-seg 640 타일 데이터셋으로 바꾸고 사전 학습(부스 학습의 --base)
- `trace-ml rehearse`: 실사진 없이 합성 사진으로 640 타일 학습 → 고정 640 ONNX → testOnly model.json까지 끝까지 시험(ML1·BO2 배선 확인)
- 블로그(AIGJ-01-002-blog/devlog) CI/CD를 옮겨 초기 구성
