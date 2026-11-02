# 빠른 확인: ML2 inspect-worker

```sh
python -m pip install -e ".[dev]"
ruff check . && pytest -q

# 사진 한 장 펼치기(오차·꼭짓점 출력)
trace-ml align photo top.jpg --face TOP --out top-rectified.png

# 합성 사진 정합 오차(px·mm), 폰 전체 면 + 부스 반쪽
trace-ml align eval --n 100 --json eval.json

# 같은 면의 부스·폰 사진 사이 정합 오차
trace-ml align pair booth-TOP-L1-CAM1.jpg phone-TOP.jpg --face TOP

# 워커(집 PC, PR 2/2에서 추가). 설정은 deploy/inspect-worker/worker.env.example
trace-ml inspect worker
```

결과 이벤트 키: 펼친 이미지 `trace-derived/rectified/{photoId}.png`, 텍스처 `trace-derived/textures/{captureSetId}/{face}.png`.
진행률: Redis 채널 `trace:progress:{rentalId}`에 `{"captureSetId","checkpointId","stage":"ALIGN|DONE","done","total"}`.
