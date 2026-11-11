# 데이터 모델: ML1

## PostgreSQL `ml` (docs db/V1__init.sql 그대로, `trace-ml db init`이 IF NOT EXISTS로 만든다)

| 표 | 칸 | 규칙 |
| --- | --- | --- |
| dataset_device | device_code(고유), model_name, color, split(TRAIN·VAL·TEST) | 기기 하나 = 분할 하나. 버전에 쓰인 뒤 split 변경 금지(앱 검사) |
| dataset_version | version(고유, `v` + 숫자 3자리), manifest_key, builder_commit(40), counts(jsonb), built_at | 덮어쓰기 금지(고유 제약) |
| training_run | dataset_version_id(FK), base_weights, metrics(jsonb), model_key, onnx_key | 모델 → 데이터셋 버전 |

`counts` = `{"images": {"train": n, "val": n, "test": n}, "background": {...}, "instances": {"scratch": n, ...}, "devices": {"TRAIN": ["G01"], ...}, "manifestSha256": "..."}`

## MinIO

```text
trace-datasets/v001/manifest.json      # 마지막에 씀
trace-datasets/v001/data.yaml
trace-datasets/v001/labels/<split>/<사진 이름>.txt
trace-models/v001/<실행 이름>/best.pt · best.onnx · metrics.json · run.json
```

## manifest.json (`trace.dataset.v1`)

```json
{
  "schema": "trace.dataset.v1",
  "version": "v001",
  "builtAt": "2026-11-05T12:00:00Z",
  "builderCommit": "<40자>",
  "builderDirty": false,
  "codesVersion": "0.5.0",
  "classes": ["scratch", "dent", "crack", "stain"],
  "devices": {"G01": "TRAIN"},
  "counts": {},
  "items": [
    {"split": "train", "device": "G01", "taskId": 12, "annotationId": 30,
     "image": {"bucket": "trace-raw", "key": "G01/G01_TOP_L1_CAM1_01.jpg", "sha256": "..."},
     "label": {"key": "labels/train/G01_TOP_L1_CAM1_01.txt", "sha256": "...", "instances": 2}}
  ]
}
```
