# 집 PC에서 v001 만들기 (MVP, 11/4~5)

민서 집 PC(Windows, RTX 4070 Ti, `D:\trace`)에서 부스 사진 → 라벨 → 데이터셋 → 학습 → Jetson용 ONNX까지 한 번에 하는 순서입니다. 집 PC 준비(가상환경·CUDA torch·로컬 PostgreSQL `trace-ml-pg`·로컬 Label Studio 1.19.0·trace-trainer 키)는 11/2에 끝났습니다(docs `decisions.md` "클러스터를 멈춘 동안 집 PC는…").

MVP 범위: 부스 사진은 TOP·BOT 두 면, CAM1·CAM2. 클래스는 codes.yaml 그대로 4개(scratch·dent·crack·stain)이고, MVP 라벨은 주로 scratch입니다.

## 사진 없이 오늘 미리 하기 (리허설)

실사진이 오기 전에 `합성 사진 → 640 타일 → 학습 → ONNX → model.json → Jetson 엔진 → 탐지 시간`을 한 번 끝까지 돌립니다. 실사진이 들어오면 데이터만 바뀌고 나머지 배선은 이미 확인된 상태가 됩니다. MinIO·PostgreSQL·Label Studio를 쓰지 않고, 결과는 로컬 폴더에만 남습니다(trace-models에 올리지 않음).

```powershell
cd D:\trace\ml
git switch develop; git pull
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,train]"
trace-ml rehearse --out D:\trace\rehearsal --epochs 30     # 4070 Ti에서 몇 분
```

끝에 이렇게 나오면 통과입니다(숫자는 다를 수 있음).

```text
ONNX 부스 모양 확인: 통과(입력 [1, 3, 640, 640], 출력 [1, 40, 8400]·[1, 32, 160, 160])
Jetson으로 옮길 파일: D:\trace\rehearsal\model\best.onnx, D:\trace\rehearsal\model\model.json
```

- 합성 사진(1920×1080, 어두운 바닥 위 은색 판에 긁힘·찍힘)이라 mAP는 참고만 합니다. 0이 아니면 학습이 돌았다는 뜻입니다.
- `model.json`은 `testOnly: true`입니다. 부스가 이 모델로 찍은 탐지는 서버에 올리지 않습니다.
- `D:\trace\rehearsal\model\`에 `best.onnx`·`best.pt`·`model.json`·`metrics.json`·`sample_defect.jpg`·`sample_clean.jpg`가 생깁니다.

### Jetson(jetson2)에서 엔진 빌드·속도 재기

`model\` 폴더를 jetson2의 `~/rehearsal/`로 옮깁니다(scp 또는 USB). 그다음 jetson2에서:

```bash
export PATH="$HOME/trace-venv/bin:$PATH"          # 부스 앱 가상환경(booth capture/README.md)
trace-booth build-engine --onnx ~/rehearsal/best.onnx --out ~/trace-models/rehearsal   # FP16, 몇 분
trace-booth bench --engine ~/trace-models/rehearsal/best.engine \
  --images ~/rehearsal/sample_defect.jpg ~/rehearsal/sample_clean.jpg ~/rehearsal/sample_defect.jpg ~/rehearsal/sample_clean.jpg
```

`bench`가 `sample_defect`에서 흠집을 찾고 `→ OK`(목표 15초 안)를 내면 Jetson 쪽 배선도 끝입니다. 실사진 v001이 나오면 `--onnx`만 바꿔 같은 두 줄을 다시 칩니다(모델마다 `--out` 폴더를 따로).

### 오늘 밤 같이 해 둘 것

아래 두 명령은 Label Studio·MinIO·DB 환경 변수가 필요합니다. 같은 창에서 1번의 `ml.env` 불러오기를 먼저 실행합니다.

```powershell
Get-Content D:\trace\ml.env | Where-Object { $_ -match '^[A-Z_]+=.' } | ForEach-Object { $n, $v = $_ -split '=', 2; Set-Item "env:$n" $v.Trim() }
docker start label-studio; docker start trace-ml-pg
trace-ml labelstudio setup      # 프로젝트·trace-raw 소스·trace-labels 타깃. 이미 있으면 그대로
trace-ml dataset device list    # 기기 분할 확인(실사진 찍을 기기 코드가 정해지면 4번)
```

## 0. 코드 받기 (PowerShell)

```powershell
cd D:\trace\ml
git switch develop; git pull
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,train]"
python -c "import torch; print(torch.cuda.is_available())"   # True여야 한다
```

## 1. 환경 변수 불러오기

키 값은 `D:\trace\ml.env`(레포 밖)에만 적습니다. 채팅·코드·커밋에 쓰지 않습니다. 파일에 둘 이름:

```text
MINIO_ENDPOINT=http://storage.java21.net:8000
TRAINER_MINIO_ACCESS_KEY=        # trace-trainer
TRAINER_MINIO_SECRET_KEY=
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_API_KEY=            # Label Studio 화면 Account & Settings
LABEL_STUDIO_MINIO_ACCESS_KEY=   # trace-labelstudio
LABEL_STUDIO_MINIO_SECRET_KEY=
POSTGRES_HOST=localhost
DB_USERNAME=trace_ml
DB_PASSWORD=
```

`MINIO_ENDPOINT`는 TLS 없는 평문 HTTP입니다(학교 MinIO S3 API가 HTTP만 엽니다). 요청은 SigV4 서명이라 secret key는 가지 않지만 사진·라벨·모델 내용은 암호화되지 않으므로, 집 유선망처럼 믿을 수 있는 네트워크에서만 쓰고 공용 Wi-Fi에서는 쓰지 않습니다. 키는 최소 권한 trace-trainer만 씁니다. 학교가 HTTPS S3 주소를 열면 이 값만 바꿉니다.

새 PowerShell 창마다:

```powershell
Get-Content D:\trace\ml.env | Where-Object { $_ -match '^[A-Z_]+=.' } | ForEach-Object { $n, $v = $_ -split '=', 2; Set-Item "env:$n" $v.Trim() }
docker start trace-ml-pg        # 로컬 기록부
```

## 2. Label Studio에 사진 가져오기 (촬영 뒤)

```powershell
docker start label-studio       # 컨테이너 이름이 다르면 docker ps -a로 확인
trace-ml labelstudio setup      # 처음 한 번. 이미 있으면 아무것도 만들지 않는다
trace-ml labelstudio sync       # 촬영할 때마다: trace-raw 새 사진 → 작업
```

## 3. 라벨링 규칙

- 브라우저 `http://localhost:8080` → 프로젝트 "TRACE 흠집 라벨링".
- 흠집마다 폴리곤 하나, 흠집 모양을 따라 촘촘히. 클래스는 `scratch`(긁힘), 찍힘은 `dent`.
- 흠집이 없는 사진은 `no_defect`만 체크하고 제출합니다. 폴리곤과 `no_defect`를 같이 넣으면 빌드가 멈춥니다.
- 사진 안에서 흠집이 너무 작아 확대해도 안 보이면 그리지 않습니다(학습이 헷갈립니다).
- 다 끝나면 다시 `trace-ml labelstudio sync` (주석이 trace-labels로 나갑니다).

## 4. 기기 분할 정하기

한 기기의 사진은 train·val·test 중 한 곳에만 들어가고, 버전에 한 번 쓰인 기기는 분할을 바꿀 수 없습니다. 그래서 **첫 빌드 전에** 정합니다. val 기기가 없으면 `dataset build`가 아무것도 쓰지 않고 멈춥니다. 기기가 2~3대뿐인 MVP에서는 흠집이 많은 기기를 TRAIN, 하나를 VAL로 두고 TEST는 비워도 됩니다.

```powershell
trace-ml dataset device add REG01 --split TRAIN --model "MacBook Air M2"
trace-ml dataset device add REG02 --split VAL   --model "LG gram"
trace-ml dataset device list
```

## 5. 데이터셋 버전 v001

```powershell
git status                       # 커밋 안 한 변경이 있으면 빌드가 거절한다
trace-ml dataset build           # → v001 (덮어쓰기 없음. 고치려면 라벨을 고친 뒤 v002)
trace-ml dataset list
```

오류 목록이 나오면 그 작업 번호를 Label Studio에서 고치고 sync → build를 다시 합니다. 오류가 있으면 아무것도 쓰지 않으므로 v001 이름은 그대로 남아 있습니다.

## 6. 학습

```powershell
trace-ml train v001 --epochs 100 --cache D:\trace\cache
```

- 부스(BO2)가 원본 사진을 640 타일(겹침 0.2)로 잘라 추론하므로, 학습도 같은 규칙으로 자른 640 타일로 합니다(`--tile 640`이 기본). 1080p 사진 한 장이 타일 8개가 됩니다.
- train의 흠집 없는 타일은 흠집 타일 수만큼만 씁니다(`--background-ratio 1.0`). 오탐이 많으면 2.0으로 올려 v001을 다시 학습합니다(데이터셋은 그대로).
- 배치는 GPU 메모리에 맞춰 자동(`--batch -1`). 메모리 부족이 나면 `--batch 8`.
- 끝나면 `trace-models/v001/<실행>/`에 `best.pt`·`best.onnx`·`model.json`·`metrics.json`·`run.json`이 올라갑니다.

```powershell
trace-ml models list             # 데이터셋 버전 · mask mAP50
```

## 7. 눈으로 확인

```powershell
yolo segment predict model=D:\trace\cache\runs\<실행>\weights\best.pt source=D:\trace\cache\datasets\v001-tile640\images\val imgsz=640 conf=0.25
```

`runs\segment\predict*` 폴더의 사진에서 흠집에 폴리곤이 그려졌는지, 깨끗한 곳에 오탐이 없는지 봅니다.

## 8. Jetson으로

`best.onnx`는 입력 1×3×640×640 고정, opset 17, FP32입니다. jetson2가 trace-jetson 키(trace-models 읽기 전용)로 `best.onnx`(또는 `best.pt`)와 `model.json`을 받아 TensorRT FP16 엔진을 직접 빌드합니다. `model.json`은 엔진 옆에 그대로 둡니다(`modelVersion` = `yolo11s-seg-v001-trt-fp16`).

## 막히면

| 증상 | 할 일 |
| --- | --- |
| `PostgreSQL에 닿지 않음` | `docker start trace-ml-pg`, 환경 변수 다시 불러오기 |
| `환경 변수가 없음: …` | 1번 명령을 이 창에서 다시 실행 |
| `val 사진이 없음` | 4번에서 기기 하나를 VAL로 두고 다시 build |
| `CUDA out of memory` | `--batch 8` 또는 `--batch 4` |
| `사진이 바뀜` | trace-raw 사진이 빌드 뒤에 바뀜. 새 버전을 빌드 |
