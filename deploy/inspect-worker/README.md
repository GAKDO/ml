# inspect-worker 집 PC 설치 (ML2)

집 PC가 켜지면 inspect-worker가 자동으로 떠서, 꺼져 있던 동안 큐 `trace.inspect.requested`에 쌓인 촬영 세트를 처리합니다(FR-OPS-01). 큐는 durable이라 메시지가 사라지지 않습니다.

## 공통

```sh
git clone https://github.com/GAKDO/ml.git ~/ml && cd ~/ml
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -e .
cp deploy/inspect-worker/worker.env.example deploy/inspect-worker/worker.env   # 값을 채운다(커밋 금지)
```

폰 흠집 탐지까지 하려면(11/2 결정, A 담당) `python -m pip install -e ".[train]"`(ultralytics, CUDA torch)로 깔고, 학습 결과 `trace-models/<버전>/<실행>/best.pt`를 받아 같은 폴더에 부스와 같은 `model.json`(`{"modelVersion": "...", "classes": ["scratch","dent","crack","stain"]}`)을 둔 뒤 `TRACE_DETECT_WEIGHTS`에 경로를 적습니다. 비우면 폰 탐지는 건너뜁니다(modelVersion `detector-pending+contour-v1`).

먼저 손으로 한 번 띄워 봅니다: `set -a; . deploy/inspect-worker/worker.env; set +a; trace-ml inspect worker --idle 30`

## 학교 서버 Compose에 붙기 (SSH 터널, 11/2)

클러스터 작업이 멈춘 동안 워커는 학교 서버(`10.116.64.12`) Compose의 RabbitMQ(`127.0.0.1:5672`)와 MinIO(`127.0.0.1:19000`)에 붙습니다. 두 포트는 밖에 열지 않고 SSH 터널로 집 PC의 `localhost:5672`·`localhost:9000`에 잇습니다. 실습 서버는 `s2.java21.net`, SSH 포트는 8822입니다.

```powershell
ssh-keygen -t ed25519      # 한 번. 암호는 비움(자동 시작 터널이 묻지 않게)
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh -p 8822 <계정>@s2.java21.net "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

터널 스크립트는 계정 이름이 들어가므로 레포 밖(예: `D:\trace\tunnel.ps1`)에 둡니다:

```powershell
while ($true) {
    ssh -N -p 8822 -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes `
        -L 5672:127.0.0.1:5672 -L 9000:127.0.0.1:19000 <계정>@s2.java21.net
    Start-Sleep -Seconds 10
}
```

`worker.env`에는 `RABBITMQ_HOST=localhost`, `MINIO_ENDPOINT=http://localhost:9000`을 적고, RabbitMQ 계정과 MinIO 키는 서버 `~/gitops/compose/.env`의 `RABBITMQ_USERNAME`·`RABBITMQ_PASSWORD`, `MINIO_ACCESS_KEY`·`MINIO_SECRET_KEY` 값을 씁니다. Windows PowerShell에서 손으로 띄울 때는 다음을 실행합니다: `Get-Content deploy\inspect-worker\worker.env | Where-Object { $_ -match '^[A-Z_]+=' } | ForEach-Object { $n, $v = $_ -split '=', 2; Set-Item "env:$n" $v }; trace-ml inspect worker --idle 30`

## Linux (systemd 사용자 서비스)

```sh
mkdir -p ~/.config/systemd/user
cp deploy/inspect-worker/trace-inspect-worker.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now trace-inspect-worker
loginctl enable-linger "$USER"     # 로그인하지 않아도 부팅 때 뜨게
journalctl --user -u trace-inspect-worker -f
```

## Windows (작업 스케줄러)

PowerShell(관리자 아님)에서:

```powershell
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File $HOME\ml\deploy\inspect-worker\start-worker.ps1"
$trigger = New-ScheduledTaskTrigger -AtLogOn
Register-ScheduledTask -TaskName "TRACE inspect-worker" -Action $action -Trigger $trigger
# 터널도 같은 방식으로 등록한다(-File D:\trace\tunnel.ps1, 작업 이름 "TRACE ssh tunnel")
```

레포를 `$HOME\ml`이 아닌 곳(예: `D:\trace\ml`)에 받았으면 `-File` 경로를 그 위치로 바꿉니다.

## 확인

- 워커 로그에 `inspect-worker 시작: 큐 trace.inspect.requested, 기종 MBA13-M2`
- 세트를 하나 올리면 `세트 … 처리: 정합 n, 실패 m, 흠집 k`
- 처리하지 못한 메시지는 RabbitMQ 큐 `trace.inspect.requested.dead`
