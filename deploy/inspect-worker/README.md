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
```

## 확인

- 워커 로그에 `inspect-worker 시작: 큐 trace.inspect.requested, 기종 MBA13-M2`
- 세트를 하나 올리면 `세트 … 처리: 정합 n, 실패 m, 흠집 k`
- 처리하지 못한 메시지는 RabbitMQ 큐 `trace.inspect.requested.dead`
