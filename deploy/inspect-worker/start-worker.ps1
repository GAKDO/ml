# 집 PC(Windows)에서 로그인할 때 inspect-worker를 띄운다 (FR-OPS-01).
# 작업 스케줄러 등록: README의 "Windows" 절. worker.env의 값을 읽어 환경 변수로 넣고, 끝나면 10초 뒤 다시 띄운다.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Get-Content (Join-Path $PSScriptRoot "worker.env") | Where-Object { $_ -match '^[A-Z_]+=' } | ForEach-Object {
    $name, $value = $_ -split '=', 2
    [Environment]::SetEnvironmentVariable($name, $value, "Process")
}
$exe = Join-Path $root ".venv\Scripts\trace-ml.exe"
while ($true) {
    & $exe inspect worker
    Start-Sleep -Seconds 10
}
