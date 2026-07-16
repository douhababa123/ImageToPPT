$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv312\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
}

if (-not (Test-Path $Python)) {
    throw "未找到虚拟环境 Python。请先创建 .venv312 或 .venv，并安装 requirements.txt。"
}

Set-Location $ProjectRoot
& $Python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
