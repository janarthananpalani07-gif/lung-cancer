$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$logPath = Join-Path $repoRoot "runs\continue_from_69_resume.log"
$modelPath = Join-Path $repoRoot "runs\detect\train_continue\weights\last.pt"
$yoloPath = Join-Path $repoRoot ".venv\Scripts\yolo.exe"

Set-Location $repoRoot
"[$(Get-Date -Format s)] Resuming training from $modelPath" | Tee-Object -FilePath $logPath -Append
& $yoloPath train resume "model=$modelPath" 2>&1 | Tee-Object -FilePath $logPath -Append
