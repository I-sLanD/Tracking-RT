$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskConda = (Get-Command conda.exe -ErrorAction Stop).Source
& $taskConda run --no-capture-output -n jetson-perception python -m pip install scipy==1.15.3
if ($LASTEXITCODE -ne 0) { throw 'Reference validation dependency installation failed.' }
& $taskConda run --no-capture-output -n jetson-perception python (Join-Path $PSScriptRoot 'prepare_tracking.py')
if ($LASTEXITCODE -ne 0) { throw 'Tracking dependencies download failed.' }
& $taskConda run --no-capture-output -n jetson-perception python (Join-Path $PSScriptRoot 'export_reid.py')
if ($LASTEXITCODE -ne 0) { throw 'ReID export failed.' }
