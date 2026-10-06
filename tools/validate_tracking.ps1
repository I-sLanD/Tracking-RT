$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskConda = (Get-Command conda.exe -ErrorAction Stop).Source
$taskEnvironment = (& $taskConda env list --json | ConvertFrom-Json).envs | Where-Object { (Split-Path -Leaf $_) -eq 'jetson-perception' } | Select-Object -First 1
if (-not $taskEnvironment) { throw 'jetson-perception Conda environment is required.' }
& (Join-Path $PSScriptRoot 'build.ps1')
& (Join-Path $taskEnvironment 'Scripts\ctest.exe') --test-dir (Join-Path $taskRoot 'build') --output-on-failure
if ($LASTEXITCODE -ne 0) { throw 'DeepSORT C++ algorithm tests failed.' }
& (Join-Path $PSScriptRoot 'track.ps1') -Provider cpu -DumpFeatures -NoRender
& (Join-Path $PSScriptRoot 'track.ps1') -Provider dml -DumpFeatures
& (Join-Path $PSScriptRoot 'track.ps1') -Provider dml -Profile -OutputName 'tracking-dml-profile' -NoRender
& $taskConda run --no-capture-output -n jetson-perception python (Join-Path $PSScriptRoot 'validate_tracking.py')
if ($LASTEXITCODE -ne 0) { throw 'Independent tracking validation failed.' }
