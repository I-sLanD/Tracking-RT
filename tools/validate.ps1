$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'build.ps1')
& (Join-Path $PSScriptRoot 'run.ps1') -Provider cpu
& (Join-Path $PSScriptRoot 'run.ps1') -Provider dml
& (Join-Path $PSScriptRoot 'run.ps1') -Provider cpu -Image 'assets\zidane.jpg' -OutputName 'cpu-zidane' -Iterations 10
& (Join-Path $PSScriptRoot 'run.ps1') -Provider dml -Image 'assets\zidane.jpg' -OutputName 'dml-zidane' -Iterations 10
& (Join-Path $PSScriptRoot 'run.ps1') -Provider dml -Profile -Iterations 1
& conda.exe run --no-capture-output -n jetson-perception python (Join-Path $PSScriptRoot 'validate.py')
if ($LASTEXITCODE -ne 0) { throw 'Reference comparison failed.' }
