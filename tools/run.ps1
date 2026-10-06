param([ValidateSet('cpu','dml')][string]$Provider = 'cpu', [int]$Iterations = 30, [switch]$Profile, [string]$Image = 'assets\bus.jpg', [string]$OutputName = '')
$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path -Parent $PSScriptRoot
$taskExecutable = Join-Path $taskProjectRoot 'build\yolo26_verify.exe'
$taskRuntime = Join-Path $taskProjectRoot 'third_party\onnxruntime-directml\runtimes\win-x64\native\onnxruntime.dll'
$taskOutputName = if ($OutputName) { $OutputName } elseif ($Profile) { "$Provider-profile" } else { $Provider }
$taskImagePath = if ([IO.Path]::IsPathRooted($Image)) { $Image } else { Join-Path $taskProjectRoot $Image }
$taskArguments = @('--model', (Join-Path $taskProjectRoot 'models\yolo26n.onnx'), '--image', $taskImagePath, '--runtime', $taskRuntime, '--labels', (Join-Path $taskProjectRoot 'models\classes.txt'), '--output', (Join-Path $taskProjectRoot "output\$taskOutputName"), '--provider', $Provider, '--iterations', $Iterations.ToString())
if ($Profile) { $taskArguments += '--profile' }
& $taskExecutable @taskArguments
if ($LASTEXITCODE -ne 0) { throw "Native $Provider validation failed." }
