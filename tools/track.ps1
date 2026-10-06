param(
  [ValidateSet('cpu','dml')][string]$Provider = 'dml',
  [string]$Video = 'assets\pedestrians.avi',
  [string]$Frames = 'assets\pedestrian-frames',
  [int]$Limit = 180,
  [string]$OutputName = '',
  [switch]$DumpFeatures,
  [switch]$Profile,
  [switch]$NoRender,
  [int]$MaxAge = 30,
  [int]$NInit = 3,
  [double]$MaxCosine = 0.2,
  [double]$MaxIou = 0.7,
  [double]$Threshold = 0.25,
  [int]$NNBudget = 100
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskConda = (Get-Command conda.exe -ErrorAction Stop).Source
$taskVideo = if ([IO.Path]::IsPathRooted($Video)) { $Video } else { Join-Path $taskRoot $Video }
$taskFrames = if ([IO.Path]::IsPathRooted($Frames)) { $Frames } else { Join-Path $taskRoot $Frames }
$taskName = if ($OutputName) { $OutputName } else { "tracking-$Provider" }
$taskOutput = Join-Path $taskRoot "output\$taskName"
& $taskConda run --no-capture-output -n jetson-perception python (Join-Path $PSScriptRoot 'video_frames.py') prepare --video $taskVideo --frames $taskFrames --limit $Limit
if ($LASTEXITCODE -ne 0) { throw 'Video frame preparation failed.' }
$taskMetadata = Get-Content -LiteralPath (Join-Path $taskFrames 'frames.json') -Raw | ConvertFrom-Json
$taskCulture = [Globalization.CultureInfo]::InvariantCulture
$taskArguments = @(
  '--model', (Join-Path $taskRoot 'models\yolo26n.onnx'),
  '--reid', (Join-Path $taskRoot 'models\osnet_x0_25_msmt17.onnx'),
  '--runtime', (Join-Path $taskRoot 'third_party\onnxruntime-directml\runtimes\win-x64\native\onnxruntime.dll'),
  '--frames', $taskFrames, '--output', $taskOutput, '--provider', $Provider,
  '--fps', $taskMetadata.fps.ToString($taskCulture), '--limit', $Limit.ToString(),
  '--max-age', $MaxAge.ToString(), '--n-init', $NInit.ToString(),
  '--max-cosine', $MaxCosine.ToString($taskCulture), '--max-iou', $MaxIou.ToString($taskCulture),
  '--threshold', $Threshold.ToString($taskCulture), '--nn-budget', $NNBudget.ToString()
)
if ($DumpFeatures) { $taskArguments += '--dump-features' }
if ($Profile) { $taskArguments += '--profile' }
& (Join-Path $taskRoot 'build\yolo26_track.exe') @taskArguments
if ($LASTEXITCODE -ne 0) { throw 'Native tracking failed.' }
if (-not $NoRender) {
  & $taskConda run --no-capture-output -n jetson-perception python (Join-Path $PSScriptRoot 'video_frames.py') render --frames $taskFrames --output $taskOutput
  if ($LASTEXITCODE -ne 0) { throw 'Tracking video rendering failed.' }
}
