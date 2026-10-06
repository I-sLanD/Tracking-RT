param([ValidateSet('Debug','Release','RelWithDebInfo')][string]$Configuration = 'RelWithDebInfo')
$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path -Parent $PSScriptRoot
$taskConda = (Get-Command conda.exe -ErrorAction Stop).Source
$taskEnvironments = (& $taskConda env list --json | ConvertFrom-Json).envs
$taskEnvironment = $taskEnvironments | Where-Object { (Split-Path -Leaf $_) -eq 'jetson-perception' } | Select-Object -First 1
if (-not $taskEnvironment) { throw 'Create the jetson-perception Conda environment first.' }
$taskCmake = Join-Path $taskEnvironment 'Scripts\cmake.exe'
$taskNinja = Join-Path $taskEnvironment 'Scripts\ninja.exe'
$taskCompiler = Join-Path $taskProjectRoot 'third_party\llvm-mingw-20260922-ucrt-x86_64\bin\x86_64-w64-mingw32-clang++.exe'
$taskBuild = Join-Path $taskProjectRoot 'build'
& $taskCmake -S $taskProjectRoot -B $taskBuild -G Ninja "-DCMAKE_BUILD_TYPE=$Configuration" "-DCMAKE_CXX_COMPILER=$taskCompiler" "-DCMAKE_MAKE_PROGRAM=$taskNinja"
if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed.' }
& $taskCmake --build $taskBuild --parallel
if ($LASTEXITCODE -ne 0) { throw 'C++ build failed.' }
