param(
  [string]$SrcRoot = "C:\dev",
  [string]$BuildDir = "C:\onnxruntime_build",
  [string]$QnnSdkDir = "C:\Qualcomm\QNN_SDK",
  [string]$Config = "Release",
  [string]$Python = "C:/Users/mostg/OneDrive/Coding/NCAAB/.venv/Scripts/python.exe",
  [string]$CMakeExe = "",
  [switch]$SkipClone,
  [switch]$UseCMakeDirect
)

$ErrorActionPreference = 'Stop'

Write-Host "=== ONNX Runtime QNN Build Script ===" -ForegroundColor Cyan
Write-Host "SrcRoot=$SrcRoot"; Write-Host "BuildDir=$BuildDir"; Write-Host "QnnSdkDir=$QnnSdkDir"; Write-Host "Config=$Config"

# Ensure folders
if(!(Test-Path $SrcRoot)){ New-Item -ItemType Directory -Force -Path $SrcRoot | Out-Null }
if(!(Test-Path $BuildDir)){ New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null }

$repoDir = Join-Path $SrcRoot "onnxruntime"
if(-not $SkipClone){
  if(!(Test-Path $repoDir)){
    Write-Host "Cloning onnxruntime repo..." -ForegroundColor Yellow
    git clone https://github.com/microsoft/onnxruntime.git $repoDir
  } else {
    Write-Host "Repo already exists at $repoDir" -ForegroundColor Yellow
  }
}

if(!(Test-Path $repoDir)){
  Write-Error "onnxruntime repo not found at $repoDir"
}

# Export QNN SDK root for build.py
$env:QNN_SDK_ROOT = $QnnSdkDir

if(-not $UseCMakeDirect){
  # build.py approach (omit unsupported flags; pass QNN via cmake extra defines)
  $buildPy = Join-Path $repoDir "tools/ci_build/build.py"
  if(!(Test-Path $buildPy)){ Write-Error "build.py not found: $buildPy" }
  # Ensure cmake is discoverable for build.py
  $cmakeDirCandidate = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
  if(Test-Path $cmakeDirCandidate){ $env:PATH = "$cmakeDirCandidate;$env:PATH" }
  $cmakeDefines = @(
      "onnxruntime_BUILD_UNIT_TESTS=OFF",
      "onnxruntime_USE_QNN=ON",
      # Correct variable consumed by onnxruntime_providers_qnn.cmake for includes
      "onnxruntime_QNN_HOME=$QnnSdkDir",
      # Optional legacy var (harmless) if future scripts reference it
      "QNN_SDK_ROOT=$QnnSdkDir",
      "ONNX_USE_PROTOBUF_SHARED_LIBS=ON",
      "protobuf_MSVC_STATIC_RUNTIME=OFF",
      "protobuf_BUILD_SHARED_LIBS=ON"
    )
  $cmakeDefinesStr = ($cmakeDefines -join ";")
  $argsList = @(
    "--update",
    "--build",
    "--config", $Config,
    "--build_shared_lib",
    "--enable_pybind",
    "--build_wheel",
    "--use_qnn",
    "--qnn_home", $QnnSdkDir,
    "--parallel",
    "--build_dir", $BuildDir,
    "--cmake_extra_defines", $cmakeDefinesStr
  )
  Write-Host "Running build.py: $Python $buildPy $($argsList -join ' ')" -ForegroundColor Yellow
  & $Python $buildPy @argsList
} else {
  # Direct cmake invocation (fallback if build.py rejects QNN flags)
  if([string]::IsNullOrWhiteSpace($CMakeExe)){
    $cmakeCmd = Get-Command cmake -ErrorAction SilentlyContinue
    if($cmakeCmd){ $CMakeExe = $cmakeCmd.Source }
  }
  if([string]::IsNullOrWhiteSpace($CMakeExe)){
    $vsCMake = Get-ChildItem -Path "C:\Program Files\Microsoft Visual Studio\2022" -Recurse -Filter cmake.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
    if($vsCMake){ $CMakeExe = $vsCMake }
  }
  if([string]::IsNullOrWhiteSpace($CMakeExe) -or -not (Test-Path $CMakeExe)){
    Write-Error "Could not locate cmake executable. Provide -CMakeExe path."; exit 1
  }
  Write-Host "Using CMakeExe=$CMakeExe" -ForegroundColor Cyan
  $src = Join-Path $repoDir "cmake"
  $cmakeArgs = @(
    "-S", $src,
    "-B", $BuildDir,
    "-G", "Visual Studio 17 2022",
    "-A", "ARM64",
    "-DCMAKE_BUILD_TYPE=$Config",
    "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL",
    "-Donnxruntime_BUILD_UNIT_TESTS=OFF",
    "-Donnxruntime_BUILD_SHARED_LIB=ON",
    "-Donnxruntime_USE_QNN=ON",
    # Provide the variable actually read for header/include paths
    "-Donnxruntime_QNN_HOME=$QnnSdkDir",
    # Keep legacy variable for completeness (not strictly required)
    "-DQNN_SDK_ROOT=$QnnSdkDir",
    "-DONNX_USE_PROTOBUF_SHARED_LIBS=ON",
    "-Dprotobuf_MSVC_STATIC_RUNTIME=OFF",
    "-Dprotobuf_BUILD_SHARED_LIBS=ON"
  )
  Write-Host "Configuring with CMake: $CMakeExe $($cmakeArgs -join ' ')" -ForegroundColor Yellow
  & "$CMakeExe" @cmakeArgs
  if($LASTEXITCODE -ne 0){ Write-Error "CMake configure failed."; exit 1 }
  Write-Host "Building with CMake --build ..." -ForegroundColor Yellow
  & "$CMakeExe" --build $BuildDir --config $Config --parallel
}

# After build, try to locate artifacts
$binCandidate = Join-Path $BuildDir $Config
$wheelDir = Join-Path $BuildDir (Join-Path $Config "dist")
$wheel = $null
if(Test-Path $wheelDir){
  $wheels = Get-ChildItem -Path $wheelDir -Filter "*.whl" | Sort-Object LastWriteTime -Descending
  if($wheels){ $wheel = $wheels[0].FullName }
}

if(!(Test-Path $binCandidate)){
  Write-Warning "Did not find bin dir at $binCandidate. Check build output logs above."
} else {
  Write-Host "Built DLLs likely under: $binCandidate" -ForegroundColor Green
}
if($wheel){
  Write-Host "Found wheel: $wheel" -ForegroundColor Green
}

# Attempt to set runtime env now using helper
try{
  $enableScript = Join-Path (Split-Path $PSCommandPath -Parent) "enable_ort_qnn.ps1"
  if(Test-Path $enableScript -and (Test-Path $binCandidate)){
    Write-Host "Configuring runtime environment with enable_ort_qnn.ps1..." -ForegroundColor Cyan
    if($wheel){
      & powershell -NoProfile -ExecutionPolicy Bypass -File $enableScript -OrtBinDir $binCandidate -OrtPyWheelOrDir $wheel -QnnSdkDir $QnnSdkDir -Python $Python
    } else {
      & powershell -NoProfile -ExecutionPolicy Bypass -File $enableScript -OrtBinDir $binCandidate -QnnSdkDir $QnnSdkDir -Python $Python
    }
  } else {
    Write-Warning "Skipping auto-config (enable script or binCandidate missing)."
  }
} catch {
  Write-Warning "enable_ort_qnn.ps1 invocation failed: $($_.Exception.Message)"
}

Write-Host "Build script completed. Run diagnostics:" -ForegroundColor Cyan
Write-Host ".\\.venv\\Scripts\\python.exe -m ncaab_model.cli ort-diagnostics"
Write-Host "If providers list is empty, retry with -UseCMakeDirect or inspect build logs for QNN support." -ForegroundColor Yellow
