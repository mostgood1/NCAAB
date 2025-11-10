param(
    [Parameter(Mandatory=$true)] [string] $OrtBinDir,
    [Parameter(Mandatory=$false)] [string] $OrtPyWheelOrDir = "",
    [Parameter(Mandatory=$false)] [string] $QnnSdkDir = "C:\Qualcomm\QNN_SDK",
    [Parameter(Mandatory=$false)] [switch] $InstallWheel,
    [Parameter(Mandatory=$false)] [string] $Python = "C:/Users/mostg/OneDrive/Coding/NCAAB/.venv/Scripts/python.exe"
)

Write-Host "Configuring ONNX Runtime (QNN) environment..."

if (-not (Test-Path -Path $OrtBinDir -PathType Container)) {
    Write-Error "OrtBinDir not found: $OrtBinDir"
    exit 1
}

$env:NCAAB_ORT_DLL_DIR = $OrtBinDir
Write-Host "Set NCAAB_ORT_DLL_DIR=$env:NCAAB_ORT_DLL_DIR"

if ($OrtPyWheelOrDir -ne "") {
    if (Test-Path -Path $OrtPyWheelOrDir) {
        $env:NCAAB_ORT_PY_DIR = $OrtPyWheelOrDir
        Write-Host "Set NCAAB_ORT_PY_DIR=$env:NCAAB_ORT_PY_DIR"
        if ($InstallWheel -and ($OrtPyWheelOrDir.ToLower().EndsWith('.whl'))) {
            Write-Host "Installing local onnxruntime wheel..."
            & $Python -m pip install --no-index --find-links $(Split-Path -Parent $OrtPyWheelOrDir) $OrtPyWheelOrDir
        }
    } else {
        Write-Warning "OrtPyWheelOrDir does not exist: $OrtPyWheelOrDir"
    }
}

if (Test-Path -Path $QnnSdkDir -PathType Container) {
    $env:NCAAB_QNN_SDK_DIR = $QnnSdkDir
    Write-Host "Set NCAAB_QNN_SDK_DIR=$env:NCAAB_QNN_SDK_DIR"
    # Auto-pick backend DLL variant (robust join without array misuse)
    $backendSubpaths = @(
        "lib/arm64x-windows-msvc/QnnHtp.dll",
        "lib/aarch64-windows-msvc/QnnHtp.dll",
        "lib/windows-aarch64/QnnHtp.dll"
    )
    $backend = $null
    foreach($sub in $backendSubpaths){
        $candidate = Join-Path $QnnSdkDir $sub
        if(Test-Path $candidate){ $backend = $candidate; break }
    }
    if($backend){
        $env:ORT_QNN_BACKEND_PATH = $backend
        Write-Host "Set ORT_QNN_BACKEND_PATH=$env:ORT_QNN_BACKEND_PATH"
    } else {
        Write-Warning "Could not locate QnnHtp.dll under expected subdirectories in $QnnSdkDir. Adjust paths or install QNN SDK with Windows ARM64 libs."
    }
} else {
    Write-Warning "QNN SDK dir not found: $QnnSdkDir"
}

Write-Host "Verifying providers via ort-info..."
& $Python -m ncaab_model.cli ort-info
Write-Host "If QNNExecutionProvider missing, ensure onnxruntime build includes QNN EP and DLLs are in OrtBinDir."
