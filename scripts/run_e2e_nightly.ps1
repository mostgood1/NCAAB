# Runs the synthetic end-to-end harness nightly (intended for Windows Task Scheduler)
# Usage: schedule this script to run daily after finalization window.

param(
    [string]$WorkspaceRoot = "",
    [string]$PythonExe = "",
    [string]$LogDir = ""
)

try {
    $pushed = $false
    $repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..") | Select-Object -ExpandProperty Path
    if ([string]::IsNullOrWhiteSpace($WorkspaceRoot)) { $WorkspaceRoot = $repoRoot }
    if ([string]::IsNullOrWhiteSpace($LogDir)) { $LogDir = Join-Path $WorkspaceRoot "outputs\\logs" }

    if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
    $dateStr = Get-Date -Format "yyyy-MM-dd"
    $logPath = Join-Path $LogDir ("e2e_nightly_" + $dateStr + ".log")

    if ([string]::IsNullOrWhiteSpace($PythonExe)) { $PythonExe = Join-Path $WorkspaceRoot ".venv\\Scripts\\python.exe" }
    if (-not (Test-Path $PythonExe)) {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($cmd) { $PythonExe = $cmd.Source }
    }
    if (-not (Test-Path $PythonExe)) { throw "Python executable not found. Create .venv in repo root or pass -PythonExe." }

    if (-not (Test-Path $WorkspaceRoot)) { throw "WorkspaceRoot not found: $WorkspaceRoot" }

    Push-Location $WorkspaceRoot
    $pushed = $true
    "$dateStr Running synthetic E2E harness..." | Out-File -FilePath $logPath -Append

    $scriptPath = Join-Path $WorkspaceRoot "scripts/synthetic_e2e_harness.py"
    $result = & $PythonExe $scriptPath
    $result | Out-String | Out-File -FilePath $logPath -Append

    "$dateStr Completed" | Out-File -FilePath $logPath -Append
} catch {
    "ERROR: $($_.Exception.Message)" | Out-File -FilePath $logPath -Append
} finally {
    if ($pushed) { Pop-Location }
}
