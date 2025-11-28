# Runs the synthetic end-to-end harness nightly (intended for Windows Task Scheduler)
# Usage: schedule this script to run daily after finalization window.

param(
    [string]$WorkspaceRoot = "C:/Users/mostg/OneDrive/Coding/NCAAB",
    [string]$PythonExe = "C:/Users/mostg/OneDrive/Coding/NCAAB/.venv/Scripts/python.exe",
    [string]$LogDir = "C:/Users/mostg/OneDrive/Coding/NCAAB/outputs/logs"
)

try {
    if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
    $dateStr = Get-Date -Format "yyyy-MM-dd"
    $logPath = Join-Path $LogDir ("e2e_nightly_" + $dateStr + ".log")

    Push-Location $WorkspaceRoot
    "$dateStr Running synthetic E2E harness..." | Out-File -FilePath $logPath -Append

    $scriptPath = Join-Path $WorkspaceRoot "scripts/synthetic_e2e_harness.py"
    $result = & $PythonExe $scriptPath
    $result | Out-String | Out-File -FilePath $logPath -Append

    "$dateStr Completed" | Out-File -FilePath $logPath -Append
} catch {
    "ERROR: $($_.Exception.Message)" | Out-File -FilePath $logPath -Append
} finally {
    Pop-Location
}
