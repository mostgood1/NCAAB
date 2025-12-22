param(
    [string]$Date = (Get-Date -Format "yyyy-MM-dd"),
    [int]$IntervalSec = 30
)

$ErrorActionPreference = "Stop"

# Use workspace root so relative paths resolve predictably
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root
try {
    $file = Join-Path $PWD "outputs/daily_results/results_$Date.csv"
    Write-Host "[watch] Watching $file"
    while ($true) {
        if (Test-Path $file) {
            try { $rows = (Import-Csv -Path $file).Count } catch { $rows = 0 }
            if ($rows -gt 0) {
                Write-Host "[action] Results detected for $Date ($rows rows). Uploading…"
                ./scripts/upload_artifacts_to_render.ps1 -Date $Date -Verbose
                Write-Host "[done] Results uploaded. Exiting watcher."
                break
            } else {
                Write-Host "[wait] File present but empty. Checking again in $IntervalSec s."
            }
        } else {
            Write-Host "[wait] File not found yet. Checking again in $IntervalSec s."
        }
        Start-Sleep -Seconds $IntervalSec
    }
}
finally {
    Pop-Location
}