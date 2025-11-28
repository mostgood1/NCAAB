# Runs the synthetic E2E harness and enforces CI thresholds.
# Exits non-zero on failure per the CLI's exit codes.

param(
    [string]$Date = (Get-Date -Format 'yyyy-MM-dd'),
    [int]$NGames = 8,
    [int]$Seed = 42,
    [float]$MinCalShareTotal = 0.6,
    [float]$MinCalShareMargin = 0.6,
    [int]$MinRows = 4,
    [float]$MaxMeanEdgeTotal = 20.0
)

$ErrorActionPreference = 'Stop'

# Activate venv if present
$venvActivate = Join-Path $PSScriptRoot '..' '.venv' 'Scripts' 'Activate.ps1'
if (Test-Path $venvActivate) {
    . $venvActivate
}

$python = Join-Path $PSScriptRoot '..' '.venv' 'Scripts' 'python.exe'
if (-not (Test-Path $python)) {
    $python = 'python'
}

$cmd = @(
    $python, '-m', 'ncaab_model.cli', 'synthetic-e2e',
    '--date', $Date,
    '--n-games', $NGames,
    '--seed', $Seed,
    '--min-cal-share-total', $MinCalShareTotal,
    '--min-cal-share-margin', $MinCalShareMargin,
    '--min-rows', $MinRows,
    '--max-mean-edge-total', $MaxMeanEdgeTotal
)

Write-Host "Running synthetic-e2e for $Date ..."
$proc = Start-Process -FilePath $cmd[0] -ArgumentList $cmd[1..($cmd.Length-1)] -NoNewWindow -Wait -PassThru

if ($proc.ExitCode -ne 0) {
    Write-Error "Synthetic E2E harness failed with exit code $($proc.ExitCode)."
    exit $proc.ExitCode
}

# Surface summary path to CI logs
$root = Split-Path $PSScriptRoot -Parent
$summary = Join-Path $root 'outputs' ("synthetic_e2e_{0}.json" -f $Date)
if (Test-Path $summary) {
    Write-Host "Summary: $summary"
    Get-Content $summary | Write-Output
} else {
    Write-Warning "Summary JSON not found: $summary"
}

Write-Host 'Synthetic E2E harness succeeded.'
