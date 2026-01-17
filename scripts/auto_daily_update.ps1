#Requires -Version 5.1
<#
.SYNOPSIS
  Automated local runner for scripts/daily_update.ps1.

.DESCRIPTION
  - Runs the daily pipeline for a specified date (default: today).
  - Leaves git commit+push enabled (daily_update.ps1 handles it).
  - Writes a transcript log under outputs/logs.
  - Optional "mode" presets for morning vs pre-tip refresh.

  Notes:
  - This script assumes the repo root is the parent directory of scripts/.
  - For unattended runs, ensure git auth is configured (Git Credential Manager or SSH keys).

.EXAMPLE
  powershell.exe -ExecutionPolicy Bypass -File scripts/auto_daily_update.ps1

.EXAMPLE
  powershell.exe -ExecutionPolicy Bypass -File scripts/auto_daily_update.ps1 -Mode Pretip

.EXAMPLE
  powershell.exe -ExecutionPolicy Bypass -File scripts/auto_daily_update.ps1 -Today 2026-01-16 -Mode Morning -SkipHeavyQuantiles
#>

param(
  [string]$Today = $(Get-Date -Format 'yyyy-MM-dd'),
  [ValidateSet('Morning','Pretip','Custom')]
  [string]$Mode = 'Morning',

  # Common knobs
  [string]$Region = 'us',
  [string]$Provider = 'espn',
  [switch]$NoCache,
  [switch]$SkipHeavyQuantiles,

  # Morning: finalize yesterday + (optionally) retrain
  [switch]$SkipFinalizePrev,
  [switch]$SkipRetrain,

  # Pretip: typically no finalize, no retrain, just refresh odds/picks
  [switch]$PretipSkipFinalizePrev,
  [switch]$PretipSkipRetrain,

  # Safety
  [switch]$DryRun,
  [int]$MaxRetries = 1
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$dailyUpdate = Join-Path $repoRoot 'scripts\daily_update.ps1'
if (-not (Test-Path $dailyUpdate)) {
  throw "daily_update.ps1 not found at: $dailyUpdate"
}

$outDir = Join-Path $repoRoot 'outputs'
$logsDir = Join-Path $outDir 'logs'
New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logPath = Join-Path $logsDir ("auto_daily_update_${Today}_${Mode}_${stamp}.log")

function Invoke-DailyUpdate([int]$attempt) {
  $args = @('-ExecutionPolicy','Bypass','-File', $dailyUpdate, '-Today', $Today, '-Region', $Region, '-Provider', $Provider)

  if ($NoCache.IsPresent) { $args += '-NoCache' }

  # Preset behavior
  if ($Mode -eq 'Morning') {
    if ($SkipHeavyQuantiles.IsPresent) { $args += '-SkipHeavyQuantiles' }
    if ($SkipFinalizePrev.IsPresent) { $args += '-SkipFinalizePrev' }
    if ($SkipRetrain.IsPresent) { $args += '-SkipRetrain' }
  }
  elseif ($Mode -eq 'Pretip') {
    # Default pre-tip behavior: fast refresh
    $args += '-SkipHeavyQuantiles'
    $args += '-SkipRetrain'
    $args += '-SkipFinalizePrev'

    if ($SkipHeavyQuantiles.IsPresent) { } # already included

    # Allow overrides
    if ($PretipSkipRetrain.IsPresent) { }  # already included
    if ($PretipSkipFinalizePrev.IsPresent) { } # already included
  }
  else {
    # Custom: only pass explicitly provided switches
    if ($SkipHeavyQuantiles.IsPresent) { $args += '-SkipHeavyQuantiles' }
    if ($SkipFinalizePrev.IsPresent) { $args += '-SkipFinalizePrev' }
    if ($SkipRetrain.IsPresent) { $args += '-SkipRetrain' }
  }

  $cmd = "powershell.exe " + ($args | ForEach-Object { if ($_ -match '\\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
  Write-Host "[auto_daily_update] attempt=$attempt cmd=$cmd" -ForegroundColor Cyan

  if ($DryRun.IsPresent) {
    Write-Host '[auto_daily_update] DryRun enabled; not executing.' -ForegroundColor Yellow
    return 0
  }

  return (Start-Process -FilePath 'powershell.exe' -ArgumentList $args -Wait -PassThru).ExitCode
}

try {
  Start-Transcript -Path $logPath -Append | Out-Null
} catch {
  Write-Warning "Transcript start failed: $($_)"
}

try {
  Set-Location $repoRoot

  # Force simulations to use feature-derived means (strict, no fallback to model/blend).
  # This keeps the sim engine independent across both manual and scheduled runs.
  $env:NCAAB_SIM_MEAN_SOURCE = 'features_strict'

  $exitCode = 1
  for ($attempt = 1; $attempt -le (1 + [Math]::Max(0,$MaxRetries)); $attempt++) {
    $exitCode = Invoke-DailyUpdate -attempt $attempt
    if ($exitCode -eq 0) { break }
    Write-Warning "daily_update.ps1 failed with exitCode=$exitCode (attempt $attempt)"
  }

  if ($exitCode -ne 0) {
    throw "auto_daily_update failed after retries; exitCode=$exitCode"
  }

  Write-Host "[auto_daily_update] Success. Log: $logPath" -ForegroundColor Green
  exit 0
} finally {
  try { Stop-Transcript | Out-Null } catch {}
}
