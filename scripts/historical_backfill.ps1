#Requires -Version 5.1
<#
Historical backfill orchestrator for NCAAB project (Windows PowerShell).

Usage examples:
  # Backfill a specific span, rebuild last odds, artifacts, calibration, and season metrics
  .\scripts\historical_backfill.ps1 -Start 2023-11-01 -End 2024-04-15

  # Backfill by season(s) using typical college season window (Nov 1 -> Apr 15 next year)
  .\scripts\historical_backfill.ps1 -Seasons 2022,2023,2024

Parameters:
  -Start / -End   Inclusive ISO dates (YYYY-MM-DD). If provided, overrides -Seasons.
  -Seasons        One or more season years. Each season uses [Nov 1, season] .. [Apr 15, season+1].
  -Provider       Games provider: espn|ncaa|fused (default: espn)
  -Region         TheOddsAPI region (default: us)
  -NoCache        Skip provider caches when fetching games (default: false)
  -EnableOrt      Enable ORT/QNN activation during backfill (default: false for speed)
#>
param(
  [string]$Start,
  [string]$End,
  [int[]]$Seasons,
  [string]$Provider = 'espn',
  [string]$Region = 'us',
  [switch]$NoCache,
  [switch]$EnableOrt
)

$ErrorActionPreference = 'Stop'

function Write-Section($msg) { Write-Host "`n==== $msg ====\n" }

# Resolve repo context
$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $VenvPython)) { throw "Python venv not found at $VenvPython. Create and install deps first." }
$OutDir = Join-Path $RepoRoot 'outputs'
$HistDir = Join-Path $OutDir 'odds_history'
New-Item -ItemType Directory -Path $HistDir -Force | Out-Null

Set-Location $RepoRoot

# Build date ranges
$ranges = @()
if ($Start -and $End) {
  $ranges += [PSCustomObject]@{ Start=[DateTime]::ParseExact($Start,'yyyy-MM-dd',$null); End=[DateTime]::ParseExact($End,'yyyy-MM-dd',$null) }
} elseif ($Seasons -and $Seasons.Count -gt 0) {
  foreach ($s in $Seasons) {
    $d0 = Get-Date -Date ("$s-11-01")
    $d1 = Get-Date -Date (([int]$s + 1).ToString() + '-04-15')
    $ranges += [PSCustomObject]@{ Start=$d0; End=$d1 }
  }
} else {
  throw 'Provide either -Start/-End or -Seasons.'
}

# Track global min/max for follow-up steps
$minStart = $null; $maxEnd = $null

# 1) Backfill daily pipeline per range
foreach ($r in $ranges) {
  $sIso = $r.Start.ToString('yyyy-MM-dd'); $eIso = $r.End.ToString('yyyy-MM-dd')
  if (-not $minStart -or $r.Start -lt $minStart) { $minStart = $r.Start }
  if (-not $maxEnd -or $r.End -gt $maxEnd) { $maxEnd = $r.End }
  Write-Section "Backfill range $sIso .. $eIso"
  $cacheFlag = @(); if ($NoCache.IsPresent) { $cacheFlag += '--no-use-cache' }
  $ortFlag = @(); if ($EnableOrt.IsPresent) { $ortFlag += '--enable-ort' } else { $ortFlag += '--no-enable-ort' }
  & $VenvPython -m ncaab_model.cli backfill-range $sIso $eIso --provider $Provider --region $Region @cacheFlag @ortFlag --accumulate-schedule --accumulate-predictions
}

# 2) Build last-odds master merge across full span
$minIso = $minStart.ToString('yyyy-MM-dd'); $maxIso = $maxEnd.ToString('yyyy-MM-dd')
Write-Section "Rebuild last odds master from $minIso .. $maxIso"
& $VenvPython -m ncaab_model.cli backfill-last-odds $minIso $maxIso --tolerance-seconds 60 --filter-mode any --allow-partial

# 3) Build closing lines from odds_history and join per-date
Write-Section 'Compute closing lines from historical snapshots'
& $VenvPython -m ncaab_model.cli make-closing-lines --in-dir $HistDir --out (Join-Path $OutDir 'closing_lines.csv')
# Per-day joins for reference (optional): rely on scripts/closing_join.py which writes games_with_closing_<date>.csv
Write-Section 'Per-day closing joins'
$cur = $minStart
while ($cur -le $maxEnd) {
  $iso = $cur.ToString('yyyy-MM-dd')
  & $VenvPython scripts/closing_join.py --date $iso
  $cur = $cur.AddDays(1)
}

# 4) Backfill artifacts (residuals, scoring, reliability, backtest) across full span
Write-Section 'Backfill artifacts across span'
& $VenvPython scripts/backfill_artifacts.py --start $minIso --end $maxIso

# 5) Calibrate spread logistic K (provisional if rows limited)
Write-Section 'Calibrate spread logistic K'
& $VenvPython scripts/calibrate_spread_logistic.py --min-rows 300 --provisional-min-rows 50

# 6) Season aggregation refresh
Write-Section 'Refresh season aggregation'
& $VenvPython scripts/season_aggregate.py

# 7) Optional: train segmented models on historical features if available
try {
  $featuresHist = Join-Path $OutDir 'features_hist.csv'
  if (Test-Path $featuresHist) {
    Write-Section 'Train segmented models (team)'
    & $VenvPython -m ncaab_model.cli train-segmented $featuresHist --out-dir (Join-Path $OutDir 'models_segmented') --segment team --min-rows 25 --alpha 1.0
    Write-Section 'Train segmented models (conference)'
    & $VenvPython -m ncaab_model.cli train-segmented $featuresHist --out-dir (Join-Path $OutDir 'models_segmented_conf') --segment conference --min-rows 25 --alpha 1.0
  } else {
    Write-Host 'features_hist.csv not found; segmented training skipped.' -ForegroundColor Yellow
  }
} catch {
  Write-Warning "Segmented training failed: $($_)"
}

Write-Host "`n[Done] Historical backfill complete for $minIso .. $maxIso" -ForegroundColor Green
