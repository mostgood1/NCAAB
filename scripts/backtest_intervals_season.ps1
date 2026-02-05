#Requires -Version 5.1
<#
Backtest 5-minute interval (cumulative) predictions vs ESPN play-by-play across a season.

This wraps the existing CLI:
  python -m ncaab_model.cli backtest-segments-5min --start ... --end ...

It runs in chunks (default 7 days) so you can resume and so it writes multiple smaller artifacts.
Optionally upserts each chunk CSV into outputs/backtests/segments_5min_master.csv.

Examples:
  .\scripts\backtest_intervals_season.ps1 -Start 2025-11-01 -End 2026-01-21
  .\scripts\backtest_intervals_season.ps1 -Start 2025-11-01 -End 2026-01-21 -ChunkDays 3 -Engine events -Samples 2000
  .\scripts\backtest_intervals_season.ps1 -Start 2025-11-01 -End 2026-01-21 -UpdateMaster
#>

param(
  [string]$Start = '2025-11-01',
  [string]$End = $( (Get-Date).AddDays(-1).ToString('yyyy-MM-dd') ),
  [ValidateSet('events','normal','auto')]
  [string]$Engine = 'events',
  [int]$Samples = 2000,
  [double]$Rho = 0.25,
  [switch]$RecomputeSims,
  [switch]$NoCache,
  [double]$SleepSeconds = 0.10,
  [int]$MaxGames = 0,
  [int]$ChunkDays = 7,
  [string]$OutPrefix = 'segments_5min_season',
  [switch]$UpdateMaster
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $VenvPython)) {
  throw "Python venv not found at $VenvPython. Run scripts/daily_update.ps1 -BootstrapEnv first (or create .venv)."
}

$OutDir = Join-Path $RepoRoot 'outputs'
$BtDir = Join-Path $OutDir 'backtests'
New-Item -ItemType Directory -Path $BtDir -Force | Out-Null

$startDt = [DateTime]::ParseExact($Start, 'yyyy-MM-dd', $null)
$endDt = [DateTime]::ParseExact($End, 'yyyy-MM-dd', $null)
if ($endDt -lt $startDt) { throw "End <$End> must be >= Start <$Start>." }

Set-Location $RepoRoot

Write-Host "[intervals-5min] start=$Start end=$End engine=$Engine samples=$Samples chunkDays=$ChunkDays updateMaster=$($UpdateMaster.IsPresent)" -ForegroundColor Cyan

$cur = $startDt
while ($cur -le $endDt) {
  $chunkStart = $cur
  $chunkEnd = $cur.AddDays([Math]::Max(1, $ChunkDays) - 1)
  if ($chunkEnd -gt $endDt) { $chunkEnd = $endDt }

  $s = $chunkStart.ToString('yyyy-MM-dd')
  $e = $chunkEnd.ToString('yyyy-MM-dd')
  Write-Host "\n==== backtest intervals: $s to $e ====\n" -ForegroundColor Green

  $cliArgs = @(
    '-m','ncaab_model.cli','backtest-segments-5min',
    '--start', $s,
    '--end', $e,
    '--engine', $Engine,
    '--samples', "$Samples",
    '--rho', "$Rho",
    '--out-prefix', $OutPrefix,
    '--sleep-seconds', "$SleepSeconds"
  )
  if ($RecomputeSims.IsPresent) { $cliArgs += '--recompute-sims' }
  if ($NoCache.IsPresent) { $cliArgs += '--no-use-cache' }
  if ($MaxGames -gt 0) { $cliArgs += @('--max-games', "$MaxGames") }

  & $VenvPython @cliArgs

  if ($UpdateMaster.IsPresent) {
    $tag = "${s}_to_${e}"
    $dailyCsv = Join-Path $BtDir ("${OutPrefix}_${tag}.csv")
    if (Test-Path $dailyCsv) {
      $dailyInfo = Get-Item $dailyCsv
      if ($dailyInfo.Length -eq 0) {
        Write-Host "[intervals-5min] Master upsert: daily CSV empty -> $dailyCsv (skipping)" -ForegroundColor Yellow
        $cur = $chunkEnd.AddDays(1)
        continue
      }
      $masterCsv = Join-Path $BtDir 'segments_5min_master.csv'
      & $VenvPython scripts\upsert_segments_5min_master.py --daily $dailyCsv --master $masterCsv
      if ($LASTEXITCODE -ne 0) {
        Write-Warning "Master upsert failed (exit=$LASTEXITCODE) for $dailyCsv; continuing."
      }
    } else {
      Write-Warning "Expected backtest CSV not found at $dailyCsv (skipping master upsert)."
    }
  }

  $cur = $chunkEnd.AddDays(1)
}

Write-Host "\n[intervals-5min] Done." -ForegroundColor Cyan

if ($UpdateMaster.IsPresent) {
  Write-Host "[intervals-5min] Refreshing outputs/segment_weights.json from master backtest..." -ForegroundColor Cyan
  try {
    $masterCsv = Join-Path $BtDir 'segments_5min_master.csv'
    $outW = Join-Path $OutDir 'segment_weights.json'
    if (Test-Path $masterCsv) {
      & $VenvPython scripts\update_segment_weights_from_master.py --master $masterCsv --out $outW --shrink-to-uniform 0.10 --min-games 200
    } else {
      Write-Warning "Master CSV not found at $masterCsv; cannot refresh segment weights."
    }
  } catch {
    Write-Warning "Segment weights refresh failed: $($_)"
  }
}
