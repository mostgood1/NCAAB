# PSScriptAnalyzer -DisableRuleName PSUseApprovedVerbs

param(
  [string]$Start = "",
  [string]$End = "",
  [int]$Days = 30,
  [string]$PythonExe = ".venv/Scripts/python.exe",
  [int]$Samples = 4000,
  [double]$Rho = 0.25,
  [string]$Engine = "events",
  [switch]$Recompute,
  [switch]$SkipExisting,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-Info($msg){ Write-Host $msg -ForegroundColor Cyan }
function Write-Warn($msg){ Write-Host $msg -ForegroundColor Yellow }
function Write-Ok($msg){ Write-Host $msg -ForegroundColor Green }
function Write-Err($msg){ Write-Host $msg -ForegroundColor Red }

$ROOT = Resolve-Path (Join-Path $PSScriptRoot "..")
$OUT = Join-Path $ROOT "outputs"

# Resolve Python
$py = Join-Path $ROOT $PythonExe
if (-not (Test-Path $py)) {
  $py = Resolve-Path $PythonExe -ErrorAction SilentlyContinue
}
if (-not $py) {
  Write-Err "Python executable not found: $PythonExe"
  exit 1
}

$startDt = $null
$endDt = $null
if (-not [string]::IsNullOrWhiteSpace($Start)) {
  try { $startDt = [datetime]::ParseExact($Start, 'yyyy-MM-dd', $null) } catch { $startDt = $null }
}
if (-not [string]::IsNullOrWhiteSpace($End)) {
  try { $endDt = [datetime]::ParseExact($End, 'yyyy-MM-dd', $null) } catch { $endDt = $null }
}

if (-not $startDt -or -not $endDt) {
  # Default: last N days ending today
  $endDt = (Get-Date)
  $startDt = $endDt.AddDays(-1 * [math]::Max(0, $Days - 1))
}
if ($endDt -lt $startDt) {
  $tmp = $startDt; $startDt = $endDt; $endDt = $tmp
}

Write-Info "== Backfill 2-min sim artifacts =="
Write-Ok "Range: $($startDt.ToString('yyyy-MM-dd')) -> $($endDt.ToString('yyyy-MM-dd'))"
Write-Ok "Python: $py"
Write-Ok "Engine=$Engine Samples=$Samples Rho=$Rho"

# Match daily defaults unless already set (operator can override externally)
if (-not $env:NCAAB_LATE_ALLOC_SHAPE) { $env:NCAAB_LATE_ALLOC_SHAPE = "1" }
if (-not $env:NCAAB_LATE_ALLOC_CLOSE_MAX) { $env:NCAAB_LATE_ALLOC_CLOSE_MAX = "6" }
if (-not $env:NCAAB_LATE_ALLOC_BLOWOUT_MIN) { $env:NCAAB_LATE_ALLOC_BLOWOUT_MIN = "11" }
if (-not $env:NCAAB_LATE_ALLOC_LAST_MULT_CLOSE) { $env:NCAAB_LATE_ALLOC_LAST_MULT_CLOSE = "1.20" }
if (-not $env:NCAAB_LATE_ALLOC_LAST_MULT_BLOWOUT) { $env:NCAAB_LATE_ALLOC_LAST_MULT_BLOWOUT = "0.65" }

$cur = $startDt
while ($cur -le $endDt) {
  $d = $cur.ToString('yyyy-MM-dd')
  $q = Join-Path $OUT "sim_quantiles_2min_$d.csv"
  $s = Join-Path $OUT "sim_segments_2min_$d.csv"
  $m = Join-Path $OUT "sim_meta_2min_$d.json"

  Write-Info "-- $d --"

  if ($SkipExisting -and (Test-Path $q) -and (Test-Path $s) -and (Test-Path $m)) {
    Write-Warn "[skip] already exists"
    $cur = $cur.AddDays(1)
    continue
  }

  if ($Recompute) {
    try { if (Test-Path $q) { Remove-Item $q -Force } } catch { }
    try { if (Test-Path $s) { Remove-Item $s -Force } } catch { }
    try { if (Test-Path $m) { Remove-Item $m -Force } } catch { }
  }

  $scriptParams = @(
    (Join-Path $ROOT "scripts/run_game_simulations.py"),
    $d,
    $OUT,
    "--segments-grid-min", "2",
    "--engine", $Engine,
    "--samples", "$Samples",
    "--rho", "$Rho",
    "--quantiles-out-prefix", "sim_quantiles_2min_",
    "--segments-out-prefix", "sim_segments_2min_",
    "--meta-out-prefix", "sim_meta_2min_"
  )

  if ($DryRun) {
    Write-Info "[dry] $py $($scriptParams -join ' ')"
  } else {
    & $py @scriptParams | Out-Host
  }

  $cur = $cur.AddDays(1)
}

Write-Ok "Done. Generated sim_quantiles_2min_*, sim_segments_2min_*, sim_meta_2min_* under outputs/."
