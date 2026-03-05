#Requires -Version 5.1
# Daily update pipeline for NCAAB project
# - Pulls previous day's finals and last pre-tip odds (strict) and closing totals (heuristic)
# - Reconciles predictions vs finals (daily-results)
# - Updates tuning
# - Retrains models (baseline)
# - Pulls and verifies today's schedule
# - Fetches today's odds and runs predictions/picks for today

param(
  [string]$Today = $(Get-Date -Format 'yyyy-MM-dd'),
  [string]$Region = 'us',
  [string]$Provider = 'espn',
  [switch]$NoCache,
  [switch]$SkipRetrain,
  [switch]$ForceModelRetrain,
  [switch]$SkipFinalizePrev,
  # Deprecated (stake sheets removed from app). Retained for backward compatibility; now a no-op.
  [switch]$SkipStakeSheets,
  [switch]$SkipGitPush,
  [switch]$SkipModelTests,
  [switch]$SkipVarianceDiag,
  [switch]$BootstrapEnv,
  [switch]$NoTranscript,
  # Heavy quantile CV + model retrain gating (weekly + drift/age overrides)
  [switch]$SkipHeavyQuantiles,
  [switch]$ForceQuantileRefresh,
  [string]$QuantileRetrainDay = 'Sunday',
  [int]$QuantileMaxAgeDays = 6,
  [string]$GitCommitMessage,
  # Simulation calibration (global mean/sigma adjustments applied by sim engine)
  [switch]$SkipSimCalibrationFit,
  [int]$SimCalibrationLookbackDays = 21,
  [int]$SimCalibrationMinGames = 80,
  [switch]$SimCalibrationFit1HOnly,
  # Simulation engine backtesting (historical join vs daily_results/results_*.csv)
  [switch]$RunSimBacktest,
  [int]$SimBacktestRecent = 30,
  [int]$SimBacktestSamples = 1000,
  [string]$SimBacktestEngine = 'events',
  [switch]$SimBacktestRecompute,
  # Accuracy backtesting (winners / totals / ATS across historical finalized days)
  [switch]$RunAccuracyBacktest,
  [int]$AccuracyBacktestRecent = 30,
  # Sim accuracy backtesting (winners / totals / ATS using sim_quantiles_* predictions)
  [switch]$RunSimAccuracyBacktest,
  [int]$SimAccuracyBacktestRecent = 30,
  [int]$SimAccuracyBacktestSamples = 2000,
  [string]$SimAccuracyBacktestEngine = 'events',
  [switch]$SimAccuracyBacktestRecompute,
  # Daily 5-min segment reconciliation + calibration refresh
  [switch]$SkipSegment5MinRecon,
  # Auto-refresh 5-min segment weights used by simulator (outputs/segment_weights.json)
  [switch]$SkipSegmentWeights5Min,
  # Daily 5-min interval actuals artifact (ESPN play-by-play -> outputs/interval_actuals_5min_<date>.csv)
  [switch]$SkipIntervalActuals5Min,
  [int]$Segment5MinBacktestSamples = 2000,
  [string]$Segment5MinBacktestEngine = 'events',
  [int]$Segment5MinCalibrationLookbackDays = 45,
  [int]$Segment5MinCalibrationMinRowsPerEndMin = 150,
  [int]$Segment5MinCalibrationMinRowsUsed = 800,
  [int]$Segment5MinCalibrationMinEndpoints = 4,
  # Probability calibration (isotonic mapping applied during distributional stake sizing)
  [switch]$SkipProbCalibrationFit,
  [int]$ProbCalibrationLookbackDays = 60,
  [int]$ProbCalibrationMinRows = 500,
  # Render upload integration
  [switch]$UploadToRender,
  [switch]$TriggerRenderRedeploy,
  # New: Upload to Render by default; opt-out with -SkipRenderUpload
  [switch]$SkipRenderUpload,
  # Redeploy after upload only when explicitly requested; opt-out with -SkipRenderRedeploy
  [switch]$SkipRenderRedeploy,
  # Optional: base URL used to fetch live snapshot JSONL from deployed app (for eval automation)
  [string]$RenderBaseUrl = $(if ($env:NCAAB_RENDER_BASE_URL) { $env:NCAAB_RENDER_BASE_URL } else { 'https://ncaab.onrender.com' }),

  # Live Lens interval evaluation (optional)
  [switch]$RunLiveIntervalEval,
  [int]$LiveIntervalEvalLookbackDays = 21,
  [int]$LiveIntervalEvalMaxFiles = 0,

  # Live Lens OVER penalty tuning (optional; requires live_lens_signals_*.jsonl + daily_results/results_*.csv)
  [switch]$RunLiveLensOverTuning,
  [int]$LiveLensOverTuningLookbackDays = 21,
  [double]$LiveLensOverTuningAssumePrice = -110.0,
  [switch]$ApplyLiveLensOverTuning,
  [int]$LiveLensOverTuningMinBucketN = 10,
  [int]$LiveLensOverTuningMinOverallN = 25,

  # Live Lens driver-tag penalty learning (optional; requires driver/driver_tags in signals)
  [switch]$RunLiveLensFlagPenalties,
  [int]$LiveLensFlagPenaltiesLookbackDays = 21,

  # Offline-first cache maintenance (keeps local caches warm for feature computation)
  [switch]$SkipOfflineCacheMaintenance,
  [int]$OfflineScoreboardPrimeLookbackDays = 60,
  [int]$OfflineGameCachePrimeLookbackDays = 2
)

$ErrorActionPreference = 'Stop'
$script:StartTime = Get-Date
$script:CriticalFailures = @()

function Add-CriticalFailure($msg) {
  $script:CriticalFailures += $msg
  try {
    if ($script:Steps -and $script:Steps.Count -gt 0) {
      $cur = $script:Steps[-1].section
      if (-not $script:StepErrors.ContainsKey($cur)) { $script:StepErrors[$cur] = @() }
      $script:StepErrors[$cur] += $msg
    }
  } catch {}
  Write-Error "[critical] $msg"
}

function Write-Section($msg) {
  $now = Get-Date
  if (-not $script:Steps) { $script:Steps = @() }
  if (-not $script:StepErrors) { $script:StepErrors = @{} }
  $script:Steps += [pscustomobject]@{section=$msg; start=$now}
  Write-Host "`n==== $msg ====\n"
}

# Resolve paths
$RepoRoot = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path (Join-Path $RepoRoot '.venv')) -or $BootstrapEnv.IsPresent) {
  Write-Section 'BOOTSTRAP: Creating / refreshing .venv and installing dependencies'
  try {
    $pyCmd = (Get-Command python -ErrorAction SilentlyContinue)
    if (-not $pyCmd) { $pyCmd = (Get-Command py -ErrorAction SilentlyContinue) }
    if (-not $pyCmd) { throw 'No base Python interpreter found on PATH (python or py). Install Python 3.11+.' }
    & $pyCmd.Source -m venv (Join-Path $RepoRoot '.venv')
    $venvPip = Join-Path $RepoRoot '.venv\Scripts\python.exe'
    & $venvPip -m pip install --upgrade pip
    $reqFile = Join-Path $RepoRoot 'requirements.txt'
    if (Test-Path $reqFile) { & $venvPip -m pip install -r $reqFile }
    $pyproj = Join-Path $RepoRoot 'pyproject.toml'
    if (Test-Path $pyproj) { & $venvPip -m pip install -e $RepoRoot }
    Write-Host 'BOOTSTRAP complete.' -ForegroundColor Green
  } catch {
    Add-CriticalFailure "Environment bootstrap failed: $($_)"
    if ($env:NCAAB_STRICT_EXIT -eq '1') { exit 1 }
  }
}

$VenvPython = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $VenvPython)) {
  Add-CriticalFailure "Python venv not found at $VenvPython after bootstrap attempt."
  if ($env:NCAAB_STRICT_EXIT -eq '1') { exit 1 } else { return }
}
$OutDir = Join-Path $RepoRoot 'outputs'
$LogsDir = Join-Path $OutDir 'logs'
New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null

$LogStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$LogPath = Join-Path $LogsDir "daily_update_$LogStamp.log"
if (-not $NoTranscript.IsPresent) {
  try { Start-Transcript -Path $LogPath -Append | Out-Null } catch { Write-Warning "Transcript start failed: $($_)" }
} else { Write-Host 'Transcript disabled via -NoTranscript.' -ForegroundColor DarkGray }

try {
  Set-Location $RepoRoot

  # Default late-game shaping for 2-min point-allocation segments.
  # This only affects simulations when NCAAB_SEGMENTS_GRID_MIN=2 is used.
  if (-not $env:NCAAB_LATE_ALLOC_SHAPE -or $env:NCAAB_LATE_ALLOC_SHAPE.Trim() -eq '') {
    $env:NCAAB_LATE_ALLOC_SHAPE = '1'
  }
  if (-not $env:NCAAB_LATE_ALLOC_CLOSE_MAX -or $env:NCAAB_LATE_ALLOC_CLOSE_MAX.Trim() -eq '') {
    $env:NCAAB_LATE_ALLOC_CLOSE_MAX = '6'
  }
  if (-not $env:NCAAB_LATE_ALLOC_BLOWOUT_MIN -or $env:NCAAB_LATE_ALLOC_BLOWOUT_MIN.Trim() -eq '') {
    $env:NCAAB_LATE_ALLOC_BLOWOUT_MIN = '11'
  }
  if (-not $env:NCAAB_LATE_ALLOC_LAST_MULT_CLOSE -or $env:NCAAB_LATE_ALLOC_LAST_MULT_CLOSE.Trim() -eq '') {
    $env:NCAAB_LATE_ALLOC_LAST_MULT_CLOSE = '1.20'
  }
  if (-not $env:NCAAB_LATE_ALLOC_LAST_MULT_BLOWOUT -or $env:NCAAB_LATE_ALLOC_LAST_MULT_BLOWOUT.Trim() -eq '') {
    $env:NCAAB_LATE_ALLOC_LAST_MULT_BLOWOUT = '0.65'
  }
  Write-Host "[late-shape] NCAAB_LATE_ALLOC_SHAPE=$($env:NCAAB_LATE_ALLOC_SHAPE) CLOSE_MAX=$($env:NCAAB_LATE_ALLOC_CLOSE_MAX) BLOWOUT_MIN=$($env:NCAAB_LATE_ALLOC_BLOWOUT_MIN) CLOSE_MULT=$($env:NCAAB_LATE_ALLOC_LAST_MULT_CLOSE) BLOWOUT_MULT=$($env:NCAAB_LATE_ALLOC_LAST_MULT_BLOWOUT)" -ForegroundColor DarkGray

  # Default market-dispersion sigma scaling knobs when dispersion is enabled.
  # This is intentionally opt-in via NCAAB_SIM_MARKET_DISPERSION_SIGMA.
  try {
    $md = $env:NCAAB_SIM_MARKET_DISPERSION_SIGMA
    $mdOn = $false
    if ($md -and $md.Trim() -ne '' -and $md.Trim() -ne '0') {
      $mdl = $md.Trim().ToLower()
      if ($mdl -ne 'false' -and $mdl -ne 'off' -and $mdl -ne 'no') {
        $mdOn = $true
      }
    }
    if ($mdOn) {
      if (-not $env:NCAAB_SIM_MARKET_DISPERSION_EXP -or $env:NCAAB_SIM_MARKET_DISPERSION_EXP.Trim() -eq '') { $env:NCAAB_SIM_MARKET_DISPERSION_EXP = '1.0' }
      if (-not $env:NCAAB_SIM_MARKET_DISPERSION_MIN_MULT -or $env:NCAAB_SIM_MARKET_DISPERSION_MIN_MULT.Trim() -eq '') { $env:NCAAB_SIM_MARKET_DISPERSION_MIN_MULT = '1.0' }
      if (-not $env:NCAAB_SIM_MARKET_DISPERSION_MAX_MULT -or $env:NCAAB_SIM_MARKET_DISPERSION_MAX_MULT.Trim() -eq '') { $env:NCAAB_SIM_MARKET_DISPERSION_MAX_MULT = '1.2' }

      # Safe defaults: totals-only, and do not scale 1H distributions.
      if (-not $env:NCAAB_SIM_MARKET_DISPERSION_APPLY_TOTAL -or $env:NCAAB_SIM_MARKET_DISPERSION_APPLY_TOTAL.Trim() -eq '') { $env:NCAAB_SIM_MARKET_DISPERSION_APPLY_TOTAL = '1' }
      if (-not $env:NCAAB_SIM_MARKET_DISPERSION_APPLY_MARGIN -or $env:NCAAB_SIM_MARKET_DISPERSION_APPLY_MARGIN.Trim() -eq '') { $env:NCAAB_SIM_MARKET_DISPERSION_APPLY_MARGIN = '0' }
      if (-not $env:NCAAB_SIM_MARKET_DISPERSION_APPLY_1H -or $env:NCAAB_SIM_MARKET_DISPERSION_APPLY_1H.Trim() -eq '') { $env:NCAAB_SIM_MARKET_DISPERSION_APPLY_1H = '0' }

      Write-Host "[market-dispersion] enabled exp=$($env:NCAAB_SIM_MARKET_DISPERSION_EXP) min=$($env:NCAAB_SIM_MARKET_DISPERSION_MIN_MULT) max=$($env:NCAAB_SIM_MARKET_DISPERSION_MAX_MULT) apply_total=$($env:NCAAB_SIM_MARKET_DISPERSION_APPLY_TOTAL) apply_margin=$($env:NCAAB_SIM_MARKET_DISPERSION_APPLY_MARGIN) apply_1h=$($env:NCAAB_SIM_MARKET_DISPERSION_APPLY_1H)" -ForegroundColor DarkGray
    }
  } catch {
    Write-Warning "market dispersion env defaulting failed (continuing): $($_)"
  }

  # Default live time-rate fallback conditioning knobs (only when enabled).
  # Enable by setting NCAAB_LIVE_TIME_RATE_ADJUST=1. This affects live remainder conditioning
  # only when pbp possession estimates are unavailable (improves early-game totals bias).
  try {
    $tr = $env:NCAAB_LIVE_TIME_RATE_ADJUST
    $trOn = $false
    if ($tr -and $tr.Trim() -ne '' -and $tr.Trim() -ne '0') {
      $trl = $tr.Trim().ToLower()
      if ($trl -ne 'false' -and $trl -ne 'off' -and $trl -ne 'no') {
        $trOn = $true
      }
    }
    if ($trOn) {
      if (-not $env:NCAAB_LIVE_TIME_RATE_RATIO_MIN -or $env:NCAAB_LIVE_TIME_RATE_RATIO_MIN.Trim() -eq '') { $env:NCAAB_LIVE_TIME_RATE_RATIO_MIN = '0.88' }
      if (-not $env:NCAAB_LIVE_TIME_RATE_RATIO_MAX -or $env:NCAAB_LIVE_TIME_RATE_RATIO_MAX.Trim() -eq '') { $env:NCAAB_LIVE_TIME_RATE_RATIO_MAX = '1.35' }
      if (-not $env:NCAAB_LIVE_TIME_RATE_TAU -or $env:NCAAB_LIVE_TIME_RATE_TAU.Trim() -eq '') { $env:NCAAB_LIVE_TIME_RATE_TAU = '0.35' }
      if (-not $env:NCAAB_LIVE_TIME_RATE_W_MAX -or $env:NCAAB_LIVE_TIME_RATE_W_MAX.Trim() -eq '') { $env:NCAAB_LIVE_TIME_RATE_W_MAX = '0.85' }

      Write-Host "[time-rate] enabled ratio_min=$($env:NCAAB_LIVE_TIME_RATE_RATIO_MIN) ratio_max=$($env:NCAAB_LIVE_TIME_RATE_RATIO_MAX) tau=$($env:NCAAB_LIVE_TIME_RATE_TAU) w_max=$($env:NCAAB_LIVE_TIME_RATE_W_MAX)" -ForegroundColor DarkGray
    }
  } catch {
    Write-Warning "time-rate env defaulting failed (continuing): $($_)"
  }

  # Compute dates
  $todayDate = [DateTime]::ParseExact($Today, 'yyyy-MM-dd', $null)
  $prevDate = $todayDate.AddDays(-1).ToString('yyyy-MM-dd')
  $todayIso = $todayDate.ToString('yyyy-MM-dd')

  # Bookmaker keys to request from TheOddsAPI. Default to a sharp-ish US trio.
  # Override via env var if needed (comma-separated keys).
  $script:TheOddsBookmakers = if ($env:NCAAB_THEODDS_BOOKMAKERS -and $env:NCAAB_THEODDS_BOOKMAKERS.Trim() -ne '') {
    $env:NCAAB_THEODDS_BOOKMAKERS.Trim()
  } else {
    'draftkings,fanduel,betmgm'
  }

  # NCAAB season spans calendar years; use season start year for providers that key on season.
  $seasonStartYear = if ($todayDate.Month -lt 7) { $todayDate.Year - 1 } else { $todayDate.Year }

  # Normalize Render base URL (used for upload/health checks). Allow override via -RenderBaseUrl or env.
  $script:RenderBaseUrlEff = $RenderBaseUrl
  if (-not $script:RenderBaseUrlEff -or $script:RenderBaseUrlEff.Trim() -eq '') {
    $script:RenderBaseUrlEff = 'https://ncaab.onrender.com'
  }
  $script:RenderBaseUrlEff = $script:RenderBaseUrlEff.Trim().TrimEnd('/')

  # Quantile heavy task gating setup
  $qselPath = Join-Path $OutDir 'quantile_model_selection.json'
  $artifactAgeDays = if (Test-Path $qselPath) { ((Get-Date) - (Get-Item $qselPath).LastWriteTime).TotalDays } else { [double]::PositiveInfinity }
  $dow = (Get-Date $todayDate).DayOfWeek.ToString()
  $RunHeavyQuantiles = $ForceQuantileRefresh.IsPresent -or (
    (-not $SkipHeavyQuantiles.IsPresent) -and (
      $dow -eq $QuantileRetrainDay -or
      (-not (Test-Path $qselPath)) -or
      $artifactAgeDays -ge $QuantileMaxAgeDays
    )
  )
  Write-Host "[quantile-gating] day=$dow targetDay=$QuantileRetrainDay ageDays=$([Math]::Round($artifactAgeDays,2)) runHeavy=$RunHeavyQuantiles" -ForegroundColor DarkGray

  # 0.tuning) Compute Live Lens pace/efficiency thresholds from cached ESPN PBP
  # Writes: outputs/live_lens_tuning.json (used by /api/live_lens_tuning -> Live Lens JS)
  Write-Section "0.tuning) Compute Live Lens tuning (pace/efficiency)"
  try {
    $tuneOut = Join-Path $OutDir 'live_lens_tuning.json'
    & $VenvPython -m ncaab_model.cli compute-live-lens-tuning --days 21 --max-files 500 --out $tuneOut
  } catch {
    Write-Warning "Live Lens tuning compute failed: $($_)"
  }

  # 0.interval-cal) Ensure canonical Live Lens interval calibration artifact is present
  # Used by /api/live_interval_calibration -> templates/index.html (Live Lens totals).
  Write-Section "0.interval-cal) Ensure Live Lens interval calibration artifact"
  try {
    $calCanon = Join-Path $OutDir 'live_interval_calibration.json'
    if (-not (Test-Path -LiteralPath $calCanon)) {
      $cands = Get-ChildItem -Path $OutDir -Filter 'live_interval_calibration_*.json' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
      if ($cands -and $cands.Count -gt 0) {
        Copy-Item -LiteralPath $cands[0].FullName -Destination $calCanon -Force
        Write-Host "[interval-cal] Copied newest -> live_interval_calibration.json ($($cands[0].Name))" -ForegroundColor DarkGray
      } else {
        Write-Host "[interval-cal] No calibration artifacts found; Live Lens will run uncalibrated." -ForegroundColor DarkGray
      }
    } else {
      Write-Host "[interval-cal] Found live_interval_calibration.json" -ForegroundColor DarkGray
    }
  } catch {
    Write-Warning "interval calibration artifact ensure failed: $($_)"
  }

  # 0.weights) Refresh 5-min segment weights from season master backtest (fast, no network)
  Write-Section "0.weights) Refresh 5-min segment weights (from master backtest)"
  try {
    if ($SkipSegmentWeights5Min.IsPresent) {
      Write-Host "[segment-weights] Skipped via -SkipSegmentWeights5Min" -ForegroundColor DarkGray
    } else {
      $master = Join-Path $OutDir 'backtests\segments_5min_master.csv'
      $outW = Join-Path $OutDir 'segment_weights.json'
      if (Test-Path $master) {
        & $VenvPython scripts\update_segment_weights_from_master.py --master $master --out $outW --shrink-to-uniform 0.10 --min-games 200
      } else {
        Write-Host "[segment-weights] Master not found ($master); leaving existing weights as-is." -ForegroundColor DarkGray
      }
    }
  } catch {
    Write-Warning "segment weights refresh failed: $($_)"
  }

  # 0) Ensure ESPN cache + TBD patch + subset parity before the rest of the flow
  Write-Section "0) ESPN schedule refresh + TBD patch + parity"
  try {
    $ScheduleRefresh = Join-Path $RepoRoot 'scripts\schedule_refresh.ps1'
    if (Test-Path $ScheduleRefresh) {
      & $ScheduleRefresh -Date $todayIso
    } else {
      Write-Warning "schedule_refresh.ps1 not found at $ScheduleRefresh; skipping preflight refresh."
    }
  } catch {
    Write-Warning "schedule_refresh preflight failed: $($_)"
  }

  # 0.cache) Offline cache maintenance: prime recent day scoreboards + prime recent ESPN per-game caches,
  # then rebuild a cache-only fused games list and audit missing artifacts.
  Write-Section "0.cache) Offline cache maintenance (prime + audit)"
  try {
    if ($SkipOfflineCacheMaintenance.IsPresent) {
      Write-Host "[offline-cache] Skipped via -SkipOfflineCacheMaintenance" -ForegroundColor DarkGray
    } else {
      $seasonStartIso = (Get-Date -Year $seasonStartYear -Month 11 -Day 1 -Format 'yyyy-MM-dd')

      # Prime recent day scoreboards (both ESPN + NCAA), but only for a rolling window to limit 404 noise.
      $scoreLb = [Math]::Max(1, [int]$OfflineScoreboardPrimeLookbackDays)
      $scoreStartDt = $todayDate.AddDays(-1 * $scoreLb)
      $seasonStartDt = [DateTime]::ParseExact($seasonStartIso, 'yyyy-MM-dd', $null)
      if ($scoreStartDt -lt $seasonStartDt) { $scoreStartDt = $seasonStartDt }
      $scoreStartIso = $scoreStartDt.ToString('yyyy-MM-dd')
      $useCacheFlag = @()
      if ($NoCache.IsPresent) { $useCacheFlag += '--no-use-cache' } else { $useCacheFlag += '--use-cache' }
      $tmpPrimeDaysCsv = Join-Path $OutDir ("_tmp_prime_days_" + $todayIso + ".csv")
      & $VenvPython -m ncaab_model.cli prime-cache --start $scoreStartIso --end $todayIso --provider both @useCacheFlag --no-fetch-summaries --no-fetch-pbp --sleep-seconds 0.05 --out-games-csv $tmpPrimeDaysCsv

      # Prime per-game ESPN summary + PBP caches for very recent games (yesterday + today by default).
      $gameLb = [Math]::Max(1, [int]$OfflineGameCachePrimeLookbackDays)
      $gameStartIso = $todayDate.AddDays(-1 * $gameLb).ToString('yyyy-MM-dd')
      if ([DateTime]::ParseExact($gameStartIso, 'yyyy-MM-dd', $null) -lt $seasonStartDt) { $gameStartIso = $seasonStartIso }
      $tmpPrimeGamesCsv = Join-Path $OutDir ("_tmp_prime_gamecache_" + $todayIso + ".csv")
      & $VenvPython -m ncaab_model.cli prime-cache --start $gameStartIso --end $todayIso --provider espn @useCacheFlag --fetch-summaries --fetch-pbp --sleep-seconds 0.05 --out-games-csv $tmpPrimeGamesCsv

      # Rebuild fused games list from caches only (for offline feature computation + diagnostics).
      $fusedOut = Join-Path $OutDir ("games_fused_cacheonly_" + $seasonStartIso + "_" + $todayIso + ".csv")
      & $VenvPython -m ncaab_model.cli fetch-games-fused --season $seasonStartYear --start $seasonStartIso --end $todayIso --cache-only --out $fusedOut

      # Audit which artifacts are still missing locally (ESPN summaries/PBP + day scoreboards).
      & $VenvPython -m ncaab_model.cli audit-local-data $fusedOut --out-json (Join-Path $OutDir 'local_data_coverage.json') --out-missing-csv (Join-Path $OutDir 'local_data_missing.csv')
    }
  } catch {
    Write-Warning "offline cache maintenance failed: $($_)"
  }

  # 0.pre) Fetch today's slate immediately and normalize display times (Central)
  Write-Section "0.pre) Fetch today's slate + normalize display times"
  try {
    $gamesTodayPath = Join-Path $OutDir ("games_" + $todayIso + ".csv")
    & $VenvPython -m ncaab_model.cli fetch-games --season $seasonStartYear --start $todayIso --end $todayIso --provider $Provider --out $gamesTodayPath
    $tmpNorm = Join-Path $OutDir "_tmp_norm_games.py"
    $normCode = @"
import pandas as pd
from pathlib import Path
from zoneinfo import ZoneInfo
import datetime as dt

out_dir = Path(r'${OutDir}')
date = '${todayIso}'
games_path = out_dir / f'games_{date}.csv'
df = pd.read_csv(games_path)

central = ZoneInfo('America/Chicago')

def parse_utc(row):
    for c in ['_start_dt','start_time_iso','commence_time','start_time']:
        v = row.get(c)
        if v is None or str(v).strip()=='':
            continue
        try:
            s = str(v).replace('Z','+00:00')
            ts = pd.to_datetime(s, errors='coerce', utc=True)
            if pd.notna(ts):
                return ts
        except Exception:
            pass
    return None

rows = []
for r in df.to_dict('records'):
    ts_utc = parse_utc(r)
    if ts_utc is not None:
        ts_loc = ts_utc.tz_convert(central)
        disp_date = ts_loc.strftime('%Y-%m-%d')
        disp_time = ts_loc.strftime('%H:%M')
        abbr = ts_loc.tzname() or 'CST'
        r['start_time_iso'] = ts_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
        r['start_time_display'] = f"{disp_date} {disp_time} {abbr}"
        r['display_time_str'] = r['start_time_display']
        r['start_time_local'] = f"{disp_date} {disp_time}"
        r['display_date'] = disp_date
        r['date'] = disp_date
        r['start_tz_abbr'] = abbr
    rows.append(r)
df2 = pd.DataFrame(rows)
df2.to_csv(games_path, index=False)
print({'path': str(games_path), 'rows': len(df2)})
"@
    $normCode | Set-Content -Path $tmpNorm -Encoding UTF8
    & $VenvPython $tmpNorm
    Remove-Item -Path $tmpNorm -ErrorAction SilentlyContinue
  } catch {
    Write-Warning "Slate fetch/normalization failed: $($_)"
  }

  # 0.pre.b) Build canonical start times for the date (single source of truth)
  Write-Section "0.pre.b) Canonical start times"
  try {
    $canon = (& $VenvPython scripts/canonical_start_times.py $todayIso) | Out-String
    Write-Host $canon.Trim()
  } catch { Write-Warning "canonical_start_times.py failed: $($_)" }

  Write-Section "1) Fetch previous day's games ($prevDate)"
  $noCacheFlag = @()
  if ($NoCache.IsPresent) { $noCacheFlag += '--no-use-cache' }
  & $VenvPython -m ncaab_model.cli fetch-games --season $seasonStartYear --start $prevDate --end $prevDate --provider $Provider @noCacheFlag --out (Join-Path $OutDir 'games_prev.csv')

  Write-Section "2) Fetch odds snapshots for $prevDate and build last/closing lines"
  & $VenvPython -m ncaab_model.cli fetch-odds-history --start $prevDate --end $prevDate --region $Region --bookmakers $script:TheOddsBookmakers --markets "h2h,spreads,totals,spreads_h1,totals_h1,spreads_h2,totals_h2" --out-dir (Join-Path $OutDir 'odds_history') --mode current
  & $VenvPython -m ncaab_model.cli make-closing-lines --in-dir (Join-Path $OutDir 'odds_history') --out (Join-Path $OutDir 'closing_lines.csv')
  & $VenvPython -m ncaab_model.cli join-closing (Join-Path $OutDir 'games_prev.csv') (Join-Path $OutDir 'closing_lines.csv') --out (Join-Path $OutDir 'games_with_closing_prev.csv')
  # Also refresh master merged closing across all days using games_all.csv to avoid losing previous-day lines
  & $VenvPython -m ncaab_model.cli join-closing (Join-Path $OutDir 'games_all.csv') (Join-Path $OutDir 'closing_lines.csv') --out (Join-Path $OutDir 'games_with_closing.csv')
  # Strict last pre-tip odds (no synthetic fallback). Use small tolerance for clock skew.
  & $VenvPython -m ncaab_model.cli make-last-odds --in-dir (Join-Path $OutDir 'odds_history') --out (Join-Path $OutDir 'last_odds.csv') --tolerance-seconds 60
  & $VenvPython -m ncaab_model.cli join-last-odds (Join-Path $OutDir 'games_prev.csv') (Join-Path $OutDir 'last_odds.csv') --out (Join-Path $OutDir 'games_with_last_prev.csv')
  # Also refresh master merged last across all days using games_all.csv so prior-day odds persist
  & $VenvPython -m ncaab_model.cli join-last-odds (Join-Path $OutDir 'games_all.csv') (Join-Path $OutDir 'last_odds.csv') --out (Join-Path $OutDir 'games_with_last.csv')

  # Build robust consensus + dispersion from strict last pre-tip odds (DK/FD/MGM by default)
  & $VenvPython -m ncaab_model.cli make-market-consensus --in-path (Join-Path $OutDir 'games_with_last_prev.csv') --out (Join-Path $OutDir 'market_consensus_prev.csv') --min-books 2 --period full_game
  & $VenvPython -m ncaab_model.cli make-market-consensus --in-path (Join-Path $OutDir 'games_with_last.csv') --out (Join-Path $OutDir 'market_consensus.csv') --min-books 2 --period full_game

  Write-Section "3) Build daily results (reconcile vs finals) for $prevDate"
  $predsAll = Join-Path $OutDir 'predictions_all.csv'
  $picksClean = Join-Path $OutDir 'picks_clean.csv'
  # Prefer strict last odds for reconciliation; closing kept for reference.
  & $VenvPython -m ncaab_model.cli daily-results --date $prevDate --games-path (Join-Path $OutDir 'games_prev.csv') --preds-path $predsAll --closing-merged (Join-Path $OutDir 'games_with_last_prev.csv') --picks-path $picksClean --out-dir (Join-Path $OutDir 'daily_results')

  # Evaluate previous day accuracy vs odds (OU/ATS) and persist metrics
  Write-Section "3c) Evaluate previous-day accuracy vs odds"
  try {
    $evalOut = (& $VenvPython scripts/evaluate_vs_odds.py --date $prevDate) | Out-String
    $metricsDir = Join-Path $OutDir 'metrics'
    New-Item -ItemType Directory -Path $metricsDir -Force | Out-Null
    $metricsPath = Join-Path $metricsDir ("accuracy_vs_odds_" + $prevDate + ".json")
    $evalOut.Trim() | Out-File -FilePath $metricsPath -Encoding UTF8
    Write-Host "[metrics] Wrote $metricsPath"
  } catch { Write-Warning "evaluate_vs_odds failed for ${prevDate}: $($_)" }

  if (-not $SkipFinalizePrev) {
    Write-Section "3a) Fetch previous day raw scores (no cache) and boxscores"
    try {
      & $VenvPython -m ncaab_model.cli fetch-scores --date $prevDate --provider both
    }
    catch {
      Write-Warning "fetch-scores failed for ${prevDate}: $($_)"
    }
    # Boxscores often don't contain final scores early; still useful for four-factors
    try {
      & $VenvPython -m ncaab_model.cli fetch-boxscores (Join-Path $OutDir 'games_prev.csv') --out (Join-Path $OutDir 'boxscores_prev.csv')
    }
    catch {
      Write-Warning "fetch-boxscores failed for ${prevDate}: $($_)"
    }

    # Merge fresh boxscores into the consolidated artifact so downstream feature refreshes
    # (team_features + sim event-rate rolling stats) always have the latest games.
    try {
      $base = Join-Path $OutDir 'boxscores.csv'
      $new = Join-Path $OutDir 'boxscores_prev.csv'
      if (Test-Path $new) {
        $mergeOut = (& $VenvPython scripts/merge_boxscores.py --base $base --new $new --out $base) | Out-String
        if ($mergeOut) { Write-Host ($mergeOut.Trim()) }
      }
    } catch {
      Write-Warning "merge_boxscores failed: $($_)"
    }
    Write-Section "3b) Finalize previous day (refresh + fallback + overrides)"
    try {
      # Force provider refresh; finalize-day will also try secondary/fused providers and apply outputs/scores_override_<date>.csv if present.
      & $VenvPython -m ncaab_model.cli finalize-day --date $prevDate --provider $Provider --no-use-cache --games-csv (Join-Path $OutDir 'games_all.csv') --boxscores-csv (Join-Path $OutDir 'boxscores_prev.csv') --out-dir (Join-Path $OutDir 'daily_results') --include-halves
    }
    catch {
      Write-Warning "finalize-day failed for ${prevDate}: $($_)"
    }

    # Cleanup: remove any non-date daily_results files (e.g., results_TEST.csv)
    try {
      $resultsDir = Join-Path $OutDir 'daily_results'
      if (Test-Path $resultsDir) {
        Get-ChildItem $resultsDir -File | Where-Object {
          $_.Name -like 'results_*.csv' -and ($_.Name -notmatch '^results_\d{4}-\d{2}-\d{2}\.csv$')
        } | ForEach-Object {
          Write-Host "[cleanup] Removing stray results file: $($_.FullName)" -ForegroundColor DarkGray
          Remove-Item $_.FullName -Force
        }
      }
    } catch {
      Write-Warning "[cleanup] Failed to remove stray results files: $($_)"
    }

    # Live Lens bet accuracy (requires UI-captured signals + finalized results)
    Write-Section "3b.i) Live Lens bet accuracy (win-rate/ROI) for $prevDate"
    try {
      function Test-HasBytes {
        param([string]$Path)
        if (-not (Test-Path -LiteralPath $Path)) { return $false }
        try { return ((Get-Item -LiteralPath $Path).Length -gt 0) } catch { return $false }
      }

      function Invoke-DownloadSignals {
        param(
          [string]$BaseUrl,
          [string]$Date,
          [string]$OutFile
        )
        $b = ("" + $BaseUrl).Trim()
        if ([string]::IsNullOrWhiteSpace($b)) { return $false }
        $b = $b.TrimEnd('/')
        $url = "$b/api/download_live_lens_signals?date=$Date"
        Write-Host "[live_lens] Attempting signals download: $url" -ForegroundColor DarkGray
        try {
          Invoke-WebRequest -Uri $url -OutFile $OutFile -UseBasicParsing -TimeoutSec 30 | Out-Null
          return (Test-HasBytes $OutFile)
        } catch {
          $resp = $_.Exception.Response
          if ($resp) {
            Write-Warning ("[live_lens] Signals download failed ({0}): {1}" -f ([int]$resp.StatusCode), $_.Exception.Message)
          } else {
            Write-Warning "[live_lens] Signals download failed: $($_.Exception.Message)"
          }
          if (Test-Path -LiteralPath $OutFile) { Remove-Item -LiteralPath $OutFile -Force -ErrorAction SilentlyContinue }
          return $false
        }
      }

      function Invoke-DownloadProjections {
        param(
          [string]$BaseUrl,
          [string]$Date,
          [string]$OutFile
        )
        $b = ("" + $BaseUrl).Trim()
        if ([string]::IsNullOrWhiteSpace($b)) { return $false }
        $b = $b.TrimEnd('/')
        $url = "$b/api/download_live_lens_projections?date=$Date"
        Write-Host "[live_lens] Attempting projections download: $url" -ForegroundColor DarkGray
        try {
          Invoke-WebRequest -Uri $url -OutFile $OutFile -UseBasicParsing -TimeoutSec 30 | Out-Null
          return (Test-HasBytes $OutFile)
        } catch {
          $resp = $_.Exception.Response
          if ($resp) {
            Write-Warning ("[live_lens] Projections download failed ({0}): {1}" -f ([int]$resp.StatusCode), $_.Exception.Message)
          } else {
            Write-Warning "[live_lens] Projections download failed: $($_.Exception.Message)"
          }
          if (Test-Path -LiteralPath $OutFile) { Remove-Item -LiteralPath $OutFile -Force -ErrorAction SilentlyContinue }
          return $false
        }
      }

      # Ensure outputs/live_lens_signals_<date>.jsonl is local so accuracy + tuning can be computed locally.
      # Always try a fresh download first so we don't accidentally use a stale/truncated local file.
      $signalsLocal = Join-Path $OutDir ("live_lens_signals_" + $prevDate + ".jsonl")
      $projectionsLocal = Join-Path $OutDir ("live_lens_projections_" + $prevDate + ".jsonl")
      try {
        $primary = $script:RenderBaseUrlEff
        $signalsTmp = Join-Path $OutDir ("_tmp_live_lens_signals_" + $prevDate + ".jsonl")
        $ok = Invoke-DownloadSignals -BaseUrl $primary -Date $prevDate -OutFile $signalsTmp
        if (-not $ok) {
          $fallback = 'https://ncaab.onrender.com'
          if ($primary.TrimEnd('/').ToLowerInvariant() -ne $fallback.ToLowerInvariant()) {
            $ok = Invoke-DownloadSignals -BaseUrl $fallback -Date $prevDate -OutFile $signalsTmp
          }
        }
        if ($ok -and (Test-HasBytes $signalsTmp)) {
          Move-Item -LiteralPath $signalsTmp -Destination $signalsLocal -Force
        } elseif (Test-Path -LiteralPath $signalsTmp) {
          Remove-Item -LiteralPath $signalsTmp -Force -ErrorAction SilentlyContinue
        }
      } catch {
        Write-Warning "[live_lens] Signals download wrapper failed: $($_)"
      }

      # Best-effort: download projections so we can compute projection accuracy.
      try {
        $primary = $script:RenderBaseUrlEff
        $projTmp = Join-Path $OutDir ("_tmp_live_lens_projections_" + $prevDate + ".jsonl")
        $okp = Invoke-DownloadProjections -BaseUrl $primary -Date $prevDate -OutFile $projTmp
        if (-not $okp) {
          $fallback = 'https://ncaab.onrender.com'
          if ($primary.TrimEnd('/').ToLowerInvariant() -ne $fallback.ToLowerInvariant()) {
            $okp = Invoke-DownloadProjections -BaseUrl $fallback -Date $prevDate -OutFile $projTmp
          }
        }
        if ($okp -and (Test-HasBytes $projTmp)) {
          Move-Item -LiteralPath $projTmp -Destination $projectionsLocal -Force
        } elseif (Test-Path -LiteralPath $projTmp) {
          Remove-Item -LiteralPath $projTmp -Force -ErrorAction SilentlyContinue
        }
      } catch {
        Write-Warning "[live_lens] Projections download wrapper failed: $($_)"
      }

      & $VenvPython -m ncaab_model.cli compute-live-lens-accuracy --date $prevDate

      # Counterfactual: re-score signals under current tuning retune penalties.
      & $VenvPython -m ncaab_model.cli compute-live-lens-accuracy-retuned --date $prevDate

      # Projection accuracy (MAE/RMSE) from logged snapshots.
      & $VenvPython -m ncaab_model.cli compute-live-lens-projection-accuracy --date $prevDate --full-game-only

      if ($RunLiveLensOverTuning.IsPresent) {
        Write-Section "3b.i.a) Live Lens OVER tuning sweep (lookback=$LiveLensOverTuningLookbackDays; apply=$($ApplyLiveLensOverTuning.IsPresent))"
        try {
          $prevDt = [datetime]::ParseExact($prevDate, 'yyyy-MM-dd', $null)
          $lb = [int]$LiveLensOverTuningLookbackDays
          if ($lb -lt 1) { $lb = 1 }
          $startDt = $prevDt.AddDays(-1 * ($lb - 1))
          $startIso = $startDt.ToString('yyyy-MM-dd')

          # Best-effort: download any missing signals files from Render for the lookback window.
          try {
            $downloaded = 0
            for ($i = 0; $i -lt $lb; $i++) {
              $dIso = $startDt.AddDays($i).ToString('yyyy-MM-dd')
              $p = Join-Path $OutDir ("live_lens_signals_${dIso}.jsonl")
              if (-not (Test-HasBytes $p)) {
                $primary = $script:RenderBaseUrlEff
                $ok = Invoke-DownloadSignals -BaseUrl $primary -Date $dIso -OutFile $p
                if (-not $ok) {
                  $fallback = 'https://ncaab.onrender.com'
                  if ($primary.TrimEnd('/').ToLowerInvariant() -ne $fallback.ToLowerInvariant()) {
                    $ok = Invoke-DownloadSignals -BaseUrl $fallback -Date $dIso -OutFile $p
                  }
                }
                if ($ok) { $downloaded++ }
              }
            }
            if ($downloaded -gt 0) { Write-Host "[live_lens] Downloaded ${downloaded} missing signals files" -ForegroundColor DarkGray }
          } catch {
            Write-Warning "[live_lens] Signals range download failed: $($_)"
          }

          $tuneArgs = @(
            'scripts/run_live_lens_over_tuning.py',
            '--start', "$startIso",
            '--end', "$prevDate",
            '--assume-price', "$LiveLensOverTuningAssumePrice",
            '--out-dir', "$OutDir",
            '--tuning-json', (Join-Path $OutDir 'live_lens_tuning.json'),
            '--min-bucket-n', "$LiveLensOverTuningMinBucketN",
            '--min-overall-n', "$LiveLensOverTuningMinOverallN"
          )
          if ($ApplyLiveLensOverTuning.IsPresent) { $tuneArgs += '--apply' }
          $tuneOut = (& $VenvPython @tuneArgs) | Out-String
          if ($tuneOut) { Write-Host ($tuneOut.Trim()) }

          # If we applied new tuning, re-score yesterday counterfactually with the updated tuning JSON.
          if ($ApplyLiveLensOverTuning.IsPresent) {
            try {
              & $VenvPython -m ncaab_model.cli compute-live-lens-accuracy-retuned --date $prevDate
            } catch {
              Write-Warning "compute-live-lens-accuracy-retuned failed (post-apply) for ${prevDate}: $($_)"
            }
          }
        } catch {
          Write-Warning "Live Lens OVER tuning sweep failed: $($_)"
        }
      } else {
        Write-Host '[skip] Live Lens OVER tuning sweep' -ForegroundColor Yellow
      }

      if ($RunLiveLensFlagPenalties.IsPresent) {
        Write-Section "3b.i.b) Live Lens flag penalties (lookback=$LiveLensFlagPenaltiesLookbackDays; apply=true)"
        try {
          # Learn a small set of strength penalties for historically-bad driver tags and
          # write them into outputs/live_lens_tuning.json under tuning.driver_tag_strength_penalties.
          # This is a safe filter: it only suppresses bad regimes; it does not flip sides.
          & $VenvPython -m ncaab_model.cli learn-live-lens-flag-penalties `
            --end-date $prevDate `
            --days $LiveLensFlagPenaltiesLookbackDays `
            --all-lenses `
            --tuning-json (Join-Path $OutDir 'live_lens_tuning.json')
        } catch {
          Write-Warning "Live Lens flag penalties learning failed: $($_)"
        }
      } else {
        Write-Host '[skip] Live Lens flag penalties learning' -ForegroundColor Yellow
      }
    } catch {
      Write-Warning "compute-live-lens-accuracy failed for ${prevDate}: $($_)"
    }

    # Live snapshots: sync from Render (refresh local history) + summarize + evaluate
    Write-Section "3b.ii) Live snapshots sync (Render->local) + summary + eval for $prevDate"
    try {
      $snapDir = Join-Path $OutDir 'live_snapshots'
      New-Item -ItemType Directory -Path $snapDir -Force | Out-Null
      $snapLocal = Join-Path $snapDir ("live_" + $prevDate + ".jsonl")

      function Test-HasBytes {
        param([string]$Path)
        if (-not (Test-Path -LiteralPath $Path)) { return $false }
        try { return ((Get-Item -LiteralPath $Path).Length -gt 0) } catch { return $false }
      }

      try {
        function Invoke-DownloadSnapshot {
          param(
            [string]$BaseUrl,
            [string]$Date,
            [string]$OutFile
          )
          $b = ("" + $BaseUrl).Trim()
          if ([string]::IsNullOrWhiteSpace($b)) { return $false }
          $b = $b.TrimEnd('/')
          $url = "$b/api/download_live_snapshots?date=$Date"
          Write-Host "[snapshots] Attempting download: $url" -ForegroundColor DarkGray
          try {
            Invoke-WebRequest -Uri $url -OutFile $OutFile -UseBasicParsing -TimeoutSec 30 | Out-Null
            return (Test-HasBytes $OutFile)
          } catch {
            $resp = $_.Exception.Response
            if ($resp) {
              Write-Warning ("[snapshots] Download failed ({0}): {1}" -f ([int]$resp.StatusCode), $_.Exception.Message)
            } else {
              Write-Warning "[snapshots] Download failed: $($_.Exception.Message)"
            }
            if (Test-Path -LiteralPath $OutFile) { Remove-Item -LiteralPath $OutFile -Force -ErrorAction SilentlyContinue }
            return $false
          }
        }

        function Sync-RemoteSnapshot {
          param(
            [string]$Date,
            [string]$LocalPath
          )
          $tmp = Join-Path $snapDir ("_tmp_live_" + $Date + ".jsonl")
          if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue }

          $primary = $script:RenderBaseUrlEff
          $ok = Invoke-DownloadSnapshot -BaseUrl $primary -Date $Date -OutFile $tmp
          if (-not $ok) {
            $fallback = 'https://ncaab.onrender.com'
            if ($primary.TrimEnd('/').ToLowerInvariant() -ne $fallback.ToLowerInvariant()) {
              $ok = Invoke-DownloadSnapshot -BaseUrl $fallback -Date $Date -OutFile $tmp
            }
          }
          if (-not ($ok -and (Test-HasBytes $tmp))) {
            if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue }
            return $false
          }

          $tmpLen = 0
          $locLen = 0
          try { $tmpLen = (Get-Item -LiteralPath $tmp).Length } catch { $tmpLen = 0 }
          try { if (Test-Path -LiteralPath $LocalPath) { $locLen = (Get-Item -LiteralPath $LocalPath).Length } } catch { $locLen = 0 }

          if ((-not (Test-Path -LiteralPath $LocalPath)) -or ($tmpLen -gt $locLen)) {
            Move-Item -LiteralPath $tmp -Destination $LocalPath -Force
            Write-Host "[snapshots] Synced ${Date}: bytes=$tmpLen" -ForegroundColor Green
            return $true
          } else {
            Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
            Write-Host "[snapshots] Up-to-date $Date (local bytes=$locLen, remote bytes=$tmpLen)" -ForegroundColor DarkGray
            return $true
          }
        }

        # Always sync previous-day snapshot (refresh local history)
        $null = Sync-RemoteSnapshot -Date $prevDate -LocalPath $snapLocal
      } catch {
        Write-Warning "[snapshots] Sync wrapper failed: $($_)"
      }

      if (Test-HasBytes $snapLocal) {
        try {
          & $VenvPython -m ncaab_model.cli summarize-live-snapshots --date $prevDate --in-path $snapLocal
        } catch {
          Write-Warning "summarize-live-snapshots failed for ${prevDate}: $($_)"
        }
        try {
          & $VenvPython -m ncaab_model.cli evaluate-live-snapshots --date $prevDate --snapshots-path $snapLocal
        } catch {
          Write-Warning "evaluate-live-snapshots failed for ${prevDate}: $($_)"
        }
        try {
          & $VenvPython -m ncaab_model.cli build-live-features --date $prevDate --snapshots-path $snapLocal
        } catch {
          Write-Warning "build-live-features failed for ${prevDate}: $($_)"
        }
      } else {
        Write-Host "[snapshots] No snapshots file for $prevDate; skipping summary/eval." -ForegroundColor DarkGray
      }
    } catch {
      Write-Warning "Live snapshot summary/eval step failed for ${prevDate}: $($_)"
    }
  } else {
    Write-Host "SkipFinalizePrev flag set; skipping finalize-day for $prevDate." -ForegroundColor Yellow
  }

  # Sync today's live snapshots from Render so we maintain a local archive of intraday polling.
  Write-Section "3c) Live snapshots sync (Render->local) for $todayIso (archive intraday polling)"
  try {
    $snapDir = Join-Path $OutDir 'live_snapshots'
    New-Item -ItemType Directory -Path $snapDir -Force | Out-Null
    $snapTodayLocal = Join-Path $snapDir ("live_" + $todayIso + ".jsonl")

    function Test-HasBytes {
      param([string]$Path)
      if (-not (Test-Path -LiteralPath $Path)) { return $false }
      try { return ((Get-Item -LiteralPath $Path).Length -gt 0) } catch { return $false }
    }

    function Invoke-DownloadSnapshot {
      param(
        [string]$BaseUrl,
        [string]$Date,
        [string]$OutFile
      )
      $b = ("" + $BaseUrl).Trim()
      if ([string]::IsNullOrWhiteSpace($b)) { return $false }
      $b = $b.TrimEnd('/')
      $url = "$b/api/download_live_snapshots?date=$Date"
      Write-Host "[snapshots] Attempting download: $url" -ForegroundColor DarkGray
      try {
        Invoke-WebRequest -Uri $url -OutFile $OutFile -UseBasicParsing -TimeoutSec 30 | Out-Null
        return (Test-HasBytes $OutFile)
      } catch {
        $resp = $_.Exception.Response
        if ($resp) {
          Write-Warning ("[snapshots] Download failed ({0}): {1}" -f ([int]$resp.StatusCode), $_.Exception.Message)
        } else {
          Write-Warning "[snapshots] Download failed: $($_.Exception.Message)"
        }
        if (Test-Path -LiteralPath $OutFile) { Remove-Item -LiteralPath $OutFile -Force -ErrorAction SilentlyContinue }
        return $false
      }
    }

    $tmp = Join-Path $snapDir ("_tmp_live_" + $todayIso + ".jsonl")
    if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue }
    $primary = $script:RenderBaseUrlEff
    $ok = Invoke-DownloadSnapshot -BaseUrl $primary -Date $todayIso -OutFile $tmp
    if (-not $ok) {
      $fallback = 'https://ncaab.onrender.com'
      if ($primary.TrimEnd('/').ToLowerInvariant() -ne $fallback.ToLowerInvariant()) {
        $ok = Invoke-DownloadSnapshot -BaseUrl $fallback -Date $todayIso -OutFile $tmp
      }
    }
    if ($ok -and (Test-HasBytes $tmp)) {
      $tmpLen = 0
      $locLen = 0
      try { $tmpLen = (Get-Item -LiteralPath $tmp).Length } catch { $tmpLen = 0 }
      try { if (Test-Path -LiteralPath $snapTodayLocal) { $locLen = (Get-Item -LiteralPath $snapTodayLocal).Length } } catch { $locLen = 0 }
      if ((-not (Test-Path -LiteralPath $snapTodayLocal)) -or ($tmpLen -gt $locLen)) {
        Move-Item -LiteralPath $tmp -Destination $snapTodayLocal -Force
        Write-Host "[snapshots] Synced ${todayIso}: bytes=$tmpLen" -ForegroundColor Green
      } else {
        Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
        Write-Host "[snapshots] Up-to-date $todayIso (local bytes=$locLen, remote bytes=$tmpLen)" -ForegroundColor DarkGray
      }
    } else {
      if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue }
      Write-Host "[snapshots] No snapshots file for $todayIso on remote (yet)." -ForegroundColor DarkGray
    }
  } catch {
    Write-Warning "Live snapshot sync failed for ${todayIso}: $($_)"
  }

  # 3d) Compute daily accuracy snapshot for previous day and persist JSON
  Write-Section "3d) Compute daily accuracy for $prevDate (winners/totals/ATS)"
  try {
    $accOut = (& $VenvPython scripts/compute_daily_accuracy.py $prevDate) | Out-String
    $accPath = Join-Path $OutDir ("daily_accuracy_" + $prevDate + ".json")
    $accOut.Trim() | Out-File -FilePath $accPath -Encoding UTF8
    # Also mirror a copy under metrics for UI consumption
    $metricsDir = Join-Path $OutDir 'metrics'
    New-Item -ItemType Directory -Path $metricsDir -Force | Out-Null
    Copy-Item -LiteralPath $accPath -Destination (Join-Path $metricsDir ("daily_accuracy_" + $prevDate + ".json")) -Force
    # Maintain a latest symlink-like copy
    Copy-Item -LiteralPath $accPath -Destination (Join-Path $metricsDir 'daily_accuracy_latest.json') -Force
    Write-Host ("[metrics] Wrote {0}" -f $accPath)
  } catch {
    Write-Warning "compute_daily_accuracy failed for ${prevDate}: $($_)"
  }

  # 3f) Evaluate simulation outputs vs finalized results (proper scoring)
  Write-Section "3f) Evaluate simulation metrics for $prevDate"
  try {
    $simEvalOut = (& $VenvPython scripts/evaluate_sim_metrics.py $prevDate --outputs $OutDir) | Out-String
    $metricsDir = Join-Path $OutDir 'metrics'
    New-Item -ItemType Directory -Path $metricsDir -Force | Out-Null
    $simEvalPath = Join-Path $metricsDir ("sim_metrics_summary_" + $prevDate + ".json")
    $simEvalOut.Trim() | Out-File -FilePath $simEvalPath -Encoding UTF8
    Write-Host "[metrics] Wrote $simEvalPath"
  } catch {
    Write-Warning "evaluate_sim_metrics failed for ${prevDate}: $($_)"
  }

  # 3f.g) Reconcile 5-min cumulative endpoints (5..40) vs ESPN play-by-play, and refresh calibration
  if (-not $SkipSegment5MinRecon.IsPresent) {
    Write-Section "3f.g) Reconcile 5-min endpoints + refresh segment calibration for $prevDate"
    try {
      $segPrefix = 'segments_5min_daily'
      $btArgs = @(
        'backtest-segments-5min',
        '--start', "$prevDate",
        '--end', "$prevDate",
        '--engine', "$Segment5MinBacktestEngine",
        '--samples', "$Segment5MinBacktestSamples",
        '--out-prefix', "$segPrefix",
        '--sleep-seconds', '0.10'
      )
      $segBtOut = (& $VenvPython -m ncaab_model.cli @btArgs) | Out-String
      if ($segBtOut) { Write-Host ($segBtOut.Trim()) }

      $btDir = Join-Path $OutDir 'backtests'
      $dailyCsv = Join-Path $btDir ("${segPrefix}_${prevDate}_to_${prevDate}.csv")
      $masterCsv = Join-Path $btDir 'segments_5min_master.csv'

      if (Test-Path $dailyCsv) {
        $upsertOut = (& $VenvPython scripts/upsert_segments_5min_master.py --daily $dailyCsv --master $masterCsv) | Out-String
        if ($upsertOut) { Write-Host ($upsertOut.Trim()) }

        if (Test-Path $masterCsv) {
          $calStart = $todayDate.AddDays(-[int]$Segment5MinCalibrationLookbackDays).ToString('yyyy-MM-dd')
          $calOut = (& $VenvPython scripts/refresh_segment_calibration_5min.py `
            --backtest-csv $masterCsv `
            --out (Join-Path $OutDir 'segment_calibration_5min.json') `
            --start $calStart `
            --end $prevDate `
            --min-rows-per-end-min $Segment5MinCalibrationMinRowsPerEndMin `
            --min-rows-used $Segment5MinCalibrationMinRowsUsed `
            --min-endpoints $Segment5MinCalibrationMinEndpoints
          ) | Out-String
          if ($calOut) { Write-Host ($calOut.Trim()) }

          # Stage 2: residual bias correction (fit from a stage2-disabled baseline window)
          try {
            $stage2WindowDays = 14
            $prevDt = [datetime]::ParseExact($prevDate, 'yyyy-MM-dd', $null)
            $stage2Start = $prevDt.AddDays(-($stage2WindowDays - 1)).ToString('yyyy-MM-dd')
            $stage2Prefix = 'segments_5min_stage2off_window'
            $stage2BtArgs = @(
              'backtest-segments-5min',
              '--start', "$stage2Start",
              '--end', "$prevDate",
              '--engine', "$Segment5MinBacktestEngine",
              '--samples', "$Segment5MinBacktestSamples",
              '--out-prefix', "$stage2Prefix",
              '--sleep-seconds', '0.10',
              '--recompute-sims'
            )

            $prevDisableStage2 = $env:NCAAB_DISABLE_SEGMENT_CALIB_STAGE2
            try {
              $env:NCAAB_DISABLE_SEGMENT_CALIB_STAGE2 = '1'
              $stage2BtOut = (& $VenvPython -m ncaab_model.cli @stage2BtArgs) | Out-String
              if ($stage2BtOut) { Write-Host ($stage2BtOut.Trim()) }
            } finally {
              if ($null -eq $prevDisableStage2 -or $prevDisableStage2 -eq '') {
                Remove-Item Env:\NCAAB_DISABLE_SEGMENT_CALIB_STAGE2 -ErrorAction SilentlyContinue
              } else {
                $env:NCAAB_DISABLE_SEGMENT_CALIB_STAGE2 = $prevDisableStage2
              }
            }

            $stage2Csv = Join-Path $btDir ("${stage2Prefix}_${stage2Start}_to_${prevDate}.csv")
            if (Test-Path $stage2Csv) {
              $stage2Out = (& $VenvPython scripts/refresh_segment_stage2_bias_5min.py `
                --backtest-csv $stage2Csv `
                --out (Join-Path $OutDir 'segment_calibration_stage2_5min.json') `
                --end $prevDate `
                --window-days $stage2WindowDays `
                --end-mins '5,10,15,20,25,30,35,40' `
                --zero-end-mins '20,40' `
                --merge-existing `
                --min-rows-per-end-min $Segment5MinCalibrationMinRowsPerEndMin `
                --min-endpoints 8 `
                --min-rows-used ([int]$Segment5MinCalibrationMinRowsPerEndMin * 8) `
                --stat mean
              ) | Out-String
              if ($stage2Out) { Write-Host ($stage2Out.Trim()) }
            } else {
              Write-Warning "[segments-5min] Stage2 baseline backtest CSV not found: $stage2Csv"
            }
          } catch {
            Write-Warning "segments-5min stage2 bias fit failed for ${prevDate}: $($_)"
          }
        } else {
          Write-Warning "[segments-5min] Master backtest CSV not found: $masterCsv"
        }
      } else {
        Write-Warning "[segments-5min] Daily backtest CSV not found: $dailyCsv"
      }
    } catch {
      Write-Warning "segments-5min reconciliation/calibration failed for ${prevDate}: $($_)"
    }
  } else {
    Write-Host "SkipSegment5MinRecon flag set; skipping 5-min segments reconciliation/calibration." -ForegroundColor Yellow
  }

  # 3f.h) Build interval actuals CSV for UI reconciliation (team scores at 5..40)
  if (-not $SkipIntervalActuals5Min.IsPresent) {
    Write-Section "3f.h) Build interval actuals (5-min) for $prevDate"
    try {
      $iaArgv = @(
        'build-interval-actuals-5min',
        '--date', "$prevDate",
        '--sleep-seconds', '0.10',
        '--include-ot-endpoints',
        '--max-ot-periods', '4'
      )
      if ($NoCache.IsPresent) { $iaArgv += '--no-use-cache' }

      $iaOut = (& $VenvPython -m ncaab_model.cli @iaArgv) | Out-String
      if ($iaOut) { Write-Host ($iaOut.Trim()) }
    } catch {
      Write-Warning "build-interval-actuals-5min failed for ${prevDate}: $($_)"
    }
  } else {
    Write-Host "SkipIntervalActuals5Min flag set; skipping 5-min interval actuals build." -ForegroundColor Yellow
  }

  # 3f.i) Evaluate sim accuracy (winners/totals/ATS) for previous day and persist JSON
  if ($RunSimAccuracyBacktest.IsPresent) {
    Write-Section "3f.i) Evaluate sim accuracy for $prevDate (winners/totals/ATS)"
    try {
      $cliArgv = @('backtest-sim-accuracy', '--start', "$prevDate", '--end', "$prevDate", '--engine', "$SimAccuracyBacktestEngine", '--samples', "$SimAccuracyBacktestSamples")
      if ($SimAccuracyBacktestRecompute.IsPresent) { $cliArgv += '--recompute' }
      $btOut = (& $VenvPython -m ncaab_model.cli @cliArgv) | Out-String
      if ($btOut) { Write-Host ($btOut.Trim()) }

      $metricsDir = Join-Path $OutDir 'metrics'
      New-Item -ItemType Directory -Path $metricsDir -Force | Out-Null
      $btDir = Join-Path $OutDir 'backtests'
      if (Test-Path $btDir) {
        $latest = Get-ChildItem $btDir -File -Filter "sim_accuracy_${prevDate}_${prevDate}_summary.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($latest) {
          Copy-Item -LiteralPath $latest.FullName -Destination (Join-Path $metricsDir ("sim_accuracy_" + $prevDate + ".json")) -Force
          Copy-Item -LiteralPath $latest.FullName -Destination (Join-Path $metricsDir 'sim_accuracy_latest.json') -Force
          Write-Host "[metrics] Wrote $(Join-Path $metricsDir ("sim_accuracy_" + $prevDate + ".json"))"
        }
      }
    } catch {
      Write-Warning "sim accuracy backtest failed: $($_)"
    }
  } else {
    Write-Host '[skip] Sim accuracy backtest' -ForegroundColor Yellow
  }

  # 3g) Backtest sim engine across recent finalized days (heavier, optional)
  if ($RunSimBacktest.IsPresent) {
    Write-Section "3g) Backtest sim engine (recent=$SimBacktestRecent, engine=$SimBacktestEngine, samples=$SimBacktestSamples)"
    try {
      $cliArgv = @('backtest-sim-engine', '--recent', "$SimBacktestRecent", '--engine', "$SimBacktestEngine", '--samples', "$SimBacktestSamples")
      if ($SimBacktestRecompute.IsPresent) { $cliArgv += '--recompute' }
      $btOut = (& $VenvPython -m ncaab_model.cli @cliArgv) | Out-String
      if ($btOut) { Write-Host ($btOut.Trim()) }

      # Copy newest summary into metrics as a "latest" snapshot for quick inspection.
      $metricsDir = Join-Path $OutDir 'metrics'
      New-Item -ItemType Directory -Path $metricsDir -Force | Out-Null
      $btDir = Join-Path $OutDir 'backtests'
      if (Test-Path $btDir) {
        $latest = Get-ChildItem $btDir -File -Filter 'sim_engine_*_summary.json' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($latest) {
          Copy-Item -LiteralPath $latest.FullName -Destination (Join-Path $metricsDir 'sim_engine_backtest_latest.json') -Force
          Write-Host "[metrics] Wrote $(Join-Path $metricsDir 'sim_engine_backtest_latest.json')"
        }
      }
    } catch {
      Write-Warning "sim engine backtest failed: $($_)"
    }
  } else {
    Write-Host '[skip] Sim engine backtest' -ForegroundColor Yellow
  }

  # 3h) Backtest prediction accuracy across recent finalized days (optional)
  if ($RunAccuracyBacktest.IsPresent) {
    Write-Section "3h) Backtest accuracy (recent=$AccuracyBacktestRecent)"
    try {
      $cliArgv = @('backtest-accuracy', '--recent', "$AccuracyBacktestRecent")
      $btOut = (& $VenvPython -m ncaab_model.cli @cliArgv) | Out-String
      if ($btOut) { Write-Host ($btOut.Trim()) }

      # Copy newest summary into metrics as a "latest" snapshot.
      $metricsDir = Join-Path $OutDir 'metrics'
      New-Item -ItemType Directory -Path $metricsDir -Force | Out-Null
      $btDir = Join-Path $OutDir 'backtests'
      if (Test-Path $btDir) {
        $latest = Get-ChildItem $btDir -File -Filter 'accuracy_*_summary.json' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($latest) {
          Copy-Item -LiteralPath $latest.FullName -Destination (Join-Path $metricsDir 'accuracy_backtest_latest.json') -Force
          Write-Host "[metrics] Wrote $(Join-Path $metricsDir 'accuracy_backtest_latest.json')"
        }
      }
    } catch {
      Write-Warning "accuracy backtest failed: $($_)"
    }
  } else {
    Write-Host '[skip] Accuracy backtest' -ForegroundColor Yellow
  }

  # 3h.i) Backtest sim accuracy across recent finalized days (optional)
  if ($RunSimAccuracyBacktest.IsPresent) {
    Write-Section "3h.i) Backtest sim accuracy (recent=$SimAccuracyBacktestRecent, engine=$SimAccuracyBacktestEngine, samples=$SimAccuracyBacktestSamples)"
    try {
      $cliArgv = @('backtest-sim-accuracy', '--recent', "$SimAccuracyBacktestRecent", '--engine', "$SimAccuracyBacktestEngine", '--samples', "$SimAccuracyBacktestSamples")
      if ($SimAccuracyBacktestRecompute.IsPresent) { $cliArgv += '--recompute' }
      $btOut = (& $VenvPython -m ncaab_model.cli @cliArgv) | Out-String
      if ($btOut) { Write-Host ($btOut.Trim()) }

      $metricsDir = Join-Path $OutDir 'metrics'
      New-Item -ItemType Directory -Path $metricsDir -Force | Out-Null
      $btDir = Join-Path $OutDir 'backtests'
      if (Test-Path $btDir) {
        $latest = Get-ChildItem $btDir -File -Filter 'sim_accuracy_*_summary.json' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($latest) {
          Copy-Item -LiteralPath $latest.FullName -Destination (Join-Path $metricsDir 'sim_accuracy_backtest_latest.json') -Force
          Write-Host "[metrics] Wrote $(Join-Path $metricsDir 'sim_accuracy_backtest_latest.json')"
        }
      }
    } catch {
      Write-Warning "sim accuracy rolling backtest failed: $($_)"
    }
  }

  # 3h.ii) Live Lens interval evaluation (optional): compare 15s vs 30s sampling using cached ESPN PBP
  if ($RunLiveIntervalEval.IsPresent) {
    Write-Section "3h.ii) Live Lens interval eval (lookback=$LiveIntervalEvalLookbackDays days; max_files=$LiveIntervalEvalMaxFiles)"
    try {
      $prevDt = [datetime]::ParseExact($prevDate, 'yyyy-MM-dd', $null)
      $lb = [int]$LiveIntervalEvalLookbackDays
      if ($lb -lt 1) { $lb = 1 }
      $startDt = $prevDt.AddDays(-1 * ($lb - 1))
      $startIso = $startDt.ToString('yyyy-MM-dd')

      $csv15 = Join-Path $OutDir ("live_interval_backtest_${startIso}_${prevDate}_15s.csv")
      $csv30 = Join-Path $OutDir ("live_interval_backtest_${startIso}_${prevDate}_30s.csv")

      $bt15Args = @(
        'backtest-live-intervals',
        '--start-date', "$startIso",
        '--end-date', "$prevDate",
        '--step-sec', '15',
        '--out-csv', "$csv15",
        '--max-files', "$LiveIntervalEvalMaxFiles"
      )
      $bt30Args = @(
        'backtest-live-intervals',
        '--start-date', "$startIso",
        '--end-date', "$prevDate",
        '--step-sec', '30',
        '--out-csv', "$csv30",
        '--max-files', "$LiveIntervalEvalMaxFiles"
      )

      $o15 = (& $VenvPython -m ncaab_model.cli @bt15Args) | Out-String
      if ($o15) { Write-Host ($o15.Trim()) }
      $o30 = (& $VenvPython -m ncaab_model.cli @bt30Args) | Out-String
      if ($o30) { Write-Host ($o30.Trim()) }

      if ((Test-Path -LiteralPath $csv15) -and (Test-Path -LiteralPath $csv30)) {
        $cmpPrefix = Join-Path $OutDir ("live_interval_compare_${startIso}_${prevDate}_15s_vs_30s")
        $cmpArgs = @(
          'compare-live-intervals',
          '--csv-a', "$csv15",
          '--csv-b', "$csv30",
          '--label-a', '15s',
          '--label-b', '30s',
          '--out-prefix', "$cmpPrefix"
        )
        $cmpOut = (& $VenvPython -m ncaab_model.cli @cmpArgs) | Out-String
        if ($cmpOut) { Write-Host ($cmpOut.Trim()) }
      } else {
        Write-Warning "[live-interval-eval] Missing backtest CSVs; skipping compare. csv15=$csv15 csv30=$csv30"
      }
    } catch {
      Write-Warning "Live interval eval failed: $($_)"
    }
  } else {
    Write-Host '[skip] Live Lens interval eval' -ForegroundColor Yellow
  }

  # 3e) Fit global simulation calibration (from recent finalized results + existing sim outputs)
  # This writes outputs/sim_calibration.json which is applied automatically by the sim engine
  # when generating today's sim quantiles.
  if (-not $SkipSimCalibrationFit) {
    Write-Section "3e) Fit simulation calibration (lookback=$SimCalibrationLookbackDays days, min_games=$SimCalibrationMinGames)"
    try {
      $prevDt = [datetime]::ParseExact($prevDate, 'yyyy-MM-dd', $null)
      $lookback = $SimCalibrationLookbackDays
      if ($env:SIM_CALIB_LOOKBACK_DAYS) {
        try { $lookback = [int]$env:SIM_CALIB_LOOKBACK_DAYS } catch {}
      }
      if ($lookback -lt 1) { $lookback = 1 }
      $startDt = $prevDt.AddDays(-1 * ($lookback - 1))
      $startIso = $startDt.ToString('yyyy-MM-dd')

      $fitArgs = @(
        'scripts/fit_sim_calibration.py',
        '--outputs', $OutDir,
        '--start', $startIso,
        '--end', $prevDate,
        '--min-games', "$SimCalibrationMinGames",
        '--no-accumulate',
        '--cap-sigma-mult', '3.0',
        '--cap-sigma-1h-mult', '1.5',
        '--cap-abs-delta', '25.0',
        '--cap-abs-delta-1h', '15.0'
      )

      # Winner/margin calibration uplift: allow scaling margin mean + spread-binned adjustments.
      # Opt-in via env vars while we validate ATS impact in sim backtests.
      $fitMarginScale = $false
      if ($env:SIM_CALIB_FIT_MARGIN_SCALE) {
        try { $fitMarginScale = ([int]$env:SIM_CALIB_FIT_MARGIN_SCALE -ne 0) } catch {}
      }
      $writeSpreadBins = $false
      if ($env:SIM_CALIB_WRITE_SPREAD_BINS) {
        try { $writeSpreadBins = ([int]$env:SIM_CALIB_WRITE_SPREAD_BINS -ne 0) } catch {}
      }

      $minGamesBin = $null
      if ($writeSpreadBins) {
        # Default tuned via sim backtests: enables more bins (incl. positive spreads)
        # and improved full-game ATS on eval windows.
        $minGamesBin = '30'
        if ($env:SIM_CALIB_MIN_GAMES_BIN) {
          try {
            $mg = [int]$env:SIM_CALIB_MIN_GAMES_BIN
            if ($mg -gt 0) { $minGamesBin = "$mg" }
          } catch {}
        }
      }

      $spreadBinsObjective = $null
      if ($env:SIM_CALIB_SPREAD_BINS_OBJECTIVE) {
        try { $spreadBinsObjective = ("$($env:SIM_CALIB_SPREAD_BINS_OBJECTIVE)".Trim().ToLower()) } catch {}
        if (@('resid','ats') -notcontains $spreadBinsObjective) { $spreadBinsObjective = $null }
      }

      if ($fitMarginScale) {
        $fitArgs += @('--fit-margin-scale', '--cap-margin-scale', '2.0')
      }
      if ($writeSpreadBins) {
        $fitArgs += '--write-spread-bins'
        if ($spreadBinsObjective) {
          $fitArgs += @('--spread-bins-objective', $spreadBinsObjective)
        }
        if ($minGamesBin) {
          $fitArgs += @('--min-games-bin', $minGamesBin)
        }
      }
      if ($SimCalibrationFit1HOnly.IsPresent) {
        $fitArgs += '--fit-1h-only'
        Write-Host '[sim-cal] 1H-only mode enabled (preserving full-game keys)' -ForegroundColor DarkGray
      }

      $fitOut = (& $VenvPython @fitArgs) | Out-String
      if ($fitOut) {
        Write-Host ($fitOut.Trim())
      }
    } catch {
      Write-Warning "fit_sim_calibration failed: $($_)"
    }
  } else {
    Write-Host '[skip] Simulation calibration fit' -ForegroundColor Yellow
  }

  # 3f) Fit isotonic probability calibration (p_over / p_home_cover_dist)
  # Writes outputs/calibration_params.json used by bankroll-optimize when --isotonic-prob-calibration is set.
  if (-not $SkipProbCalibrationFit) {
    Write-Section "3f) Fit isotonic probability calibration (lookback=$ProbCalibrationLookbackDays days, min_rows=$ProbCalibrationMinRows)"
    try {
      $prevDt = [datetime]::ParseExact($prevDate, 'yyyy-MM-dd', $null)
      $lookback = $ProbCalibrationLookbackDays
      if ($env:PROB_CALIB_LOOKBACK_DAYS) {
        try { $lookback = [int]$env:PROB_CALIB_LOOKBACK_DAYS } catch {}
      }
      if ($lookback -lt 7) { $lookback = 7 }
      $startDt = $prevDt.AddDays(-1 * ($lookback - 1))
      $startIso = $startDt.ToString('yyyy-MM-dd')
      $minRows = $ProbCalibrationMinRows
      if ($env:PROB_CALIB_MIN_ROWS) {
        try { $minRows = [int]$env:PROB_CALIB_MIN_ROWS } catch {}
      }
      $probOut = (& $VenvPython scripts/calibrate_probs_history.py --outputs $OutDir --start $startIso --end $prevDate --min-rows $minRows) | Out-String
      if ($probOut) { Write-Host ($probOut.Trim()) }
    } catch {
      Write-Warning "probability calibration fit failed: $($_)"
    }
  } else {
    Write-Host '[skip] Probability calibration fit' -ForegroundColor Yellow
  }

  Write-Section '4) Update model tuning from recent daily results'
  & $VenvPython -m ncaab_model.cli update-tuning --results-dir (Join-Path $OutDir 'daily_results') --window-days 7 --min-valid-games 10 --cap-abs-bias 25 --out (Join-Path $OutDir 'model_tuning.json')

  if (-not $SkipRetrain) {
    Write-Section '5) Retrain baseline models on latest features_all.csv'
    $featuresAll = Join-Path $OutDir 'features_all.csv'
    if (-not (Test-Path $featuresAll)) {
      Write-Host 'features_all.csv not found; building from games_all.csv (schedule+ratings+four-factors if available)'
      $gamesAll = Join-Path $OutDir 'games_all.csv'
      $boxscores = Join-Path $OutDir 'boxscores.csv'
      if (Test-Path $boxscores) {
        & $VenvPython -m ncaab_model.cli build-features $gamesAll --boxscores-path $boxscores --out $featuresAll
      } else {
        & $VenvPython -m ncaab_model.cli build-features $gamesAll --out $featuresAll
      }
    }
    & $VenvPython -m ncaab_model.cli train-baseline $featuresAll --out-dir (Join-Path $OutDir 'models') --loss-totals huber --huber-delta 8.0
  } else {
    Write-Host 'SkipRetrain flag set; using existing models.'
  }

  # Build engineered features for quantiles (rest/rolling) + (legacy overall quantile training block)
  if ($RunHeavyQuantiles) {
    try { & $VenvPython scripts/build_features.py } catch { Write-Warning "build_features.py failed: $($_)" }
    try { & $VenvPython scripts/train_quantiles_cv.py } catch { Write-Warning "train_quantiles_cv.py failed: $($_)" }
    try { & $VenvPython scripts/train_quantiles.py } catch { Write-Warning "train_quantiles.py failed: $($_)" }
  } else {
    Write-Host '[skip] Heavy quantile CV + base quantile retrain (weekly gating)' -ForegroundColor Yellow
  }

  # Generate team-level historical features EARLY so inference has the freshest aggregates
  Write-Section '5b) Generate/refresh team-level historical features (pre-inference)'
  try {
    & $VenvPython -m src.modeling.team_features --out (Join-Path $OutDir 'team_features.csv')
  } catch { Write-Warning "team_features pre-inference generation failed: $($_)" }

  # Ensure deterministic per-day feature rows exist for inference (lightweight placeholder ratings)
  $featuresCurr = Join-Path $OutDir 'features_curr.csv'
  $needsFeaturesRefresh = $true
  if (Test-Path $featuresCurr) {
    try {
      $rows = Import-Csv -Path $featuresCurr
      if ($null -ne $rows -and $rows.Count -gt 0) {
        # If a date column exists and matches today for any row, we can skip
        $hasDateColumn = $rows[0].PSObject.Properties.Name -contains 'date'
        if ($hasDateColumn) {
          $todayRows = $rows | Where-Object { $_.date -eq $todayIso }
          if ($todayRows -and $todayRows.Count -gt 0) { $needsFeaturesRefresh = $false }
        } else {
          # No date column means ambiguous content; force refresh
          $needsFeaturesRefresh = $true
        }
      }
    } catch { Write-Warning "Failed probing features_curr.csv; forcing refresh: $($_)"; $needsFeaturesRefresh = $true }
  }
  if ($needsFeaturesRefresh) {
    Write-Section "5c) Generate today's placeholder features (features_curr.csv)"
    try {
      & $VenvPython -m src.modeling.gen_features_today --date $todayIso --write-dated
    } catch { Write-Warning "gen_features_today failed: $($_)" }
  } else {
    Write-Host "features_curr.csv contains rows for $todayIso; skipping generation."
  }

  # Force fresh prediction artifacts: always remove and regenerate today's model predictions, calibration & intervals
  $modelPredPath = Join-Path $OutDir ("predictions_model_" + $todayIso + ".csv")
  $calibratedPath = Join-Path $OutDir ("predictions_model_calibrated_" + $todayIso + ".csv")
  $intervalPath = Join-Path $OutDir ("predictions_model_interval_" + $todayIso + ".csv")
  foreach ($p in @($intervalPath,$calibratedPath,$modelPredPath)) { if (Test-Path $p) { Write-Host "Removing stale artifact -> $p" -ForegroundColor DarkGray; Remove-Item $p -Force } }

  Write-Section '5d) Run model inference harness (forced refresh)'
  try {
    & $VenvPython -m src.modeling.infer --date $todayIso
  } catch { Write-Warning "model inference failed: $($_)" }

  Write-Section '5e) Calibrate model predictions (forced refresh)'
  try {
    & $VenvPython -m src.modeling.calibrate_predictions --date $todayIso --predictions-file $modelPredPath --results-dir (Join-Path $OutDir 'daily_results') --window-days 14
  } catch { Write-Warning "calibration failed: $($_)" }

  Write-Section '5f) Generate prediction intervals (forced refresh)'
  try {
    if (Test-Path $calibratedPath) {
      & $VenvPython -m src.modeling.interval_predictions --date $todayIso --predictions-file $modelPredPath --calibrated-file $calibratedPath --results-dir (Join-Path $OutDir 'daily_results') --window-days 30
    } else {
      & $VenvPython -m src.modeling.interval_predictions --date $todayIso --predictions-file $modelPredPath --results-dir (Join-Path $OutDir 'daily_results') --window-days 30
    }
  } catch { Write-Warning "interval predictions failed: $($_)" }

  # Auto-recalibrate conformal buffers (writes scale hints JSON)
  try {
    & $VenvPython scripts/auto_recalibrate_conformal.py
  } catch { Write-Warning "auto_recalibrate_conformal.py failed: $($_)" }

  # Rolling backtest over trailing 28 days and publish latest summary
  Write-Section '6i) Rolling backtest (28 days)'
  try {
    & $VenvPython scripts/backtest_models.py --days 28 --name latest
    # Normalize latest summary/daily filenames for UI consumption
    $btSummaries = Get-ChildItem -Path $OutDir -Filter 'backtest_summary_latest_*.csv' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    if ($btSummaries -and $btSummaries.Count -gt 0) {
      Copy-Item $btSummaries[0].FullName (Join-Path $OutDir 'backtest_summary_latest.csv') -Force
    }
    $btDailies = Get-ChildItem -Path $OutDir -Filter 'backtest_daily_latest_*.csv' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    if ($btDailies -and $btDailies.Count -gt 0) {
      Copy-Item $btDailies[0].FullName (Join-Path $OutDir 'backtest_daily_latest.csv') -Force
    }
  } catch { Write-Warning "backtest_models.py failed: $($_)" }

  # Generate ATS picks for today (non-strict prob-side; fallback to delta where missing)
  Write-Section '6i.b) Generate ATS picks (today)'
  try {
    & $VenvPython scripts/select_ats_picks.py --date $todayIso --use-closing --prob-side --prob-side-threshold 0.52
    # Convert ATS picks into picks_raw.csv for uploader/recommendations
    & $VenvPython scripts/make_picks_raw_from_ats.py --date $todayIso --outputs $OutDir

    # If selector produced no ATS picks (or ats_picks is empty), synthesize full-slate ATS from display
    $atsToday = Join-Path $OutDir ("picks\ats_picks_" + $todayIso + ".csv")
    $atsRowsLocal = 0
    if (Test-Path -Path $atsToday -PathType Leaf) {
      try {
        $atsRowsLocal = ((Get-Content $atsToday | Measure-Object).Count - 1)
        if ($atsRowsLocal -lt 0) { $atsRowsLocal = 0 }
      } catch { $atsRowsLocal = 0 }
    }
    if ($atsRowsLocal -le 0) {
      Write-Host ("[ATS] Selector yielded 0 rows; building fallback from display for {0}" -f $todayIso) -ForegroundColor DarkCyan
      try {
        & $VenvPython scripts/build_ats_picks_from_display.py $todayIso
      } catch {
        Write-Warning ("build_ats_picks_from_display failed in 6i.b: {0}" -f $_.Exception.Message)
      }
      # Rebuild picks_raw from synthesized ATS picks so downstream steps and uploads see ATS
      if (Test-Path -Path $atsToday -PathType Leaf) {
        try {
          & $VenvPython scripts/make_picks_raw_from_ats.py --date $todayIso --outputs $OutDir
        } catch {
          Write-Warning ("make_picks_raw_from_ats failed in 6i.b after fallback: {0}" -f $_.Exception.Message)
        }
      }
    }
  } catch { Write-Warning "ATS picks generation failed: $($_)" }

  # Normalize odds and enrich backtest for quantile training/selection
  Write-Section '6i.a) Normalize odds and enrich backtest (rest/travel/market)'
  try {
    & $VenvPython scripts/normalize_odds.py
  } catch { Write-Warning "normalize_odds.py failed: $($_)" }
  try {
    & $VenvPython scripts/enrich_backtest_features.py
  } catch { Write-Warning "enrich_backtest_features.py failed: $($_)" }
  # Refresh segment-specific LGBM quantile models using CV-selected features (if available)
  if ($RunHeavyQuantiles) {
    try { & $VenvPython scripts/train_quantiles_cv.py } catch { Write-Warning "train_quantiles_cv.py (pre-selection refresh) failed: $($_)" }
    try { & $VenvPython scripts/train_quantiles_lgbm.py } catch { Write-Warning "train_quantiles_lgbm.py failed: $($_)" }
  } else {
    Write-Host '[skip] Segment LGBM quantile retrain (weekly gating)' -ForegroundColor Yellow
  }

  # Quantile selection for today's slate using residual-based central intervals
  Write-Section '6j) Quantile selection (28d window, 80% coverage)'
  try {
    & $VenvPython scripts/select_quantiles_multi.py --window-days 28 --target-coverage 0.8
  } catch {
    Write-Warning "select_quantiles_multi.py failed: $($_). Falling back to simple residual-based selection."
    try { & $VenvPython scripts/select_quantiles.py --window-days 28 --target-coverage 0.8 } catch { Write-Warning "select_quantiles.py failed: $($_)" }
  }

  # Tune OU segment thresholds over trailing 28 days before generating totals picks
  Write-Section '6j.a) Tune OU segment thresholds (28 days)'
  try {
    & $VenvPython scripts/tune_ou_segment_thresholds.py --days 28 --outputs $OutDir --min-per-segment 30
  } catch { Write-Warning "tune_ou_segment_thresholds.py failed: $($_)" }

  # Generate OU picks for today using locked policy (tau/sigma_max/pmin)
  Write-Section '6j.b) Generate OU picks (policy)'
  try {
    & $VenvPython scripts/make_picks_raw_from_totals.py --date $todayIso --outputs $OutDir
  } catch { Write-Warning "OU picks generation failed: $($_)" }

  # Daily quantile retrain + scoring artifacts (lightweight; safe to run post-predictions)
  Write-Section '6j.ii) Daily quantile retrain + scoring'
  try {
    $dq = Join-Path $RepoRoot 'scripts\daily_quantile_scoring.ps1'
    if (Test-Path $dq) {
      & $dq
    } else {
      Write-Warning "daily_quantile_scoring.ps1 not found: $dq"
    }
  } catch { Write-Warning "daily quantile scoring failed: $($_)" }

  # Quantile CRPS + coverage metrics and short-term/weekly trends
  Write-Section '6j.iii) Quantile CRPS metrics + trends'
  try { & $VenvPython scripts/evaluate_crps.py } catch { Write-Warning "evaluate_crps.py failed: $($_)" }
  try { & $VenvPython scripts/score_quantile_trend.py } catch { Write-Warning "score_quantile_trend.py failed: $($_)" }

  # Weekly drift summary (probability + quantile coverage/CRPS aggregates)
  Write-Section '6j.iv) Weekly drift summary'
  try { & $VenvPython scripts/drift_monitor.py } catch { Write-Warning "drift_monitor.py failed: $($_)" }

  # Synthetic pipeline: calibrate + quantiles preference + stake sizing
  Write-Section '6k) Synthetic pipeline (calibrate + intervals + stake sizing)'
  try {
    & $VenvPython scripts/run_synthetic_pipeline_end_to_end.py
  } catch { Write-Warning "run_synthetic_pipeline_end_to_end.py failed: $($_)" }

  # Post-inference variance summary (inference-level dispersion)
  try {
    $predCsv = Get-Content $modelPredPath | Select-Object -Skip 1
    $totals = @(); $margins = @()
    foreach ($line in $predCsv) {
      if (-not $line) { continue }
      $parts = $line.Split(',')
      if ($parts.Length -ge 3) {
        [double]$t = $parts[1]; [double]$m = $parts[2];
        if ($t -ne [double]::NaN) { $totals += $t }
        if ($m -ne [double]::NaN) { $margins += $m }
      }
    }
    if ($totals.Count -gt 0) {
      $totMean = ($totals | Measure-Object -Average).Average
      if ($totals.Count -gt 1) {
        $totVarSum = ($totals | ForEach-Object { $d = ($_ - $totMean); $d * $d } | Measure-Object -Sum).Sum
        $totVar = [Math]::Round(($totVarSum / $totals.Count), 4)
      } else { $totVar = 0 }
      $totStd = [Math]::Round([Math]::Sqrt($totVar), 4)
      $totMin = ($totals | Measure-Object -Minimum).Minimum
      $totMax = ($totals | Measure-Object -Maximum).Maximum
      $margMean = ($margins | Measure-Object -Average).Average
      if ($margins.Count -gt 1) {
        $margVarSum = ($margins | ForEach-Object { $d = ($_ - $margMean); $d * $d } | Measure-Object -Sum).Sum
        $margVar = [Math]::Round(($margVarSum / $margins.Count), 4)
      } else { $margVar = 0 }
      $margStd = [Math]::Round([Math]::Sqrt($margVar), 4)
      $margMin = ($margins | Measure-Object -Minimum).Minimum
      $margMax = ($margins | Measure-Object -Maximum).Maximum
      $infVarDir = Join-Path $OutDir 'variance'
      New-Item -ItemType Directory -Path $infVarDir -Force | Out-Null
      $infVarPath = Join-Path $infVarDir ("inference_variance_" + $todayIso + ".json")
      $payload = @{date=$todayIso; rows=$totals.Count; total_mean=[Math]::Round($totMean,2); total_var=$totVar; total_std=$totStd; total_min=$totMin; total_max=$totMax; margin_mean=[Math]::Round($margMean,2); margin_var=$margVar; margin_std=$margStd; margin_min=$margMin; margin_max=$margMax; timestamp_utc=(Get-Date).ToUniversalTime().ToString('o') }
      ($payload | ConvertTo-Json -Depth 4) | Out-File -FilePath $infVarPath -Encoding UTF8
      Write-Host "Inference variance summary -> $infVarPath"
    }
  } catch { Write-Warning "inference variance summary failed: $($_)" }

  Write-Section "6) Fetch today's schedule, odds, and run predictions/picks"
  $dailyArgs = @('daily-run', '--date', $todayIso, '--season', $todayDate.Year, '--region', $Region, '--provider', $Provider, '--segment', 'team', '--preseason-weight', '0.4', '--threshold', '1.5', '--default-price', '-110')
  if ($NoCache.IsPresent) { $dailyArgs += '--no-use-cache' }
  & $VenvPython -m ncaab_model.cli @dailyArgs

  Write-Section '6a.pre) Ensure full game & prediction coverage (force-fill)'
  try {
    & $VenvPython scripts/ensure_full_game_prediction_coverage.py $todayIso
  } catch { Write-Warning "force-fill coverage script failed: $($_)" }
  try {
    & $VenvPython scripts/promote_force_fill_today.py $todayIso
  } catch { Write-Warning "promotion of force-filled enriched artifact failed: $($_)" }

  # Persist normalized start fields in enriched predictions and assert no NaN _start_dt remain
  Write-Section '6a.post) Normalize start fields and persist _start_dt'
  try {
    $normOutput = (& $VenvPython scripts/normalize_start_fields.py $todayIso --inplace) | Out-String
    Write-Host $normOutput.Trim()
  } catch {
    Write-Warning "normalize_start_fields.py failed: $($_)"
  }
  # Ensure predictions_display_<date>.csv exists for snapshot-first UI parity
  Write-Section '6a.post.i) Persist display snapshot for today if missing'
  try {
    $displaySnap = Join-Path $OutDir ("predictions_display_" + $todayIso + ".csv")
    $enrichedPred = Join-Path $OutDir ("predictions_unified_enriched_" + $todayIso + ".csv")
    if (-not (Test-Path $displaySnap)) {
      if (Test-Path $enrichedPred) {
        $pyPersist = @"
import pandas as pd
from pathlib import Path
out_dir = Path(r'${OutDir}')
date = '${todayIso}'
enr = out_dir / f'predictions_unified_enriched_{date}.csv'
dis = out_dir / f'predictions_display_{date}.csv'
try:
    df = pd.read_csv(enr)
except Exception as e:
    print(f'[persist-display] read enriched failed: {e}'); raise SystemExit(1)
# Keep core columns plus display fields if present
keep = [
    'game_id','date','home_team','away_team',
    'pred_total','pred_margin','pred_total_basis','pred_margin_basis',
    'market_total','spread_home',
    'start_time_iso','_start_dt','start_time','commence_time',
    'display_date','start_time_display','display_time_str','start_tz_abbr'
]
cols = [c for c in keep if c in df.columns]
df = df[cols].copy()
# Basic sanitization
df['game_id'] = df['game_id'].astype(str).str.replace(r'\\.0$','', regex=True) if 'game_id' in df.columns else df.get('game_id')
df['date'] = df['date'].astype(str) if 'date' in df.columns else df.get('date')
if 'date' in df.columns:
    df = df[df['date'].astype(str) == date]
df.to_csv(dis, index=False)
print({'path': str(dis), 'rows': len(df)})
"@
        & $VenvPython -c $pyPersist
      } else {
        Write-Warning "Enriched predictions not found at $enrichedPred; cannot persist display snapshot."
      }
    } else {
      Write-Host "Display snapshot already present -> $displaySnap" -ForegroundColor DarkGray
    }
  } catch { Write-Warning "Persisting display snapshot failed: $($_)" }
  try {
    $pyCheck = @"
import pandas as pd, sys
from pathlib import Path
date = '$todayIso'
path = Path(r'$OutDir')/f'predictions_unified_enriched_{date}.csv'
try:
    df = pd.read_csv(path)
except Exception as e:
    print(f'[check] unable to read {path}: {e}')
    sys.exit(2)
mask = df['date'].astype(str).eq(date) if 'date' in df.columns else pd.Series([True]*len(df))
sd = pd.to_datetime(df.loc[mask, '_start_dt'], errors='coerce', utc=True) if '_start_dt' in df.columns else pd.Series([], dtype='datetime64[ns, UTC]')
nan_count = int(sd.isna().sum()) if len(sd) else 0
print(f'[check] normalized rows={int(mask.sum())} nan__start_dt={nan_count}')
sys.exit(1 if nan_count>0 else 0)
"@
    & $VenvPython -c $pyCheck
    if ($LASTEXITCODE -eq 1) {
      Add-CriticalFailure "Normalization left NaN _start_dt rows in predictions_unified_enriched_${todayIso}.csv"
    }
  } catch {
    Write-Warning "post-normalization check failed: $($_)"
  }

    # Safeguard: drop unposted/no-market games not present in provider slate
    Write-Section '6a.post.ii) Safeguard: remove No Market (Unposted) games'
    try {
      & $VenvPython scripts/remove_unposted_no_market.py $todayIso $OutDir
    } catch { Write-Warning "safeguard removal failed: $($_)" }

    # Model-based totals quantiles (q10/q50/q90) -> merge into unified_enriched
    # This is the preferred source for sim quantile targeting vs sigma-derived normal fallback.
    Write-Section '6a.post.ii.q) Score totals quantiles + integrate'
    try {
      $TotalsQuantileModel = $env:NCAAB_TOTALS_QUANTILE_MODEL

      # If user provided a model path but it doesn't exist, warn and fall back.
      if ($TotalsQuantileModel -and $TotalsQuantileModel.Trim() -ne '') {
        if (-not (Test-Path $TotalsQuantileModel)) {
          Write-Warning "NCAAB_TOTALS_QUANTILE_MODEL path not found: $TotalsQuantileModel (falling back to defaults)"
          $TotalsQuantileModel = $null
        }
      }

      # Default to roll-feature totals quantile model when available.
      if (-not $TotalsQuantileModel -or $TotalsQuantileModel.Trim() -eq '') {
        $rollModelPath = Join-Path $OutDir 'models\totals_roll_v1.joblib'
        if (Test-Path $rollModelPath) {
          $TotalsQuantileModel = $rollModelPath
          $env:NCAAB_TOTALS_QUANTILE_MODEL = $TotalsQuantileModel
          Write-Host "NCAAB_TOTALS_QUANTILE_MODEL not set; defaulting to $TotalsQuantileModel" -ForegroundColor DarkGray
        }
      }

      if ($TotalsQuantileModel -and $TotalsQuantileModel.Trim() -ne '') {
        Write-Host "[totals-quantiles] Using model: $TotalsQuantileModel" -ForegroundColor DarkGray
        & $VenvPython -m src.score_totals --date $todayIso --model $TotalsQuantileModel
      } else {
        & $VenvPython -m src.score_totals --date $todayIso
      }
    } catch { Write-Warning "src.score_totals failed: $($_)" }
    try {
      & $VenvPython -m src.integrate_model_totals --date $todayIso
    } catch { Write-Warning "src.integrate_model_totals failed: $($_)" }

    # Model-based margin quantiles (q10/q50/q90) -> merge into unified_enriched
    # Enables sim quantile targeting for margins when explicit coverage is high.
    Write-Section '6a.post.ii.r) Score margins quantiles + integrate'
    $didIntegrateMarginsQuantiles = $false
    try {
      $MarginsQuantileModel = $env:NCAAB_MARGINS_QUANTILE_MODEL

      # If user provided a model path but it doesn't exist, warn and fall back.
      if ($MarginsQuantileModel -and $MarginsQuantileModel.Trim() -ne '') {
        if (-not (Test-Path $MarginsQuantileModel)) {
          Write-Warning "NCAAB_MARGINS_QUANTILE_MODEL path not found: $MarginsQuantileModel (falling back to defaults)"
          $MarginsQuantileModel = $null
        }
      }

      # Default to roll-feature margins quantile model when available.
      if (-not $MarginsQuantileModel -or $MarginsQuantileModel.Trim() -eq '') {
        $rollModelPath = Join-Path $OutDir 'models\margins_roll_v1.joblib'
        if (Test-Path $rollModelPath) {
          $MarginsQuantileModel = $rollModelPath
          $env:NCAAB_MARGINS_QUANTILE_MODEL = $MarginsQuantileModel
          Write-Host "NCAAB_MARGINS_QUANTILE_MODEL not set; defaulting to $MarginsQuantileModel" -ForegroundColor DarkGray
        }
      }

      if ($MarginsQuantileModel -and $MarginsQuantileModel.Trim() -ne '') {
        Write-Host "[margins-quantiles] Using model: $MarginsQuantileModel" -ForegroundColor DarkGray
        & $VenvPython -m src.score_margins --date $todayIso --model $MarginsQuantileModel
        & $VenvPython -m src.integrate_model_margins --date $todayIso
        try {
          $margOut = Join-Path $OutDir ("predictions_model_margins_" + $todayIso + ".csv")
          if (Test-Path -LiteralPath $margOut) {
            $probe = Import-Csv -LiteralPath $margOut -ErrorAction Stop | Select-Object -First 1
            if ($probe -and ($probe.PSObject.Properties.Name -contains 'pred_margin_q10') -and ($probe.PSObject.Properties.Name -contains 'pred_margin_q50') -and ($probe.PSObject.Properties.Name -contains 'pred_margin_q90')) {
              $didIntegrateMarginsQuantiles = $true
            }
          }
        } catch {}
      } else {
        Write-Host "[margins-quantiles] No margin quantile model found; skipping." -ForegroundColor DarkGray
      }
    } catch { Write-Warning "margin quantile scoring/integration failed: $($_)" }

  # Enrich meta probabilities in-place using aligned features; guard against model/schema gaps
  Write-Section '6a.post.b) Enrich meta probabilities (aligned)'
  try {
    & $VenvPython scripts/enrich_meta_probs.py $todayIso --inplace
  } catch { Write-Warning "enrich_meta_probs.py failed: $($_)" }

  # (moved) Feature parity checks run after sigma+blend to avoid false missing columns

  # Inject sigma fields and adjusted Kelly after enrichment to ensure availability downstream
  Write-Section '6a.post.c) Inject sigma and adjusted Kelly'
  try {
    & $VenvPython scripts/inject_sigma_and_kelly.py --date $todayIso
  } catch { Write-Warning "inject_sigma_and_kelly.py failed: $($_)" }

  # Market-aware posterior blend + mismatch guardrails
  Write-Section '6a.post.d) Market-aware blend + guardrails'
  try {
    & $VenvPython scripts/apply_market_blend.py --date $todayIso --w-market-total 0.8 --w-market-margin 0.7 --thr-total 20 --thr-margin 8
  } catch { Write-Warning "apply_market_blend.py failed: $($_)" }

  # Monte Carlo simulations for OU/margins and blend with model quantiles
  # NOTE: this must run after daily-run + force-fill promotion + sigma/blend so it sees populated enriched preds.
  if (-not $env:NCAAB_SIM_SEED -or $env:NCAAB_SIM_SEED.Trim() -eq '') {
    $env:NCAAB_SIM_SEED = $todayIso.Replace('-','')
    Write-Host "NCAAB_SIM_SEED not set; defaulting to $($env:NCAAB_SIM_SEED)" -ForegroundColor DarkGray
  } else {
    Write-Host "Using NCAAB_SIM_SEED=$($env:NCAAB_SIM_SEED)" -ForegroundColor DarkGray
  }
  if (-not $env:NCAAB_SIM_BLEND_EVENT_PACE -or $env:NCAAB_SIM_BLEND_EVENT_PACE.Trim() -eq '') {
    $env:NCAAB_SIM_BLEND_EVENT_PACE = '1'
    Write-Host "NCAAB_SIM_BLEND_EVENT_PACE not set; defaulting to $($env:NCAAB_SIM_BLEND_EVENT_PACE)" -ForegroundColor DarkGray
  } else {
    Write-Host "Using NCAAB_SIM_BLEND_EVENT_PACE=$($env:NCAAB_SIM_BLEND_EVENT_PACE)" -ForegroundColor DarkGray
  }

  # If quantile targeting is enabled, default the knobs to the sim-backtest winner
  # (totals-only + partial strength) unless explicitly overridden in the shell.
  $tq = $env:NCAAB_SIM_TARGET_QUANTILES
  $tqOn = $false
  if ($tq -and $tq.Trim() -ne '' -and $tq.Trim() -ne '0') {
    $tql = $tq.Trim().ToLower()
    if ($tql -ne 'false' -and $tql -ne 'off' -and $tql -ne 'no') {
      $tqOn = $true
    }
  }
  if ($tqOn) {
    if (-not $env:NCAAB_SIM_TARGET_QUANTILES_SOURCE -or $env:NCAAB_SIM_TARGET_QUANTILES_SOURCE.Trim() -eq '') {
      $env:NCAAB_SIM_TARGET_QUANTILES_SOURCE = 'auto'
      Write-Host "NCAAB_SIM_TARGET_QUANTILES_SOURCE not set; defaulting to $($env:NCAAB_SIM_TARGET_QUANTILES_SOURCE)" -ForegroundColor DarkGray
    } else {
      Write-Host "Using NCAAB_SIM_TARGET_QUANTILES_SOURCE=$($env:NCAAB_SIM_TARGET_QUANTILES_SOURCE)" -ForegroundColor DarkGray
    }
    if (-not $env:NCAAB_SIM_TARGET_QUANTILES_TOTAL -or $env:NCAAB_SIM_TARGET_QUANTILES_TOTAL.Trim() -eq '') {
      $env:NCAAB_SIM_TARGET_QUANTILES_TOTAL = '1'
      Write-Host "NCAAB_SIM_TARGET_QUANTILES_TOTAL not set; defaulting to $($env:NCAAB_SIM_TARGET_QUANTILES_TOTAL)" -ForegroundColor DarkGray
    } else {
      Write-Host "Using NCAAB_SIM_TARGET_QUANTILES_TOTAL=$($env:NCAAB_SIM_TARGET_QUANTILES_TOTAL)" -ForegroundColor DarkGray
    }
    if (-not $env:NCAAB_SIM_TARGET_QUANTILES_MARGIN -or $env:NCAAB_SIM_TARGET_QUANTILES_MARGIN.Trim() -eq '') {
      $env:NCAAB_SIM_TARGET_QUANTILES_MARGIN = (if ($didIntegrateMarginsQuantiles) { '1' } else { '0' })
      Write-Host "NCAAB_SIM_TARGET_QUANTILES_MARGIN not set; defaulting to $($env:NCAAB_SIM_TARGET_QUANTILES_MARGIN)" -ForegroundColor DarkGray
    } else {
      Write-Host "Using NCAAB_SIM_TARGET_QUANTILES_MARGIN=$($env:NCAAB_SIM_TARGET_QUANTILES_MARGIN)" -ForegroundColor DarkGray
    }
    if (-not $env:NCAAB_SIM_TARGET_QUANTILES_ALPHA -or $env:NCAAB_SIM_TARGET_QUANTILES_ALPHA.Trim() -eq '') {
      $env:NCAAB_SIM_TARGET_QUANTILES_ALPHA = '0.25'
      Write-Host "NCAAB_SIM_TARGET_QUANTILES_ALPHA not set; defaulting to $($env:NCAAB_SIM_TARGET_QUANTILES_ALPHA)" -ForegroundColor DarkGray
    } else {
      Write-Host "Using NCAAB_SIM_TARGET_QUANTILES_ALPHA=$($env:NCAAB_SIM_TARGET_QUANTILES_ALPHA)" -ForegroundColor DarkGray
    }
  }
  Write-Section '6a.post.d.s) Monte Carlo simulations + blend'
  # Default sim means to the model/blend columns (auto). This keeps sim margins aligned
  # with our betting-intent predictions and avoids unrealistically "too many ties" when
  # feature-only margins compress toward 0.
  #
  # To override (e.g., feature-only experiments), set NCAAB_SIM_MEAN_SOURCE in the shell
  # before running this script.
  if (-not $env:NCAAB_SIM_MEAN_SOURCE -or $env:NCAAB_SIM_MEAN_SOURCE.Trim() -eq '') {
    $env:NCAAB_SIM_MEAN_SOURCE = 'auto'
  }
  try {
    & $VenvPython scripts/validate_sim_inputs.py $todayIso $OutDir
  } catch {
    Write-Warning "validate_sim_inputs.py failed (continuing): $($_)"
  }
  try {
    & $VenvPython scripts/run_game_simulations.py $todayIso $OutDir
  } catch { Write-Warning "run_game_simulations.py failed: $($_)" }

  # Generate 2-min segments side-by-side for the cards UI (outputs/sim_segments_2min_<date>.csv)
  Write-Section '6a.post.d.s.i) Generate 2-min sim segments (cards/UI)'
  try {
    & $VenvPython scripts/run_game_simulations.py $todayIso $OutDir --segments-grid-min 2 --segments-out-prefix sim_segments_2min_ --quantiles-out-prefix sim_quantiles_2min_ --meta-out-prefix sim_meta_2min_
  } catch { Write-Warning "2-min run_game_simulations.py failed: $($_)" }

  # Safety: ensure 2-min artifacts exist (Live Lens prefers 2-min when present).
  # If missing, retry once to avoid silent fallback to 5-min.
  try {
    $seg2 = Join-Path $OutDir ("sim_segments_2min_" + $todayIso + ".csv")
    $q2 = Join-Path $OutDir ("sim_quantiles_2min_" + $todayIso + ".csv")
    $m2 = Join-Path $OutDir ("sim_meta_2min_" + $todayIso + ".json")
    if ((-not (Test-Path $seg2)) -or (-not (Test-Path $q2)) -or (-not (Test-Path $m2))) {
      Write-Warning "[2-min] Missing after sim run; retrying once: seg=$([bool](Test-Path $seg2)) q=$([bool](Test-Path $q2)) meta=$([bool](Test-Path $m2))"
      try {
        & $VenvPython scripts/run_game_simulations.py $todayIso $OutDir --segments-grid-min 2 --segments-out-prefix sim_segments_2min_ --quantiles-out-prefix sim_quantiles_2min_ --meta-out-prefix sim_meta_2min_
      } catch {
        Write-Warning "[2-min] Retry failed: $($_)"
      }
    }
  } catch {
    Write-Warning "[2-min] Artifact existence check failed: $($_)"
  }
  try {
    $BlendSimWeight = if ($env:BLEND_SIM_WEIGHT) { [double]$env:BLEND_SIM_WEIGHT } else { 0.2 }
    Write-Host "Blending simulations with weight $BlendSimWeight"
    & $VenvPython scripts/blend_sim_quantiles.py $todayIso $OutDir $BlendSimWeight
  } catch { Write-Warning "blend_sim_quantiles.py failed: $($_)" }

  # Feature parity check against trained meta probability schemas (post sigma+blend)
  Write-Section '6a.post.e) Feature parity checks (meta probs)'
  try {
    $parityOut = (& $VenvPython scripts/check_feature_parity.py --date $todayIso) | Out-String
    $metricsDir = Join-Path $OutDir 'metrics'
    New-Item -ItemType Directory -Path $metricsDir -Force | Out-Null
    $parityPath = Join-Path $metricsDir ("feature_parity_" + $todayIso + ".txt")
    $parityOut.Trim() | Out-File -FilePath $parityPath -Encoding UTF8
    Write-Host "[metrics] Wrote $parityPath"
  } catch { Write-Warning "feature parity check failed: $($_)" }

  # Tune OU/ATS thresholds using recent history
  Write-Section '6a.post.f) Tune OU/ATS thresholds vs odds'
  try {
    & $VenvPython scripts/tune_thresholds_vs_odds.py --window-days 60
  } catch { Write-Warning "tune_thresholds_vs_odds.py failed: $($_)" }

  # Tune OU selection policy to achieve >=75% accuracy with gating
  Write-Section '6a.post.f.i) Tune OU selection policy (>=75% acc)'
  try {
    & $VenvPython scripts/tune_ou_selection.py --window-days 60 --target-accuracy 0.75 --min-coverage 20 --use-closing
  } catch { Write-Warning "tune_ou_selection.py failed: $($_)" }

  # Evaluate selected OU policy and persist metrics
  Write-Section '6a.post.f.ii) Evaluate OU policy (coverage + accuracy)'
  try {
    & $VenvPython scripts/evaluate_ou_policy.py --window-days 90 --use-closing
  } catch { Write-Warning "evaluate_ou_policy.py failed: $($_)" }

  # Tune ATS selection policy to achieve >=75% accuracy with gating
  Write-Section '6a.post.f.iii) Tune ATS selection policy (>=75% acc)'
  try {
    & $VenvPython scripts/tune_ats_selection.py --window-days 60 --target-accuracy 0.75 --min-coverage 20 --use-closing
  } catch { Write-Warning "tune_ats_selection.py failed: $($_)" }

  # Evaluate selected ATS policy and persist metrics
  Write-Section '6a.post.f.iv) Evaluate ATS policy (coverage + accuracy)'
  try {
    & $VenvPython scripts/evaluate_ats_policy.py --window-days 90 --use-closing
  } catch { Write-Warning "evaluate_ats_policy.py failed: $($_)" }

  # Overall accuracy snapshot (all games) to track baseline improvements
  Write-Section '6a.post.g) Evaluate overall OU/ATS accuracy (all games)'
  try {
    $evalAll = (& $VenvPython scripts/evaluate_vs_odds.py --use-closing) | Out-String
    $metricsDir = Join-Path $OutDir 'metrics'
    New-Item -ItemType Directory -Path $metricsDir -Force | Out-Null
    $overallPath = Join-Path $metricsDir 'accuracy_vs_odds_overall.json'
    $evalAll.Trim() | Out-File -FilePath $overallPath -Encoding UTF8
    Write-Host "[metrics] Wrote $overallPath"
  } catch { Write-Warning "evaluate_vs_odds (overall) failed: $($_)" }

  # Now regenerate team-level historical features with any newly completed games merged by daily-run
  Write-Section '6b) Refresh team-level historical features post-ingestion'
  try {
    & $VenvPython -m src.modeling.team_features --out (Join-Path $OutDir 'team_features.csv')
  } catch { Write-Warning "team_features post-ingestion generation failed: $($_)" }

  if (-not $SkipModelTests) {
    Write-Section '6b.i) Run model integrity tests'
    try {
      & $VenvPython -m pytest tests/test_team_feature_artifact.py tests/test_training_frame_richness.py -q
    } catch {
      Add-CriticalFailure "Model integrity tests failed: $($_)"
    }
  } else { Write-Host 'SkipModelTests flag set; skipping pytest integrity checks.' -ForegroundColor Yellow }
  $modelDir = Join-Path $OutDir 'models'
  $latestModelTotal = Get-ChildItem -Path $modelDir -Filter 'total_model.pkl' -Recurse -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $latestModelMargin = Get-ChildItem -Path $modelDir -Filter 'margin_model.pkl' -Recurse -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $teamFeatPath = Join-Path $OutDir 'team_features.csv'
  $teamFeatStamp = if (Test-Path $teamFeatPath) { (Get-Item $teamFeatPath).LastWriteTimeUtc } else { Get-Date }
  $modelTotalStamp = if ($latestModelTotal) { $latestModelTotal.LastWriteTimeUtc } else { (Get-Date).AddYears(-10) }
  $modelMarginStamp = if ($latestModelMargin) { $latestModelMargin.LastWriteTimeUtc } else { (Get-Date).AddYears(-10) }
  $needModelRefresh = (-not $latestModelTotal -or -not $latestModelMargin) -or ($teamFeatStamp -gt $modelTotalStamp) -or ($teamFeatStamp -gt $modelMarginStamp) -or $ForceModelRetrain.IsPresent
  if (-not $SkipRetrain -and $needModelRefresh) {
    Write-Section '6c) Train/refresh model-first total & margin predictors (LightGBM/XGBoost)'
    try {
      & $VenvPython -m src.modeling.train_total --algo auto --split random
    } catch { Write-Warning "train_total (post-ingestion) failed: $($_)" }
    try {
      & $VenvPython -m src.modeling.train_margin --algo auto --split random
    } catch { Write-Warning "train_margin (post-ingestion) failed: $($_)" }
  } else {
    Write-Host 'Model-first artifacts considered fresh post-ingestion; skipping retrain.' -ForegroundColor DarkGray
  }

  if (-not $SkipVarianceDiag) {
    Write-Section '6d) Prediction variance diagnostics'
    try {
      $varTotal = (& $VenvPython -m src.modeling.diagnose_variance --target total --algo auto --split random) | Out-String
      $varMargin = (& $VenvPython -m src.modeling.diagnose_variance --target margin --algo auto --split random) | Out-String
      $varDir = Join-Path $OutDir 'variance'
      New-Item -ItemType Directory -Path $varDir -Force | Out-Null
      $varTotalPath = Join-Path $varDir ("variance_total_" + $todayIso + ".json")
      $varMarginPath = Join-Path $varDir ("variance_margin_" + $todayIso + ".json")
      $varTotal | Out-File -FilePath $varTotalPath -Encoding UTF8
      $varMargin | Out-File -FilePath $varMarginPath -Encoding UTF8
      Write-Host "Wrote variance diagnostics -> $varTotalPath, $varMarginPath"
    } catch { Write-Warning "Variance diagnostics failed: $($_)" }
  } else { Write-Host 'SkipVarianceDiag flag set; skipping prediction variance diagnostics.' -ForegroundColor Yellow }

  # Contingency: if earlier model inference/calibration/intervals (5d-5f) failed due to missing features,
  # re-run them now that today's schedule/odds/features are present.
  try {
    $predModelToday = Join-Path $OutDir ("predictions_model_" + $todayIso + ".csv")
    $predModelCalToday = Join-Path $OutDir ("predictions_model_calibrated_" + $todayIso + ".csv")
    $predModelIntToday = Join-Path $OutDir ("predictions_model_interval_" + $todayIso + ".csv")
    if (-not (Test-Path $predModelToday)) {
      Write-Section '6d.i) Re-run model inference harness (contingency)'
      try { & $VenvPython -m src.modeling.infer --date $todayIso } catch { Write-Warning "contingency infer failed: $($_)" }
    }
    if ((Test-Path $predModelToday) -and (-not (Test-Path $predModelCalToday))) {
      Write-Section '6d.ii) Re-run calibration (contingency)'
      try { & $VenvPython -m src.modeling.calibrate_predictions --date $todayIso --predictions-file $predModelToday --results-dir (Join-Path $OutDir 'daily_results') --window-days 14 } catch { Write-Warning "contingency calibration failed: $($_)" }
    }
    if ((Test-Path $predModelToday) -and (-not (Test-Path $predModelIntToday))) {
      Write-Section '6d.iii) Re-run interval generation (contingency)'
      try {
        if (Test-Path $predModelCalToday) {
          & $VenvPython -m src.modeling.interval_predictions --date $todayIso --predictions-file $predModelToday --calibrated-file $predModelCalToday --results-dir (Join-Path $OutDir 'daily_results') --window-days 30
        } else {
          & $VenvPython -m src.modeling.interval_predictions --date $todayIso --predictions-file $predModelToday --results-dir (Join-Path $OutDir 'daily_results') --window-days 30
        }
      } catch { Write-Warning "contingency interval generation failed: $($_)" }
    }
  } catch { Write-Warning "contingency block error: $($_)" }

  # Meta stacking, stability, and auto calibration steps
  Write-Section '6e) Train meta probability models (cover/over)'
  try {
    & $VenvPython scripts/train_meta_probs.py --limit-days 45
    & $VenvPython scripts/train_meta_probs_lgbm.py --limit-days 45
    # Emit sidecar schemas aligned to trained LGBM models for app-time alignment
    & $VenvPython scripts/emit_meta_sidecars.py
  } catch { Write-Warning "train_meta_probs failed: $($_)" }

  Write-Section '6f) Probability distribution stability (JS divergence)'
  try {
    & $VenvPython scripts/probability_stability.py
  } catch { Write-Warning "probability_stability failed: $($_)" }

  if (-not $SkipProbCalibrationFit) {
    Write-Section "6g) Auto-refresh probability calibration (as-of=$prevDate; ECE/drift/age)"
    try {
      & $VenvPython scripts/auto_refresh_calibration.py --date $prevDate
    } catch { Write-Warning "auto_refresh_calibration failed: $($_)" }
  } else {
    Write-Host '[skip] Auto-refresh probability calibration' -ForegroundColor Yellow
  }

  Write-Section '6g.i) Meta probability reliability + calibration'
  try {
    & $VenvPython scripts/compute_meta_reliability.py --limit-days 45
  } catch { Write-Warning "compute_meta_reliability failed: $($_)" }
  try {
    & $VenvPython scripts/auto_calibrate_meta.py
  } catch { Write-Warning "auto_calibrate_meta failed: $($_)" }

  Write-Section '6h) Explain meta models (feature contributions)'
  try {
    & $VenvPython scripts/explain_meta.py --date $todayIso
  } catch { Write-Warning "explain_meta failed: $($_)" }

  # Guard: daily-run may overwrite the historical games_with_last.csv with a subset (today's slate).
  # Reconstruct full historical last odds merge to ensure persistence for downstream joins.
  Write-Section '6a) Restore full games_with_last.csv (historical) after daily-run'
  try {
    $gamesAll = Join-Path $OutDir 'games_all.csv'
    $lastOdds = Join-Path $OutDir 'last_odds.csv'
    if ((Test-Path $gamesAll) -and (Test-Path $lastOdds)) {
      & $VenvPython -m ncaab_model.cli join-last-odds $gamesAll $lastOdds --out (Join-Path $OutDir 'games_with_last.csv')
    } else {
      Write-Warning 'Cannot restore games_with_last.csv (missing games_all.csv or last_odds.csv)'
    }
  } catch { Write-Warning "Restore games_with_last.csv failed: $($_)" }

  # Stake sheets are deprecated and removed from the app; still produce today's
  # merged/edges/display artifacts for recommendations.
  Write-Section "7) Filter merged last odds to today's slate (with closing fallback)"
  $mergedAll = Join-Path $OutDir 'games_with_last.csv'
  $mergedToday = Join-Path $OutDir 'games_with_last_today.csv'
  $gamesCurrPath = Join-Path $OutDir "games_${todayIso}.csv"
  if (Test-Path $mergedAll) {
  $pyFilter = @"
import pandas as pd, sys
inp = r'$mergedAll'
outp = r'$mergedToday'
target = '$todayIso'
games_curr = r'$gamesCurrPath'
try:
    df = pd.read_csv(inp)
except Exception as e:
    print(f'[read-fail] merged: {e}')
    sys.exit(1)
gid_today = set()
try:
    gc = pd.read_csv(games_curr)
    if 'game_id' in gc.columns:
      gc['game_id'] = gc['game_id'].astype(str).str.replace(r'\\.0$','', regex=True)
    if 'date' in gc.columns:
      gc['date'] = gc['date'].astype(str)
      gc = gc[gc['date'] == target]
    if 'game_id' in gc.columns:
      gid_today = set(gc['game_id'].astype(str))
except Exception as e:
    print(f'[warn] games_curr read failed: {e}')
if 'game_id' in df.columns:
  df['game_id'] = df['game_id'].astype(str).str.replace(r'\\.0$','', regex=True)
# Prefer strict inner-join on today's game_id set to avoid oversized selections
if gid_today:
  gc_ids = pd.DataFrame({'game_id': list(gid_today)})
  df_today = df.merge(gc_ids, on='game_id', how='inner')
  # Ensure one row per game for downstream alignment
  if 'game_id' in df_today.columns:
    df_today = df_today.drop_duplicates(subset=['game_id'])
else:
  df_today = df.iloc[0:0].copy()
if df_today.empty and ('date' in df.columns):
  # Secondary filter by date only if gid-based yielded nothing
  df['date'] = df['date'].astype(str)
  df_today = df[df['date'] == target].copy()
# Backfill schedule start times (and related display fields) from today's games file.
try:
  if (not df_today.empty) and ('game_id' in df_today.columns):
    # Drop any stray rows that don't correspond to a scheduled game
    df_today['game_id'] = df_today['game_id'].astype(str).str.replace(r'\\.0$','', regex=True)
    df_today = df_today[df_today['game_id'].notna() & (df_today['game_id'].astype(str) != '')].copy()
    try:
      gc = pd.read_csv(games_curr)
      if 'game_id' in gc.columns:
        gc['game_id'] = gc['game_id'].astype(str).str.replace(r'\\.0$','', regex=True)
      keep_cols = [c for c in ['game_id','start_time','start_time_iso','start_time_local','start_tz_abbr'] if c in gc.columns]
      if keep_cols:
        gc_small = gc[keep_cols].drop_duplicates(subset=['game_id'])
        df_today = df_today.merge(gc_small, on='game_id', how='left', suffixes=('','_sched'))
        for c in ['start_time','start_time_iso','start_time_local','start_tz_abbr']:
          sc = c + '_sched'
          if (c in df_today.columns) and (sc in df_today.columns):
            df_today[c] = df_today[c].where(df_today[c].notna() & (df_today[c].astype(str) != ''), df_today[sc])
            df_today.drop(columns=[sc], inplace=True, errors='ignore')
          elif (c not in df_today.columns) and (sc in df_today.columns):
            df_today[c] = df_today[sc]
            df_today.drop(columns=[sc], inplace=True, errors='ignore')
    except Exception as e:
      print('[warn] schedule backfill failed: ' + str(e))
except Exception:
  pass
# Debug: show unique dates and gid_today size, avoid f-strings to prevent parser issues
try:
  udates = sorted(set(df['date'].astype(str)))[:5] if 'date' in df.columns else []
  print('[debug] unique_dates_in_merged=' + str(udates))
except Exception:
  pass
print('[debug] gid_today_count=' + str(len(gid_today)))
df_today.to_csv(outp, index=False)
print(f'Filtered games_with_last.csv -> {len(df)} total, {len(df_today)} rows for {target}')
"@
    & $VenvPython -c $pyFilter
    # If last odds yielded 0 rows, fallback to closing lines
    try {
      $rows = @(Import-Csv -LiteralPath $mergedToday)
      if (-not $rows -or $rows.Count -le 0) {
        Write-Host "[fallback] No last-odds rows for $todayIso; using closing lines." -ForegroundColor Yellow
        $mergedClosingAll = Join-Path $OutDir 'games_with_closing.csv'
        $mergedClosingToday = Join-Path $OutDir 'games_with_closing_today.csv'
        if (Test-Path $mergedClosingAll) {
          $pyFilterClosing = @"
import pandas as pd, sys
inp = r'$mergedClosingAll'
outp = r'$mergedClosingToday'
target = '$todayIso'
games_curr = r'$gamesCurrPath'
try:
    df = pd.read_csv(inp)
except Exception as e:
    print(f'[read-fail] merged(closing): {e}')
    sys.exit(1)
gid_today = set()
try:
    gc = pd.read_csv(games_curr)
    if 'game_id' in gc.columns:
        gc['game_id'] = gc['game_id'].astype(str).str.replace(r'\\.0$','', regex=True)
    if 'date' in gc.columns:
        gc['date'] = gc['date'].astype(str)
        gc = gc[gc['date'] == target]
    if 'game_id' in gc.columns:
        gid_today = set(gc['game_id'].astype(str))
except Exception as e:
    print(f'[warn] games_curr read failed: {e}')
if 'game_id' in df.columns:
  df['game_id'] = df['game_id'].astype(str).str.replace(r'\\.0$','', regex=True)
if gid_today:
  gc_ids = pd.DataFrame({'game_id': list(gid_today)})
  df_today = df.merge(gc_ids, on='game_id', how='inner')
  if 'game_id' in df_today.columns:
    df_today = df_today.drop_duplicates(subset=['game_id'])
else:
  df_today = df.iloc[0:0].copy()
if df_today.empty and ('date' in df.columns):
  df['date'] = df['date'].astype(str)
  df_today = df[df['date'] == target].copy()
# Backfill schedule start times (and related display fields) from today's games file.
try:
  if (not df_today.empty) and ('game_id' in df_today.columns):
    df_today['game_id'] = df_today['game_id'].astype(str).str.replace(r'\\.0$','', regex=True)
    df_today = df_today[df_today['game_id'].notna() & (df_today['game_id'].astype(str) != '')].copy()
    try:
      gc = pd.read_csv(games_curr)
      if 'game_id' in gc.columns:
        gc['game_id'] = gc['game_id'].astype(str).str.replace(r'\\.0$','', regex=True)
      keep_cols = [c for c in ['game_id','start_time','start_time_iso','start_time_local','start_tz_abbr'] if c in gc.columns]
      if keep_cols:
        gc_small = gc[keep_cols].drop_duplicates(subset=['game_id'])
        df_today = df_today.merge(gc_small, on='game_id', how='left', suffixes=('','_sched'))
        for c in ['start_time','start_time_iso','start_time_local','start_tz_abbr']:
          sc = c + '_sched'
          if (c in df_today.columns) and (sc in df_today.columns):
            df_today[c] = df_today[c].where(df_today[c].notna() & (df_today[c].astype(str) != ''), df_today[sc])
            df_today.drop(columns=[sc], inplace=True, errors='ignore')
          elif (c not in df_today.columns) and (sc in df_today.columns):
            df_today[c] = df_today[sc]
            df_today.drop(columns=[sc], inplace=True, errors='ignore')
    except Exception as e:
      print('[warn] schedule backfill failed (closing): ' + str(e))
except Exception:
  pass
# Debug: show unique dates and gid_today size, avoid f-strings to prevent parser issues
try:
  udates = sorted(set(df['date'].astype(str)))[:5] if 'date' in df.columns else []
  print('[debug] unique_dates_in_merged_closing=' + str(udates))
except Exception:
  pass
print('[debug] gid_today_count=' + str(len(gid_today)))
df_today.to_csv(outp, index=False)
print(f'Filtered games_with_closing.csv -> {len(df)} total, {len(df_today)} rows for {target}')
"@
          & $VenvPython -c $pyFilterClosing
          # Switch mergedToday to closing fallback if it has rows
          try {
            $rowsClosing = @(Import-Csv -LiteralPath $mergedClosingToday)
            if ($rowsClosing -and $rowsClosing.Count -gt 0) {
              $mergedToday = $mergedClosingToday
            }
          } catch { Write-Warning "closing fallback probe failed: $($_)" }
        } else {
          Write-Warning "games_with_closing.csv not found; cannot fallback to closing lines."
        }
      }
    } catch { Write-Warning "last-odds filter probe failed: $($_)" }
  } else {
    Write-Warning "Merged last odds file not found at $mergedAll; downstream joins may be sparse."
  }

  Write-Section "7b) Align predictions to period and compute edges"
  $predsToday = Join-Path $OutDir ("predictions_" + $todayIso + ".csv")
  $alignCsv = Join-Path $OutDir ("align_period_" + $todayIso + ".csv")
  $alignEdges = Join-Path $OutDir ("align_period_" + $todayIso + "_edges.csv")
  try {
    & $VenvPython -m ncaab_model.cli align-period-preds --merged-csv $mergedToday --predictions-csv $predsToday --out $alignCsv --half-ratio 0.485 --margin-half-ratio 0.5
  }
  catch {
    Write-Warning "align-period-preds failed: $($_)"
  }

  # 7b.post) Ensure display snapshot exists (fallback from edges if enriched had no rows)
  try {
    $displaySnap = Join-Path $OutDir ("predictions_display_" + $todayIso + ".csv")
    $needBuild = $false
    if (-not (Test-Path -LiteralPath $displaySnap)) {
      $needBuild = $true
    } else {
      try {
        $rows = Import-Csv -LiteralPath $displaySnap -ErrorAction Stop
        if (-not $rows -or ($rows | Measure-Object).Count -le 0) { $needBuild = $true }
      } catch {
        $needBuild = $true
      }
    }
    if ($needBuild -and (Test-Path -LiteralPath $alignEdges)) {
      Write-Host "[display] Building predictions_display from edges -> $displaySnap" -ForegroundColor Cyan
      & $VenvPython (Join-Path $RepoRoot 'scripts\generate_display_from_edges.py') $todayIso
      # Archive copy for lightweight browsing
      try {
        $archiveDir = Join-Path $OutDir ("archive\" + $todayIso)
        New-Item -ItemType Directory -Path $archiveDir -Force | Out-Null
        if (Test-Path -LiteralPath $displaySnap) {
          Copy-Item -LiteralPath $displaySnap -Destination (Join-Path $archiveDir ("predictions_display_" + $todayIso + ".csv")) -Force
        }
      } catch { Write-Warning "Failed to archive display snapshot: $($_)" }
    }
  } catch { Write-Warning "Display-from-edges fallback failed: $($_)" }

  Write-Host "[stake] Stake sheets are deprecated; skipping stake sizing + compare." -ForegroundColor DarkGray

  # Optional hygiene: archive any legacy stake-sheet artifacts that still exist
  # (from older runs) so they don't confuse day-to-day ops.
  try {
    $stakeCandidates = @(
      (Join-Path $OutDir 'stake_sheet_today.csv'),
      (Join-Path $OutDir 'stake_sheet_today_cal.csv'),
      (Join-Path $OutDir 'stake_sheet_today_iso.csv'),
      (Join-Path $OutDir 'stake_sheet_today_compare.csv'),
      (Join-Path $OutDir 'stake_sheet_today_summary.csv'),
      (Join-Path $OutDir 'stake_sheet_calibrated.csv'),
      (Join-Path $OutDir 'stake_risk_summary.csv'),
      (Join-Path $OutDir 'stake_sheet.csv')
    )
    $existingStake = @($stakeCandidates | Where-Object { Test-Path -LiteralPath $_ })
    if ($existingStake.Count -gt 0) {
      $stakeBackupDir = Join-Path $OutDir ("_stake_sheets_backup_{0}" -f (Get-Date).ToString('yyyyMMdd_HHmmss'))
      New-Item -ItemType Directory -Path $stakeBackupDir -Force | Out-Null
      foreach ($p in $existingStake) {
        try { Move-Item -LiteralPath $p -Destination (Join-Path $stakeBackupDir (Split-Path -Leaf $p)) -Force } catch {}
      }
      Write-Host "[stake] Archived legacy stake-sheet artifacts -> $stakeBackupDir" -ForegroundColor DarkGray
    }
  } catch {}

  # Enforce invariant: no NaN/Inf tokens anywhere in persisted outputs
  Write-Section '10.z) Sanitize outputs (eliminate NaN/Inf)'
  try {
    & $VenvPython -m ncaab_model.cli sanitize-artifacts --date $todayIso --date $prevDate --outputs-dir $OutDir
    if ($LASTEXITCODE -ne 0) {
      Add-CriticalFailure "sanitize-artifacts failed (exit=$LASTEXITCODE)"
    }
  } catch {
    Add-CriticalFailure "sanitize-artifacts crashed: $($_)"
  }

    if (-not $SkipGitPush) {
      Write-Section "11) Commit and push updated data files"
      # Add a small, curated set of whitelisted artifacts per .gitignore
      $toStage = @()
      # Keep small set of stable merged references
      $gwl = Join-Path $OutDir 'games_with_last.csv'
      if (Test-Path $gwl) { $toStage += $gwl }
      $gwc = Join-Path $OutDir 'games_with_closing.csv'
      if (Test-Path $gwc) { $toStage += $gwc }
      $pri = Join-Path $OutDir 'priors.csv'
      if (Test-Path $pri) { $toStage += $pri }

      # Live Lens interval calibration (small JSON). If missing, Live Lens will fallback to uncalibrated.
      $liveIntervalCal = Join-Path $OutDir 'live_interval_calibration.json'
      if (Test-Path $liveIntervalCal) { $toStage += $liveIntervalCal }

      # Dated artifacts for reproducibility (allowlisted in .gitignore)
      $gamesToday = Join-Path $OutDir ("games_" + $todayIso + ".csv")
      if (Test-Path $gamesToday) { $toStage += $gamesToday }
      $oddsTodayDated = Join-Path $OutDir ("odds_" + $todayIso + ".csv")
      if (Test-Path $oddsTodayDated) { $toStage += $oddsTodayDated }
      $mergedTodayDated = Join-Path $OutDir ("games_with_odds_" + $todayIso + ".csv")
      if (Test-Path $mergedTodayDated) { $toStage += $mergedTodayDated }
      $predsTodayDated = Join-Path $OutDir ("predictions_" + $todayIso + ".csv")
      if (Test-Path $predsTodayDated) { $toStage += $predsTodayDated }
  $predsModelToday = Join-Path $OutDir ("predictions_model_" + $todayIso + ".csv")
  if (Test-Path $predsModelToday) { $toStage += $predsModelToday }
  $predsModelCalibToday = Join-Path $OutDir ("predictions_model_calibrated_" + $todayIso + ".csv")
  if (Test-Path $predsModelCalibToday) { $toStage += $predsModelCalibToday }
  $predsModelIntervalToday = Join-Path $OutDir ("predictions_model_interval_" + $todayIso + ".csv")
  if (Test-Path $predsModelIntervalToday) { $toStage += $predsModelIntervalToday }
  # Interval meta JSON (RMSE + z values) for reproducibility if present
  $predsModelIntervalMetaToday = Join-Path $OutDir ("predictions_model_interval_" + $todayIso + ".json")
  if (Test-Path $predsModelIntervalMetaToday) { $toStage += $predsModelIntervalMetaToday }
  # Coverage enriched + status sidecar for frontend consumption
  $enrichedToday = Join-Path $OutDir ("predictions_unified_enriched_" + $todayIso + ".csv")
  if (Test-Path $enrichedToday) { $toStage += $enrichedToday }
  $coverageSummary = Join-Path $OutDir ("coverage_status_summary_" + $todayIso + ".json")
  if (Test-Path $coverageSummary) { $toStage += $coverageSummary }
  # Daily results (recaps) for previous and current dates
  $dailyResultsPrev = Join-Path $OutDir ("daily_results\results_" + $prevDate + ".csv")
  if (Test-Path $dailyResultsPrev) { $toStage += $dailyResultsPrev }
  $dailyResultsToday = Join-Path $OutDir ("daily_results\results_" + $todayIso + ".csv")
  if (Test-Path $dailyResultsToday) { $toStage += $dailyResultsToday }

  # Newly produced meta/stability/calibration artifacts
  $metaMetrics = Join-Path $OutDir 'meta_probs_metrics.json'
  if (Test-Path $metaMetrics) { $toStage += $metaMetrics }
  $metaMetricsLgbm = Join-Path $OutDir 'meta_probs_metrics_lgbm.json'
  if (Test-Path $metaMetricsLgbm) { $toStage += $metaMetricsLgbm }
  # Quantile model + proper scoring for today
  $quantModel = Join-Path $OutDir 'quantile_model.json'
  if (Test-Path $quantModel) { $toStage += $quantModel }
  $quantMetrics = Join-Path $OutDir 'quantile_metrics.csv'
  if (Test-Path $quantMetrics) { $toStage += $quantMetrics }
  $quantTrend2w = Join-Path $OutDir 'quantile_trend_2w.csv'
  if (Test-Path $quantTrend2w) { $toStage += $quantTrend2w }
  $quantTrendWeekly = Join-Path $OutDir 'quantile_trend_weekly.csv'
  if (Test-Path $quantTrendWeekly) { $toStage += $quantTrendWeekly }
  $scoreToday = Join-Path $OutDir ("scoring_" + $todayIso + ".json")
  if (Test-Path $scoreToday) { $toStage += $scoreToday }
  $probStability = Get-ChildItem -Path $OutDir -Filter ('prob_stability_' + $todayIso + '.json') -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($probStability) { $toStage += $probStability.FullName }
  $autoCal = Get-ChildItem -Path $OutDir -Filter 'auto_refresh_calibration_*.json' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($autoCal) { $toStage += $autoCal.FullName }
  $metaExplain = Get-ChildItem -Path $OutDir -Filter ('meta_explain_' + $todayIso + '.json') -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($metaExplain) { $toStage += $metaExplain.FullName }
  $metaECE = Join-Path $OutDir 'meta_ece.json'
  if (Test-Path $metaECE) { $toStage += $metaECE }
  $metaRel = Join-Path $OutDir 'meta_reliability.csv'
  if (Test-Path $metaRel) { $toStage += $metaRel }
  $metaCal = Join-Path $OutDir 'meta_calibration.json'
  if (Test-Path $metaCal) { $toStage += $metaCal }
  $driftWeekly = Join-Path $OutDir 'drift_summary_weekly.csv'
  if (Test-Path $driftWeekly) { $toStage += $driftWeekly }

  # Newly produced aligned artifacts
  $alignCsv = Join-Path $OutDir ("align_period_" + $todayIso + ".csv")
  if (Test-Path $alignCsv) { $toStage += $alignCsv }
  $alignEdges = Join-Path $OutDir ("align_period_" + $todayIso + "_edges.csv")
  if (Test-Path $alignEdges) { $toStage += $alignEdges }
  # Picks raw snapshot for recommendations fallback
  $picksRaw = Join-Path $OutDir 'picks_raw.csv'
  if (Test-Path $picksRaw) { $toStage += $picksRaw }

  # ATS picks for UI consumption (publish per-date CSV)
  $picksAts = Join-Path $OutDir ("picks\ats_picks_" + $todayIso + ".csv")
  if (Test-Path $picksAts) { $toStage += $picksAts }

  # Model selection and conformal autotune artifacts
  $qcv = Join-Path $OutDir 'quantile_cv_results.csv'
  if (Test-Path $qcv) { $toStage += $qcv }
  $qhist = Join-Path $OutDir 'quantiles_history.csv'
  if (Test-Path $qhist) { $toStage += $qhist }
  $confAuto = Join-Path $OutDir 'conformal_autotune.json'
  if (Test-Path $confAuto) { $toStage += $confAuto }

  # Backtest latest summaries for dashboard
  $btLatest = Join-Path $OutDir 'backtest_summary_latest.csv'
  if (Test-Path $btLatest) { $toStage += $btLatest }

  # Frontend display snapshots and enriched predictions for current date
  # Rebuild display from enriched to ensure non-even margins before staging
  try {
    $enrichedToday = Join-Path $OutDir ("predictions_unified_enriched_" + $todayIso + ".csv")
    if (Test-Path $enrichedToday) {
      & $VenvPython scripts/rebuild_display_from_enriched.py --date $todayIso
    }
  } catch { Write-Warning "rebuild_display_from_enriched failed: $($_)" }

  $predDisplay = Join-Path $OutDir ("predictions_display_" + $todayIso + ".csv")
  if (Test-Path $predDisplay) { $toStage += $predDisplay }
  $predEnriched = Join-Path $OutDir ("predictions_unified_enriched_" + $todayIso + ".csv")
  if (Test-Path $predEnriched) { $toStage += $predEnriched }
  # 5-min trajectory artifact (used by game cards UI)
  $simSegmentsToday = Join-Path $OutDir ("sim_segments_" + $todayIso + ".csv")
  if (Test-Path $simSegmentsToday) { $toStage += $simSegmentsToday }

  # 2-min trajectory artifact (preferred by cards + Live Lens)
  $simSegments2MinToday = Join-Path $OutDir ("sim_segments_2min_" + $todayIso + ".csv")
  if (Test-Path $simSegments2MinToday) { $toStage += $simSegments2MinToday }

  # Rolling 5-min reconciliation + calibration artifacts
  $segCalib = Join-Path $OutDir 'segment_calibration_5min.json'
  if (Test-Path $segCalib) { $toStage += $segCalib }
  $segCalib2 = Join-Path $OutDir 'segment_calibration_stage2_5min.json'
  if (Test-Path $segCalib2) { $toStage += $segCalib2 }
  $segMaster = Join-Path $OutDir 'backtests\segments_5min_master.csv'
  if (Test-Path $segMaster) { $toStage += $segMaster }
  $segDailyPrev = Join-Path $OutDir ("backtests\segments_5min_daily_" + $prevDate + "_to_" + $prevDate + ".csv")
  if (Test-Path $segDailyPrev) { $toStage += $segDailyPrev }
  # Archive copy of today's display snapshot for lightweight historical browsing
  $predDisplayArchive = Join-Path $OutDir ("archive\" + $todayIso + "\predictions_display_" + $todayIso + ".csv")
  if (Test-Path $predDisplayArchive) { $toStage += $predDisplayArchive }

  # ROI backtest generation and staging
  Write-Section '10b) ROI backtest (28 days)'
  try {
    & $VenvPython scripts/backtest_roi.py --days 28 --name latest
    $roiLatest = Join-Path $OutDir 'backtest_roi_latest.csv'
    if (Test-Path $roiLatest) { $toStage += $roiLatest }
  } catch { Write-Warning "backtest_roi.py failed: $($_)" }

      # Allowlist per-date odds snapshots so historical odds persist on Render
      $oddsPrev = Join-Path $OutDir ("odds_history/odds_" + $prevDate + ".csv")
      if (Test-Path $oddsPrev) { $toStage += $oddsPrev }
      $oddsTodayHist = Join-Path $OutDir ("odds_history/odds_" + $todayIso + ".csv")
      if (Test-Path $oddsTodayHist) { $toStage += $oddsTodayHist }

      # Accuracy snapshot + diagnostics (ensure UI and coverage are in sync)
      $accSnap = Join-Path $OutDir 'metrics\season_accuracy_summary.json'
      if (Test-Path $accSnap) { $toStage += $accSnap }
      $accDailyPrev = Join-Path $OutDir ("daily_accuracy_" + $prevDate + ".json")
      if (Test-Path $accDailyPrev) { $toStage += $accDailyPrev }
      $accDailyLatest = Join-Path $OutDir 'metrics\daily_accuracy_latest.json'
      if (Test-Path $accDailyLatest) { $toStage += $accDailyLatest }
      $accDiagJson = Join-Path $OutDir 'diagnostics\accuracy_missing_by_date.json'
      if (Test-Path $accDiagJson) { $toStage += $accDiagJson }
      $accDiagCsv = Join-Path $OutDir 'diagnostics\accuracy_missing_by_date.csv'
      if (Test-Path $accDiagCsv) { $toStage += $accDiagCsv }

      # OU policy tuning + evaluation artifacts
      $ouPol = Join-Path $OutDir 'metrics\ou_selection_policy.json'
      if (Test-Path $ouPol) { $toStage += $ouPol }
      $ouEval = Join-Path $OutDir 'metrics\ou_selection_eval.json'
      if (Test-Path $ouEval) { $toStage += $ouEval }

      # Daily update status JSON for observability on Render
      $statusJson = Join-Path $OutDir ("logs\daily_update_status_" + $todayIso + ".json")
      if (Test-Path $statusJson) { $toStage += $statusJson }

      if ($toStage.Count -gt 0) {
        function Add-PathSmart {
          param(
            [Parameter(Mandatory=$true)][string]$Path
          )
          if (-not (Test-Path -LiteralPath $Path)) { return }
          $ignored = $false
          try {
            git check-ignore -q -- $Path
            if ($LASTEXITCODE -eq 0) { $ignored = $true }
          } catch { $ignored = $false }

          if ($ignored) {
            git add -f -- $Path
          } else {
            git add -- $Path
          }
        }

        # Stage curated artifacts (force-add if ignored)
        foreach ($p in $toStage) { if ($p) { Add-PathSmart -Path $p } }

        # Also stage core code paths if they exist (so code fixes ship with Option A)
        $codePaths = @(
          (Join-Path $RepoRoot 'app.py'),
          (Join-Path $RepoRoot 'render.yaml'),
          (Join-Path $RepoRoot 'scripts\daily_update.ps1'),
          (Join-Path $RepoRoot 'scripts\upload_artifacts_to_render.ps1'),
          (Join-Path $RepoRoot 'scripts\upsert_segments_5min_master.py'),
          (Join-Path $RepoRoot 'scripts\refresh_segment_calibration_5min.py'),
          (Join-Path $RepoRoot 'scripts\fit_segment_stage2_bias_5min.py'),
          (Join-Path $RepoRoot 'scripts\refresh_segment_stage2_bias_5min.py'),
          (Join-Path $RepoRoot 'scripts\persist_odds_into_daily_results.py'),
          (Join-Path $RepoRoot 'templates\index.html'),
          (Join-Path $RepoRoot 'static\css\app.css'),
          (Join-Path $RepoRoot 'src\ncaab_model\cli.py'),
          (Join-Path $RepoRoot 'src\simulation\game_sim.py')
        )
        foreach ($cp in $codePaths) { if ($cp) { Add-PathSmart -Path $cp } }
        # Optionally stage variance diagnostics if produced today
        $varTotalPath = Join-Path $OutDir ("variance/variance_total_" + $todayIso + ".json")
        $varMarginPath = Join-Path $OutDir ("variance/variance_margin_" + $todayIso + ".json")
        if (Test-Path $varTotalPath) { Add-PathSmart -Path $varTotalPath }
        if (Test-Path $varMarginPath) { Add-PathSmart -Path $varMarginPath }
  # Inference variance summary (produced earlier in step 5d) if present
  $infVarSummaryPath = Join-Path $OutDir ("variance/inference_variance_" + $todayIso + ".json")
  if (Test-Path $infVarSummaryPath) { Add-PathSmart -Path $infVarSummaryPath }
        # Guard: Unstage any files not explicitly allowlisted to keep repo lean
        try {
          $stagedRel = git diff --name-only --cached
          $allowedRel = @()
          $repoRootFull = [System.IO.Path]::GetFullPath($RepoRoot).TrimEnd('\','/')
          function Get-RelPath {
            param([string]$Full)
            if (-not $Full) { return $null }
            $full2 = [System.IO.Path]::GetFullPath($Full)
            $r1 = $repoRootFull.ToLowerInvariant()
            $r2 = $full2.ToLowerInvariant()
            if ($r2.StartsWith($r1)) {
              return ($full2.Substring($repoRootFull.Length)).TrimStart('\','/') -replace '\\', '/'
            }
            return ($full2 -replace '\\', '/').Trim()
          }

          foreach ($p in $toStage) { $rel = Get-RelPath -Full $p; if ($rel) { $allowedRel += $rel } }
          foreach ($cp in $codePaths) { $rel = Get-RelPath -Full $cp; if ($rel) { $allowedRel += $rel } }
          if (Test-Path $varTotalPath) { $allowedRel += (Get-RelPath -Full $varTotalPath) }
          if (Test-Path $varMarginPath) { $allowedRel += (Get-RelPath -Full $varMarginPath) }
          if (Test-Path $infVarSummaryPath) { $allowedRel += (Get-RelPath -Full $infVarSummaryPath) }

          # Normalize and trim paths for reliable comparison with git output
          $allowedRel = $allowedRel | ForEach-Object { ($_ -replace '\\', '/').Trim() } | Where-Object { $_ }
          $stagedRel = $stagedRel | ForEach-Object { ($_ -replace '\\', '/').Trim() } | Where-Object { $_ }
          foreach ($s in $stagedRel) {
            # Never unstage core daily UI/data snapshots even if comparison fails
            if (
              $s -like 'outputs/predictions_display_*' -or
              $s -like 'outputs/predictions_unified_enriched_*' -or
              $s -like 'outputs/archive/*/predictions_display_*' -or
              $s -like 'outputs/daily_results/results_*' -or
              $s -like 'outputs/align_period_*_edges.csv' -or
              $s -like 'outputs/picks_raw.csv' -or
              $s -like 'outputs/picks/ats_picks_*'
            ) { continue }
            if (-not ($allowedRel -contains $s)) {
              Write-Host "[unstage] Non-allowlisted: $s" -ForegroundColor DarkGray
              git restore --staged -- $s
            }
          }
        } catch {
          Write-Warning "Failed to enforce allowlist on staged files: $($_)"
        }
        # Capture Render baseline version BEFORE pushing, so we can wait for auto-deploy to finish
        $baseUrl = $script:RenderBaseUrlEff
        function Get-RenderVersion {
          param()
          try {
            $ts = [int](Get-Date -UFormat %s)
            return (Invoke-RestMethod -Uri ("{0}/api/version?t={1}" -f $baseUrl, $ts) -Method Get)
          } catch { return $null }
        }
        function Wait-For-Render-Deploy {
          param(
            [object]$baseline,
            [int]$timeoutSec = 240,
            [int]$intervalMs = 3000
          )
          $deadline = (Get-Date).AddSeconds($timeoutSec)
          $changed = $false
          while ((Get-Date) -lt $deadline) {
            Start-Sleep -Milliseconds $intervalMs
            try {
              $ver = Get-RenderVersion
              if ($ver) {
                $sha = $ver.app_sha
                $bt  = $ver.build_time_utc
                $bsha = if ($baseline) { $baseline.app_sha } else { $null }
                $bbt  = if ($baseline) { $baseline.build_time_utc } else { $null }
                if (($bsha -and $sha -and $sha -ne $bsha) -or ($bbt -and $bt -and $bt -ne $bbt) -or (-not $bbt -and $bt)) { $changed = $true; break }
              }
            } catch { }
          }
          if ($changed) { Write-Host "[Render] Detected new deployment version." -ForegroundColor Green }
          else { Write-Host "[Render] Version unchanged within wait window; proceeding." -ForegroundColor Yellow }
        }
        $baselineVersion = $null
        try {
          $baselineVersion = Get-RenderVersion
          if ($baselineVersion) {
            Write-Host ("[Render] Baseline version: sha={0} build_time={1}" -f $baselineVersion.app_sha, $baselineVersion.build_time_utc) -ForegroundColor Gray
          } else {
            Write-Host "[Render] Baseline version unavailable (pre-push)." -ForegroundColor Yellow
          }
        } catch { Write-Host "[Render] Baseline version check failed." -ForegroundColor Yellow }

        $msg = if ($GitCommitMessage) { $GitCommitMessage } else { "chore(data+ui): update outputs and UI for $prevDate (today $todayIso)" }
        $stagedNow = git diff --name-only --cached
        if ($stagedNow) {
          try {
            git commit -m $msg
            git push
          } catch {
            Add-CriticalFailure "git commit/push failed: $($_)"
          }
          try {
            Write-Host "[Render] Waiting for auto-deploy to complete before uploads..." -ForegroundColor Gray
            Wait-For-Render-Deploy -baseline $baselineVersion -timeoutSec 240 -intervalMs 3000
          } catch { Write-Warning ("Auto-deploy wait failed: {0}" -f $_.Exception.Message) }
        }
        else {
          Write-Host "No staged changes to commit." -ForegroundColor Yellow
        }
      }
      else {
        Write-Host "No whitelisted data artifacts found to stage." -ForegroundColor Yellow
      }
    }

    # Upload artifacts to Render by default (opt-out with -SkipRenderUpload)
    $doRenderUpload = $true
    if ($PSBoundParameters.ContainsKey('SkipRenderUpload') -and $SkipRenderUpload.IsPresent) { $doRenderUpload = $false }
    # Back-compat: if user explicitly supplied -UploadToRender, honor true; otherwise default remains true
    if ($doRenderUpload -or $UploadToRender.IsPresent) {
      Write-Section "11b) Upload artifacts to Render + verify ($prevDate, $todayIso)"
      try {
        $uploader = Join-Path $RepoRoot 'scripts\upload_artifacts_to_render.ps1'
        if (Test-Path $uploader) {
          # Preflight: ensure today's sim artifacts exist locally before attempting upload.
          # The sim generation earlier is best-effort; if it failed, Render cards will be missing the Sim row.
          function Test-HasDataRow {
            param([string]$Path)
            if (-not (Test-Path -LiteralPath $Path)) { return $false }
            try {
              $n = (Get-Content -LiteralPath $Path -TotalCount 2 | Measure-Object).Count
              return ($n -ge 2)
            } catch { return $false }
          }

          $simQuantilesToday = Join-Path $OutDir ("sim_quantiles_" + $todayIso + ".csv")
          $simBlendToday     = Join-Path $OutDir ("sim_blend_" + $todayIso + ".csv")
          $simSegmentsToday2 = Join-Path $OutDir ("sim_segments_" + $todayIso + ".csv")
          $simSegments2MinToday2 = Join-Path $OutDir ("sim_segments_2min_" + $todayIso + ".csv")
          $needSim = (-not (Test-HasDataRow $simQuantilesToday)) -or (-not (Test-HasDataRow $simBlendToday)) -or (-not (Test-HasDataRow $simSegmentsToday2)) -or (-not (Test-HasDataRow $simSegments2MinToday2))

          if ($needSim) {
            Write-Section "11b.pre) Missing local sim artifacts; regenerating sims for $todayIso"
            try {
              if (-not $env:NCAAB_SIM_SEED -or $env:NCAAB_SIM_SEED.Trim() -eq '') {
                $env:NCAAB_SIM_SEED = $todayIso.Replace('-','')
              }
              if (-not $env:NCAAB_SIM_BLEND_EVENT_PACE -or $env:NCAAB_SIM_BLEND_EVENT_PACE.Trim() -eq '') {
                $env:NCAAB_SIM_BLEND_EVENT_PACE = '1'
              }
              if (-not $env:NCAAB_SIM_MEAN_SOURCE -or $env:NCAAB_SIM_MEAN_SOURCE.Trim() -eq '') {
                $env:NCAAB_SIM_MEAN_SOURCE = 'auto'
              }
              try { & $VenvPython scripts/validate_sim_inputs.py $todayIso $OutDir } catch { Write-Warning "validate_sim_inputs retry failed: $($_)" }
              try { & $VenvPython scripts/run_game_simulations.py $todayIso $OutDir } catch { Write-Warning "run_game_simulations retry failed: $($_)" }
              # Ensure 2-min segments exist for cards + Live Lens.
              try { & $VenvPython scripts/run_game_simulations.py $todayIso $OutDir --segments-grid-min 2 --segments-out-prefix sim_segments_2min_ --quantiles-out-prefix sim_quantiles_2min_ --meta-out-prefix sim_meta_2min_ } catch { Write-Warning "run_game_simulations (2-min) retry failed: $($_)" }
              try {
                $BlendSimWeight = if ($env:BLEND_SIM_WEIGHT) { [double]$env:BLEND_SIM_WEIGHT } else { 0.2 }
                & $VenvPython scripts/blend_sim_quantiles.py $todayIso $OutDir $BlendSimWeight
              } catch { Write-Warning "blend_sim_quantiles retry failed: $($_)" }
            } catch {
              Write-Warning "Sim regeneration preflight failed: $($_)"
            }

            $needSimAfter = (-not (Test-HasDataRow $simQuantilesToday)) -or (-not (Test-HasDataRow $simBlendToday)) -or (-not (Test-HasDataRow $simSegmentsToday2)) -or (-not (Test-HasDataRow $simSegments2MinToday2))
            if ($needSimAfter) {
              Write-Warning "Local sim artifacts are still missing/empty after retry; Render cards may lack Sim rows. Expected: $simQuantilesToday"
            } else {
              Write-Host "[Sim] Local sim artifacts regenerated; proceeding with upload." -ForegroundColor Green
            }
          }

          # Determine redeploy behavior: default false unless explicitly requested; explicit -TriggerRenderRedeploy forces true
          $doRedeploy = $false
          if ($PSBoundParameters.ContainsKey('SkipRenderRedeploy') -and $SkipRenderRedeploy.IsPresent) { $doRedeploy = $false }
          if ($TriggerRenderRedeploy.IsPresent) { $doRedeploy = $true }

          # Upload both yesterday (for reconciliation/results) and today (for cards/recs).
          # Note: Render storage is ephemeral unless a persistent disk is enabled; this ensures data is present post-run.
          $datesToUpload = @($prevDate, $todayIso) | Where-Object { $_ -and $_.Trim() -ne '' } | Select-Object -Unique

          if ($doRedeploy) {
            # Trigger at most one redeploy to avoid wiping uploads mid-loop.
            Write-Host "[Render] Uploading artifacts and triggering a single redeploy (sanitized display)" -ForegroundColor Cyan
            powershell.exe -ExecutionPolicy Bypass -File $uploader -Date $todayIso -TriggerRedeploy
            foreach ($d in $datesToUpload) {
              if ($d -eq $todayIso) { continue }
              Write-Host ("[Render] Uploading additional date artifacts: {0}" -f $d) -ForegroundColor Cyan
              powershell.exe -ExecutionPolicy Bypass -File $uploader -Date $d
            }
          } else {
            Write-Host "[Render] Uploading artifacts (sanitized display)" -ForegroundColor Cyan
            foreach ($d in $datesToUpload) {
              Write-Host ("[Render] Upload date artifacts: {0}" -f $d) -ForegroundColor Cyan
              powershell.exe -ExecutionPolicy Bypass -File $uploader -Date $d
            }
          }
        } else {
          Write-Warning "Uploader script not found at $uploader; skipping Render upload."
        }
      } catch {
        Write-Warning "Render upload step failed: $($_)"
      }
    } else {
      Write-Host 'SkipRenderUpload flag set; skipping Render upload.' -ForegroundColor Yellow
    }

    # Verify Render health: ensure today's predictions rows recognized and bootstrap not needed
    try {
      Write-Section "11c) Render health verification ($todayIso)"
      $baseUrl = $script:RenderBaseUrlEff
      $healthUri = "$baseUrl/api/health?date=$todayIso"
      $rowsTodayUri = "$baseUrl/api/rows-today"
      $health = Invoke-RestMethod -Uri $healthUri -Method Get
      $predsTodayRows = if ($health.today) { $health.today.preds_today_rows } else { $null }
      $displayRows = $health.display_rows
      $enrichedRows = $health.enriched_rows
      $needBootstrap = $health.need_bootstrap
      $predSource = $health.predictions_source
      Write-Host ("[Health] preds_today_rows={0} display_rows={1} enriched_rows={2} need_bootstrap={3}" -f $predsTodayRows, $displayRows, $enrichedRows, $needBootstrap) -ForegroundColor White
      if ($predSource) { Write-Host ("[Health] predictions_source={0}" -f $predSource) -ForegroundColor Gray }
      # Cross-check rows-today snapshot
      try {
        $rt = Invoke-RestMethod -Uri $rowsTodayUri -Method Get
        Write-Host ("[RowsToday] date={0} row_count={1} source={2}" -f $rt.date, $rt.row_count, $rt.source) -ForegroundColor Gray
      } catch { Write-Warning "rows-today check failed: $($_.Exception.Message)" }
      # Recommendations parity verification (robust count)
      try {
        $recUri = "$baseUrl/api/recommendations?date=$todayIso"
        $recs = Invoke-RestMethod -Uri $recUri -Method Get
        $recCount = if ($recs) {
          if ($recs.PSObject.Properties.Name -contains 'rows' -and ($recs.rows -is [int])) {
            $recs.rows
          } elseif ($recs.PSObject.Properties.Name -contains 'data') {
            ($recs.data | Measure-Object).Count
          } elseif ($recs.PSObject.Properties.Name -contains 'recommendations') {
            ($recs.recommendations | Measure-Object).Count
          } else { 0 }
        } else { -1 }
        Write-Host ("[Recommendations] date={0} count={1}" -f $todayIso, $recCount) -ForegroundColor Gray
        # Breakdown by market code for coverage diagnostics
        try {
          $ouCount = 0; $atsCount = 0; $mlCount = 0
          if ($recs -and ($recs.PSObject.Properties.Name -contains 'data')) {
            foreach ($it in $recs.data) {
              $code = ''
              if ($it.PSObject.Properties.Name -contains 'code') { $code = [string]$it.code }
              elseif ($it.PSObject.Properties.Name -contains 'rec_code') { $code = [string]$it.rec_code }
              if ($null -eq $code) { $code = '' } else { $code = [string]$code }
              $code = $code.ToUpper()
              switch ($code) {
                'OU' { $ouCount++ }
                'ATS' { $atsCount++ }
                'ML' { $mlCount++ }
                Default { }
              }
            }
          }
          Write-Host ("[Recommendations] OU={0} ATS={1} ML={2}" -f $ouCount, $atsCount, $mlCount) -ForegroundColor Gray
        } catch { Write-Warning ("recommendations market breakdown failed: {0}" -f $_.Exception.Message) }
        if ($displayRows -gt 0 -and $recCount -ge 0 -and $recCount -lt $displayRows) {
          Write-Warning ("Recommendations count ({0}) is less than display rows ({1}); UI may not show full-card OU yet." -f $recCount, $displayRows)
        }
        # Debug artifacts: confirm ATS picks and picks_raw presence for the date
        try {
          $dbgUri = "$baseUrl/api/debug_artifacts?date=$todayIso"
          $dbg = Invoke-RestMethod -Uri $dbgUri -Method Get
          $atsKey = "picks/ats_picks_${todayIso}.csv"
          $atsInfoProp = $dbg.artifacts.PSObject.Properties | Where-Object { $_.Name -eq $atsKey } | Select-Object -First 1
          $atsRows = if ($atsInfoProp) { $atsInfoProp.Value.rows } else { $null }
          $simKey = "sim_quantiles_${todayIso}.csv"
          $simInfoProp = $dbg.artifacts.PSObject.Properties | Where-Object { $_.Name -eq $simKey } | Select-Object -First 1
          $simRows = if ($simInfoProp) { $simInfoProp.Value.rows } else { $null }
          $prInfoProp = $dbg.artifacts.PSObject.Properties | Where-Object { $_.Name -eq 'picks_raw.csv' } | Select-Object -First 1
          $prRows = if ($prInfoProp) { $prInfoProp.Value.rows } else { $null }
          $atsRowsInt = if ($null -ne $atsRows) { [int]$atsRows } else { 0 }
          $simRowsInt = if ($null -ne $simRows) { [int]$simRows } else { 0 }
          Write-Host ("[Artifacts] picks_raw_rows={0} ats_picks_rows={1} sim_quantiles_rows={2}" -f $prRows, $atsRows, $simRows) -ForegroundColor Gray
          if (($null -eq $atsRows -or $atsRowsInt -le 0) -and $displayRows -gt 0) {
            Write-Warning "ATS picks artifact missing or empty; API will synthesize spreads fallback, but consider re-generating ats_picks for full coverage."
          } elseif ($displayRows -gt 0 -and $atsRowsInt -lt $displayRows) {
            Write-Warning ("ATS picks artifact incomplete: ats_picks_rows={0} < display_rows={1}; topping up from display is recommended." -f $atsRowsInt, $displayRows)
          }
          if (($null -eq $simRows -or $simRowsInt -le 0) -and $displayRows -gt 0) {
            Write-Warning "sim_quantiles artifact missing or empty; Cards will show no Sim row. Consider re-running sims + upload_artifacts_to_render.ps1 for today."
          } elseif ($displayRows -gt 0 -and $simRowsInt -gt 0 -and $simRowsInt -lt $displayRows) {
            Write-Warning ("sim_quantiles artifact incomplete: sim_rows={0} < display_rows={1}; Sim row coverage may be partial." -f $simRowsInt, $displayRows)
          }
          # Conditional rebuild + re-upload: if server artifacts are empty, (re)build from display/enriched and persist
          try {
            $baseUrl = "https://ncaab.onrender.com"
            $outsDir = Join-Path $PWD "outputs"
            $atsLocal = Join-Path $outsDir "picks\ats_picks_${todayIso}.csv"
            $picksRawLocal = Join-Path $outsDir "picks_raw.csv"

            # Compute local row counts (excluding header) when present
            $atsLocalRows = 0
            if (Test-Path -Path $atsLocal -PathType Leaf) {
              try {
                $atsLocalRows = ((Get-Content $atsLocal | Measure-Object).Count - 1)
                if ($atsLocalRows -lt 0) { $atsLocalRows = 0 }
              } catch { $atsLocalRows = 0 }
            }
            $picksRawLocalRows = 0
            if (Test-Path -Path $picksRawLocal -PathType Leaf) {
              try {
                $picksRawLocalRows = ((Get-Content $picksRawLocal | Measure-Object).Count - 1)
                if ($picksRawLocalRows -lt 0) { $picksRawLocalRows = 0 }
              } catch { $picksRawLocalRows = 0 }
            }

            $needAtsTopUp = ($displayRows -gt 0 -and $atsRowsInt -lt $displayRows)

            # If sim artifacts are missing or incomplete on Render, regenerate locally (best-effort) and re-upload.
            # NOTE: debug_artifacts now includes sim_segments + sim_segments_2min, so we can gate on those too.
            $needSimTopUp = ($displayRows -gt 0 -and ($null -eq $simRows -or $simRowsInt -lt $displayRows))
            $simQuantLocal = Join-Path $outsDir "sim_quantiles_${todayIso}.csv"
            $simBlendLocal = Join-Path $outsDir "sim_blend_${todayIso}.csv"
            $simSegLocal   = Join-Path $outsDir "sim_segments_${todayIso}.csv"
            $simSeg2Local  = Join-Path $outsDir "sim_segments_2min_${todayIso}.csv"
            $simDiagLocal  = Join-Path $outsDir "sim_inputs_diagnostic_${todayIso}.json"
            $simCalibLocal = Join-Path $outsDir "sim_calibration.json"

            function Get-LocalRows {
              param([string]$p)
              try {
                if (-not (Test-Path -LiteralPath $p)) { return 0 }
                $n = ((Get-Content -LiteralPath $p | Measure-Object).Count - 1)
                if ($n -lt 0) { $n = 0 }
                return [int]$n
              } catch { return 0 }
            }

            if ($needSimTopUp) {
              $simLocalRows = Get-LocalRows $simQuantLocal
              $blendLocalRows = Get-LocalRows $simBlendLocal
              $segLocalRows = Get-LocalRows $simSegLocal
              $seg2LocalRows = Get-LocalRows $simSeg2Local

              if ($simLocalRows -lt $displayRows -or $blendLocalRows -lt $displayRows -or $segLocalRows -le 0 -or $seg2LocalRows -le 0) {
                Write-Host ("[Sim] Local sim artifacts missing/incomplete; regenerating for {0}" -f $todayIso) -ForegroundColor DarkCyan
                try {
                  if (-not $env:NCAAB_SIM_SEED -or $env:NCAAB_SIM_SEED.Trim() -eq '') { $env:NCAAB_SIM_SEED = $todayIso.Replace('-','') }
                  if (-not $env:NCAAB_SIM_BLEND_EVENT_PACE -or $env:NCAAB_SIM_BLEND_EVENT_PACE.Trim() -eq '') { $env:NCAAB_SIM_BLEND_EVENT_PACE = '1' }
                  if (-not $env:NCAAB_SIM_MEAN_SOURCE -or $env:NCAAB_SIM_MEAN_SOURCE.Trim() -eq '') {
                    $env:NCAAB_SIM_MEAN_SOURCE = 'auto'
                  }
                  try { & $VenvPython scripts/validate_sim_inputs.py $todayIso $outsDir } catch { Write-Warning ("validate_sim_inputs (health retry) failed: {0}" -f $_.Exception.Message) }
                  try { & $VenvPython scripts/run_game_simulations.py $todayIso $outsDir } catch { Write-Warning ("run_game_simulations (health retry) failed: {0}" -f $_.Exception.Message) }
                  try { & $VenvPython scripts/run_game_simulations.py $todayIso $outsDir --segments-grid-min 2 --segments-out-prefix sim_segments_2min_ --quantiles-out-prefix sim_quantiles_2min_ --meta-out-prefix sim_meta_2min_ } catch { Write-Warning ("run_game_simulations (2-min, health retry) failed: {0}" -f $_.Exception.Message) }
                  try {
                    $BlendSimWeight2 = if ($env:BLEND_SIM_WEIGHT) { [double]$env:BLEND_SIM_WEIGHT } else { 0.2 }
                    & $VenvPython scripts/blend_sim_quantiles.py $todayIso $outsDir $BlendSimWeight2
                  } catch { Write-Warning ("blend_sim_quantiles (health retry) failed: {0}" -f $_.Exception.Message) }
                } catch {
                  Write-Warning ("Sim regeneration (health retry) failed: {0}" -f $_.Exception.Message)
                }
              }

              # Re-upload sim artifacts via the canonical uploader script (multipart/form-data).
              # This avoids brittle raw-body uploads that can silently fail on the Flask side.
              try {
                $uploader2 = Join-Path $RepoRoot 'scripts\upload_artifacts_to_render.ps1'
                if (Test-Path -LiteralPath $uploader2) {
                  Write-Host ("[Re-upload] Running uploader (SlimSimOnly) for {0}" -f $todayIso) -ForegroundColor DarkCyan
                  powershell.exe -ExecutionPolicy Bypass -File $uploader2 -Date $todayIso -SlimSimOnly
                } else {
                  Write-Warning "Uploader script not found for sim-only re-upload; skipping."
                }
              } catch { Write-Warning ("sim artifacts re-upload via uploader failed: {0}" -f $_.Exception.Message) }
            }

            # Option A (expanded): if ATS picks are missing or incomplete on Render, synthesize full-slate from display snapshot
            if ($needAtsTopUp -and ($atsLocalRows -lt $displayRows)) {
              Write-Host ("[ATS] Local ats_picks missing/incomplete; building from display for {0}" -f $todayIso) -ForegroundColor DarkCyan
              try {
                & $VenvPython scripts/build_ats_picks_from_display.py $todayIso
              } catch {
                Write-Warning ("build_ats_picks_from_display failed: {0}" -f $_.Exception.Message)
              }
              # Refresh local ATS path + row count after rebuild
              if (Test-Path -Path $atsLocal -PathType Leaf) {
                try {
                  $atsLocalRows = ((Get-Content $atsLocal | Measure-Object).Count - 1)
                  if ($atsLocalRows -lt 0) { $atsLocalRows = 0 }
                } catch { $atsLocalRows = 0 }
              }
            }

            # Ensure picks_raw.csv exists and is non-empty when we have local ATS picks
            if ($atsLocalRows -gt 0 -and $picksRawLocalRows -le 0) {
              Write-Host ("[ATS] Rebuilding picks_raw from ats_picks for {0}" -f $todayIso) -ForegroundColor DarkCyan
              try {
                & $VenvPython scripts/make_picks_raw_from_ats.py --date $todayIso --outputs $outsDir
              } catch {
                Write-Warning ("make_picks_raw_from_ats failed: {0}" -f $_.Exception.Message)
              }
              if (Test-Path -Path $picksRawLocal -PathType Leaf) {
                try {
                  $picksRawLocalRows = ((Get-Content $picksRawLocal | Measure-Object).Count - 1)
                  if ($picksRawLocalRows -lt 0) { $picksRawLocalRows = 0 }
                } catch { $picksRawLocalRows = 0 }
              }
            }

            # Re-upload artifacts when server-side copies are missing or incomplete but local files are now populated
            if ((Test-Path -Path $atsLocal -PathType Leaf) -and ($atsLocalRows -gt 0) -and $needAtsTopUp) {
              Write-Host ("[Re-upload] Posting ATS picks -> {0}" -f $atsLocal) -ForegroundColor DarkCyan
              Invoke-WebRequest -UseBasicParsing -TimeoutSec 30 -Uri ("{0}/api/upload_ats_picks?date={1}" -f $baseUrl, $todayIso) -Method Post -InFile $atsLocal -ContentType 'text/csv' | Out-Null
            }
            if ((Test-Path -Path $picksRawLocal -PathType Leaf) -and ($picksRawLocalRows -gt 0) -and $needAtsTopUp) {
              Write-Host ("[Re-upload] Posting picks_raw -> {0}" -f $picksRawLocal) -ForegroundColor DarkCyan
              Invoke-WebRequest -UseBasicParsing -TimeoutSec 30 -Uri ("{0}/api/upload_picks_raw" -f $baseUrl) -Method Post -InFile $picksRawLocal -ContentType 'text/csv' | Out-Null
            }
          } catch { Write-Warning ("conditional rebuild/re-upload failed: {0}" -f $_.Exception.Message) }
        } catch { Write-Warning ("debug_artifacts check failed: {0}" -f $_.Exception.Message) }
      } catch {
        Write-Warning "recommendations check failed: $($_.Exception.Message)"
      }
      # Advisory: if bootstrap still flagged despite uploads, warn
      if ($needBootstrap -and ($displayRows -gt 0 -or $enrichedRows -gt 0)) {
        Write-Warning "Render health indicates need_bootstrap=true even though today's artifacts are present. Server may still be redeploying; parity should resolve shortly."
      }
    } catch {
      Write-Warning "Render health verification failed: $($_.Exception.Message)"
    }

    # Auto-start results watcher job for today's date
    try {
      $runDate = $todayIso
      $watcher = Join-Path $RepoRoot 'scripts\watch_results.ps1'
      if (Test-Path $watcher) {
        Write-Host ("Starting results watcher for {0}" -f $runDate) -ForegroundColor Cyan
        Start-Job -ScriptBlock {
          param($d,$root)
          Push-Location $root
          & ./scripts/watch_results.ps1 -Date $d -IntervalSec 30
          Pop-Location
        } -ArgumentList $runDate, $RepoRoot | Out-Null
      } else {
        Write-Warning "watch_results.ps1 not found; skipping watcher start."
      }
    } catch {
      Write-Warning "Failed to start watcher job: $($_)"
    }

  Write-Section 'DONE'
  $elapsed = (Get-Date) - $script:StartTime
  Write-Host ("Completed in {0:c}" -f $elapsed)

  # Emit structured status JSON for external diagnosis
  try {
    $statusRows = @()
    foreach ($s in $script:Steps) {
      $sec = $s.section
      $errs = if ($script:StepErrors.ContainsKey($sec)) { $script:StepErrors[$sec] } else { @() }
      $statusRows += [pscustomobject]@{
        section = $sec
        start   = $s.start.ToString('o')
        errors  = $errs
        status  = if ($errs.Count -gt 0) { 'error' } else { 'ok' }
      }
    }
    $summary = [pscustomobject]@{
      date     = $todayIso
      finished = (Get-Date).ToString('o')
      elapsed_seconds = [Math]::Round($elapsed.TotalSeconds,2)
      critical_failures = $script:CriticalFailures
      steps    = $statusRows
    }
    $diagDir = Join-Path $OutDir 'logs'
    $diagPath = Join-Path $diagDir ("daily_update_status_" + $todayIso + ".json")
    ($summary | ConvertTo-Json -Depth 6) | Out-File -FilePath $diagPath -Encoding UTF8
    Write-Host "Wrote status summary -> $diagPath" -ForegroundColor Green
  } catch {
    Write-Warning "Failed writing status summary JSON: $($_)"
  }

  if ($script:CriticalFailures.Count -gt 0) {
    Write-Host "Critical failures encountered: $($script:CriticalFailures.Count)" -ForegroundColor Red
    foreach ($cf in $script:CriticalFailures) { Write-Host " - $cf" -ForegroundColor Red }
    if ($env:NCAAB_STRICT_EXIT -eq '1') {
      Write-Host 'STRICT mode enabled (NCAAB_STRICT_EXIT=1); exiting with code 1.' -ForegroundColor Red
      exit 1
    } else {
      Write-Host 'STRICT mode disabled; returning success (0) despite failures.' -ForegroundColor Yellow
    }
  }
}
catch {
  Add-CriticalFailure "Unhandled top-level error: $($_)"
  if ($env:NCAAB_STRICT_EXIT -eq '1') { exit 1 }
}
finally {
  try {
    Stop-Transcript | Out-Null
  } catch {
    # Safely ignore when transcription isn't active (e.g., -NoTranscript)
  }
}
