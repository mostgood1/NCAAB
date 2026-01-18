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
  # Probability calibration (isotonic mapping applied during distributional stake sizing)
  [switch]$SkipProbCalibrationFit,
  [int]$ProbCalibrationLookbackDays = 60,
  [int]$ProbCalibrationMinRows = 500,
  # Render upload integration
  [switch]$UploadToRender,
  [switch]$TriggerRenderRedeploy,
  # New: Upload to Render by default; opt-out with -SkipRenderUpload
  [switch]$SkipRenderUpload,
  # New: Redeploy after upload by default; opt-out with -SkipRenderRedeploy
  [switch]$SkipRenderRedeploy
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

  # Compute dates
  $todayDate = [DateTime]::ParseExact($Today, 'yyyy-MM-dd', $null)
  $prevDate = $todayDate.AddDays(-1).ToString('yyyy-MM-dd')
  $todayIso = $todayDate.ToString('yyyy-MM-dd')

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

  # 0.pre) Fetch today's slate immediately and normalize display times (Central)
  Write-Section "0.pre) Fetch today's slate + normalize display times"
  try {
    $gamesTodayPath = Join-Path $OutDir ("games_" + $todayIso + ".csv")
    & $VenvPython -m ncaab_model.cli fetch-games --season $todayDate.Year --start $todayIso --end $todayIso --provider $Provider --out $gamesTodayPath
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
  & $VenvPython -m ncaab_model.cli fetch-games --season $todayDate.Year --start $prevDate --end $prevDate --provider $Provider @noCacheFlag --out (Join-Path $OutDir 'games_prev.csv')

  Write-Section "2) Fetch odds snapshots for $prevDate and build last/closing lines"
  & $VenvPython -m ncaab_model.cli fetch-odds-history --start $prevDate --end $prevDate --region $Region --markets "h2h,spreads,totals,spreads_1st_half,totals_1st_half,spreads_2nd_half,totals_2nd_half" --out-dir (Join-Path $OutDir 'odds_history') --mode current
  & $VenvPython -m ncaab_model.cli make-closing-lines --in-dir (Join-Path $OutDir 'odds_history') --out (Join-Path $OutDir 'closing_lines.csv')
  & $VenvPython -m ncaab_model.cli join-closing (Join-Path $OutDir 'games_prev.csv') (Join-Path $OutDir 'closing_lines.csv') --out (Join-Path $OutDir 'games_with_closing_prev.csv')
  # Also refresh master merged closing across all days using games_all.csv to avoid losing previous-day lines
  & $VenvPython -m ncaab_model.cli join-closing (Join-Path $OutDir 'games_all.csv') (Join-Path $OutDir 'closing_lines.csv') --out (Join-Path $OutDir 'games_with_closing.csv')
  # Strict last pre-tip odds (no synthetic fallback). Use small tolerance for clock skew.
  & $VenvPython -m ncaab_model.cli make-last-odds --in-dir (Join-Path $OutDir 'odds_history') --out (Join-Path $OutDir 'last_odds.csv') --tolerance-seconds 60
  & $VenvPython -m ncaab_model.cli join-last-odds (Join-Path $OutDir 'games_prev.csv') (Join-Path $OutDir 'last_odds.csv') --out (Join-Path $OutDir 'games_with_last_prev.csv')
  # Also refresh master merged last across all days using games_all.csv so prior-day odds persist
  & $VenvPython -m ncaab_model.cli join-last-odds (Join-Path $OutDir 'games_all.csv') (Join-Path $OutDir 'last_odds.csv') --out (Join-Path $OutDir 'games_with_last.csv')

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
  } else {
    Write-Host "SkipFinalizePrev flag set; skipping finalize-day for $prevDate." -ForegroundColor Yellow
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
      $fitOut = (& $VenvPython scripts/fit_sim_calibration.py --outputs $OutDir --start $startIso --end $prevDate --min-games $SimCalibrationMinGames) | Out-String
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
  Write-Section '6a.post.d.s) Monte Carlo simulations + blend'
  # Force simulation means to be feature-based (independent of model/blend).
  # This is intentionally strict: if feature rows are missing, sims may be marked sim_ok=false.
  $env:NCAAB_SIM_MEAN_SOURCE = 'features_strict'
  try {
    & $VenvPython scripts/validate_sim_inputs.py $todayIso $OutDir
  } catch {
    Write-Warning "validate_sim_inputs.py failed (continuing): $($_)"
  }
  try {
    & $VenvPython scripts/run_game_simulations.py $todayIso $OutDir
  } catch { Write-Warning "run_game_simulations.py failed: $($_)" }
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

  Write-Section '6g) Auto-refresh probability calibration (ECE/drift/age)'
  try {
    & $VenvPython scripts/auto_refresh_calibration.py --date $todayIso
  } catch { Write-Warning "auto_refresh_calibration failed: $($_)" }

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
  # Reconstruct full historical last odds merge to ensure persistence before filtering for stake sheets.
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

  if (-not $SkipStakeSheets) {
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
      Write-Warning "Merged last odds file not found at $mergedAll; stake sheet generation may fail."
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

    # Determine the actual slate date for stake sheet archiving.
    # In some runs (timezone / late-night windows), $todayIso can differ from the date inside align edges.
    $slateIso = $todayIso
    try {
      if (Test-Path -LiteralPath $alignEdges) {
        $rows = @(Import-Csv -LiteralPath $alignEdges -ErrorAction Stop)
        if ($rows -and $rows.Count -gt 0 -and ($rows[0].PSObject.Properties.Name -contains 'date')) {
          $g = $rows | Group-Object -Property date | Sort-Object Count -Descending | Select-Object -First 1
          if ($g -and $g.Name) { $slateIso = [string]$g.Name }
        }
      }
    } catch { Write-Warning "Failed inferring slateIso from align edges: $($_)" }

    Write-Section "8) Generate baseline stake sheet (edge-based Kelly)"
    $stakeBase = Join-Path $OutDir 'stake_sheet_today.csv'
    try {
      & $VenvPython -m ncaab_model.cli bankroll-optimize --merged-csv $alignEdges --out $stakeBase --bankroll 1000 --kelly-fraction 0.5 --include-markets 'totals,spreads' --min-edge-total 0.5 --min-edge-margin 0.5 --min-kelly 0.01 --max-pct-per-bet 0.03 --max-daily-risk-pct 0.10
    }
    catch {
      Write-Warning "bankroll-optimize baseline failed: $($_)"
    }

    Write-Section "9) Generate calibrated distributional stake sheet (if distributional columns present)"
  $stakeCal = Join-Path $OutDir 'stake_sheet_today_cal.csv'
  $calArtifact = Join-Path $OutDir 'models_dist\calibration_totals.json'
  $qselForCli = Join-Path $OutDir 'quantiles_selected.csv'
  # "cal" keeps the existing behavior: distributional + z-recenter probability calibration (from calibration_totals.json)
  # Keep risk controls aligned with baseline.
  $distributionalArgs = @(
    '--merged-csv', $alignEdges,
    '--out', $stakeCal,
    '--bankroll', '1000',
    '--kelly-fraction', '0.5',
    '--include-markets', 'totals,spreads',
    '--use-distributional',
    '--calibrate-probabilities',
    '--min-edge-total', '0.5',
    '--min-edge-margin', '0.5',
    '--min-kelly', '0.01',
    '--max-pct-per-bet', '0.03',
    '--max-daily-risk-pct', '0.10'
  )
    if (Test-Path $qselForCli) { $distributionalArgs += @('--quantiles-csv', $qselForCli) }
    if (Test-Path $calArtifact) { $distributionalArgs += @('--calibration-artifact', $calArtifact) }
    try {
      & $VenvPython -m ncaab_model.cli bankroll-optimize @distributionalArgs
    }
    catch {
      Write-Warning "bankroll-optimize distributional failed: $($_)"
    }

    Write-Section "9b) Generate isotonic probability-calibrated stake sheet (distributional + isotonic probs)"
    $stakeIso = Join-Path $OutDir 'stake_sheet_today_iso.csv'
    # "iso" adds isotonic probability calibration on top of distributional sizing.
    # Keep risk controls aligned with baseline.
    $isoArgs = @(
      '--merged-csv', $alignEdges,
      '--out', $stakeIso,
      '--bankroll', '1000',
      '--kelly-fraction', '0.5',
      '--include-markets', 'totals,spreads',
      '--use-distributional',
      '--calibrate-probabilities',
      '--isotonic-prob-calibration',
      '--min-edge-total', '0.5',
      '--min-edge-margin', '0.5',
      '--min-kelly', '0.01',
      '--max-pct-per-bet', '0.03',
      '--max-daily-risk-pct', '0.10'
    )
    if (Test-Path $qselForCli) { $isoArgs += @('--quantiles-csv', $qselForCli) }
    if (Test-Path $calArtifact) { $isoArgs += @('--calibration-artifact', $calArtifact) }
    try {
      & $VenvPython -m ncaab_model.cli bankroll-optimize @isoArgs
    }
    catch {
      Write-Warning "bankroll-optimize isotonic failed: $($_)"
    }

    # Enrich stake sheets with quantile columns if available
    Write-Section "9a) Annotate stake sheets with quantiles (q10/q50/q90)"
    try {
      $qselPath = Join-Path $OutDir 'quantiles_selected.csv'
      if (Test-Path $qselPath) {
        $pyAnnotate = @"
import pandas as pd
from pathlib import Path
out_dir = Path(r'$OutDir')
slate = '$slateIso'
q = pd.read_csv(out_dir/'quantiles_selected.csv')
q['game_id'] = q['game_id'].astype(str).str.replace(r'\\.0$','', regex=True)
if 'date' in q.columns:
    q = q[q['date'].astype(str) == slate]
keep = ['game_id','q10_total','q50_total','q90_total','q10_margin','q50_margin','q90_margin']
q = q[[c for c in keep if c in q.columns]].drop_duplicates('game_id')
for name in ['stake_sheet_today.csv','stake_sheet_today_cal.csv','stake_sheet_today_iso.csv']:
    p = out_dir/name
    try:
        df = pd.read_csv(p)
    except Exception:
        continue
    if 'game_id' not in df.columns:
        # cannot safely join; skip
        continue
    df['game_id'] = df['game_id'].astype(str).str.replace(r'\\.0$','', regex=True)
    # Drop existing quantile columns to avoid duplicate suffix conflicts
    df = df[[c for c in df.columns if c not in {'q10_total','q50_total','q90_total','q10_margin','q50_margin','q90_margin'}]]
    merged = df.merge(q, on='game_id', how='left')
    # Ensure a date column exists for backtests/inspection
    if 'date' not in merged.columns or merged['date'].isna().all():
      merged['date'] = slate
    merged.to_csv(p, index=False)
print('Annotated stake sheets with quantiles (if matched by game_id).')
"@
        & $VenvPython -c $pyAnnotate
      } else {
        Write-Host 'quantiles_selected.csv not found; skipping stake sheet annotation.' -ForegroundColor Yellow
      }
    } catch { Write-Warning "Stake sheet quantile annotation failed: $($_)" }

    # Archive dated copies of stake sheets for ROI backtests
    try {
      if (Test-Path $stakeBase) { Copy-Item $stakeBase (Join-Path $OutDir ("stake_sheet_" + $slateIso + "_base.csv")) -Force }
      if (Test-Path $stakeCal)  { Copy-Item $stakeCal  (Join-Path $OutDir ("stake_sheet_" + $slateIso + "_cal.csv")) -Force }
      if (Test-Path $stakeIso)  { Copy-Item $stakeIso  (Join-Path $OutDir ("stake_sheet_" + $slateIso + "_iso.csv")) -Force }
    } catch { Write-Warning "Failed archiving dated stake sheets: $($_)" }

    if ((Test-Path $stakeBase) -and (Test-Path $stakeCal)) {
      Write-Section "10) Compare baseline vs calibrated stake sheets"
      $stakeCompare = Join-Path $OutDir 'stake_sheet_today_compare.csv'
      try {
        & $VenvPython scripts/compare_stake_sheets.py --orig $stakeBase --cal $stakeCal --out $stakeCompare
      }
      catch {
        Write-Warning "Stake sheet comparison failed: $($_)"
      }
    } else {
      Write-Host "Stake sheet comparison skipped (missing one or both stake sheets)." -ForegroundColor Yellow
    }
  } else {
    Write-Host "SkipStakeSheets flag set; skipping stake sheet generation." -ForegroundColor Yellow
  }

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

  # Newly produced aligned and stake artifacts
  $alignCsv = Join-Path $OutDir ("align_period_" + $todayIso + ".csv")
  if (Test-Path $alignCsv) { $toStage += $alignCsv }
  $alignEdges = Join-Path $OutDir ("align_period_" + $todayIso + "_edges.csv")
  if (Test-Path $alignEdges) { $toStage += $alignEdges }
  # Picks raw snapshot for recommendations fallback
  $picksRaw = Join-Path $OutDir 'picks_raw.csv'
  if (Test-Path $picksRaw) { $toStage += $picksRaw }
  $stakeBase = Join-Path $OutDir 'stake_sheet_today.csv'
  if (Test-Path $stakeBase) { $toStage += $stakeBase }
  $stakeCal = Join-Path $OutDir 'stake_sheet_today_cal.csv'
  if (Test-Path $stakeCal) { $toStage += $stakeCal }
  # Synthetic calibrated stake sheet and today's calibrated snapshot
  $stakeCalibrated = Join-Path $OutDir 'stake_sheet_calibrated.csv'
  if (Test-Path $stakeCalibrated) { $toStage += $stakeCalibrated }
  $predsTodayCalibrated = Join-Path $OutDir 'predictions_today_calibrated.csv'
  if (Test-Path $predsTodayCalibrated) { $toStage += $predsTodayCalibrated }

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
          $repoRootFull = [System.IO.Path]::GetFullPath($RepoRoot).TrimEnd('\\','/')
          function Get-RelPath {
            param([string]$Full)
            if (-not $Full) { return $null }
            $full2 = [System.IO.Path]::GetFullPath($Full)
            $r1 = $repoRootFull.ToLowerInvariant()
            $r2 = $full2.ToLowerInvariant()
            if ($r2.StartsWith($r1)) {
              return ($full2.Substring($repoRootFull.Length)).TrimStart('\\','/') -replace '\\', '/'
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
        $baseUrl = "https://ncaab.onrender.com"
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
      Write-Section "11b) Upload artifacts to Render + verify ($todayIso)"
      try {
        $uploader = Join-Path $RepoRoot 'scripts\upload_artifacts_to_render.ps1'
        if (Test-Path $uploader) {
          # Determine redeploy behavior: default false unless explicitly requested; explicit -TriggerRenderRedeploy forces true
          $doRedeploy = $false
          if ($PSBoundParameters.ContainsKey('SkipRenderRedeploy') -and $SkipRenderRedeploy.IsPresent) { $doRedeploy = $false }
          if ($TriggerRenderRedeploy.IsPresent) { $doRedeploy = $true }
          if ($doRedeploy) {
            Write-Host "[Render] Uploading artifacts and triggering redeploy (sanitized display)" -ForegroundColor Cyan
            powershell.exe -ExecutionPolicy Bypass -File $uploader -Date $todayIso -TriggerRedeploy
          } else {
            Write-Host "[Render] Uploading artifacts (sanitized display)" -ForegroundColor Cyan
            powershell.exe -ExecutionPolicy Bypass -File $uploader -Date $todayIso
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
      $healthUri = "https://ncaab.onrender.com/api/health?date=$todayIso"
      $rowsTodayUri = "https://ncaab.onrender.com/api/rows-today"
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
        $recUri = "https://ncaab.onrender.com/api/recommendations?date=$todayIso"
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
          $dbgUri = "https://ncaab.onrender.com/api/debug_artifacts?date=$todayIso"
          $dbg = Invoke-RestMethod -Uri $dbgUri -Method Get
          $atsKey = "picks/ats_picks_${todayIso}.csv"
          $atsInfoProp = $dbg.artifacts.PSObject.Properties | Where-Object { $_.Name -eq $atsKey } | Select-Object -First 1
          $atsRows = if ($atsInfoProp) { $atsInfoProp.Value.rows } else { $null }
          $prInfoProp = $dbg.artifacts.PSObject.Properties | Where-Object { $_.Name -eq 'picks_raw.csv' } | Select-Object -First 1
          $prRows = if ($prInfoProp) { $prInfoProp.Value.rows } else { $null }
          $atsRowsInt = if ($atsRows -ne $null) { [int]$atsRows } else { 0 }
          Write-Host ("[Artifacts] picks_raw_rows={0} ats_picks_rows={1}" -f $prRows, $atsRows) -ForegroundColor Gray
          if (($atsRows -eq $null -or $atsRowsInt -le 0) -and $displayRows -gt 0) {
            Write-Warning "ATS picks artifact missing or empty; API will synthesize spreads fallback, but consider re-generating ats_picks for full coverage."
          } elseif ($displayRows -gt 0 -and $atsRowsInt -lt $displayRows) {
            Write-Warning ("ATS picks artifact incomplete: ats_picks_rows={0} < display_rows={1}; topping up from display is recommended." -f $atsRowsInt, $displayRows)
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
