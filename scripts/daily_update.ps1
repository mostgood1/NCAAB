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
  [string]$GitCommitMessage
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
  } else {
    Write-Host "SkipFinalizePrev flag set; skipping finalize-day for $prevDate." -ForegroundColor Yellow
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

  # Enrich meta probabilities in-place using aligned features; guard against model/schema gaps
  Write-Section '6a.post.b) Enrich meta probabilities (aligned)'
  try {
    & $VenvPython scripts/enrich_meta_probs.py $todayIso --inplace
  } catch { Write-Warning "enrich_meta_probs.py failed: $($_)" }

  # Inject sigma fields and adjusted Kelly after enrichment to ensure availability downstream
  Write-Section '6a.post.c) Inject sigma and adjusted Kelly'
  try {
    & $VenvPython scripts/inject_sigma_and_kelly.py --date $todayIso
  } catch { Write-Warning "inject_sigma_and_kelly.py failed: $($_)" }

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
    Write-Section "7) Filter merged last odds to today's slate"
    $mergedAll = Join-Path $OutDir 'games_with_last.csv'
    $mergedToday = Join-Path $OutDir 'games_with_last_today.csv'
    if (Test-Path $mergedAll) {
    $gamesCurrPath = Join-Path $OutDir "games_${todayIso}.csv"
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
    gc['game_id'] = gc['game_id'].astype(str)
    if 'date' in gc.columns:
      gc['date'] = gc['date'].astype(str)
      gc = gc[gc['date'] == target]
    gid_today = set(gc['game_id'].astype(str))
except Exception as e:
  print(f'[warn] games_curr read failed: {e}')
if 'game_id' in df.columns:
  df['game_id'] = df['game_id'].astype(str)
if 'date' in df.columns:
  df['date'] = df['date'].astype(str)
  mask_date = df['date'] == target
else:
  mask_date = pd.Series([True]*len(df))
mask_gid = df['game_id'].isin(gid_today) if gid_today else mask_date
df_today = df[mask_gid & mask_date].copy() if 'date' in df.columns else df[mask_gid].copy()
if df_today.empty:
  # Fallback: if no date column, attempt to infer using presence in games_curr IDs only
  df_today = df[df['game_id'].isin(list(gid_today))].copy() if gid_today else df.head(0)
df_today.to_csv(outp, index=False)
print(f'Filtered games_with_last.csv -> {len(df)} total, {len(df_today)} rows for {target}')
"@
      & $VenvPython -c $pyFilter
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
  $distributionalArgs = @('--merged-csv', $alignEdges, '--out', $stakeCal, '--bankroll', '1000', '--kelly-fraction', '0.5', '--include-markets', 'totals,spreads', '--use-distributional', '--calibrate-probabilities')
    if (Test-Path $qselForCli) { $distributionalArgs += @('--quantiles-csv', $qselForCli) }
    if (Test-Path $calArtifact) { $distributionalArgs += @('--calibration-artifact', $calArtifact) }
    try {
      & $VenvPython -m ncaab_model.cli bankroll-optimize @distributionalArgs
    }
    catch {
      Write-Warning "bankroll-optimize distributional failed: $($_)"
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
today = '$todayIso'
q = pd.read_csv(out_dir/'quantiles_selected.csv')
q['game_id'] = q['game_id'].astype(str).str.replace(r'\\.0$','', regex=True)
if 'date' in q.columns:
    q = q[q['date'].astype(str) == today]
keep = ['game_id','q10_total','q50_total','q90_total','q10_margin','q50_margin','q90_margin']
q = q[[c for c in keep if c in q.columns]].drop_duplicates('game_id')
for name in ['stake_sheet_today.csv','stake_sheet_today_cal.csv']:
    p = out_dir/name
    try:
        df = pd.read_csv(p)
    except Exception:
        continue
    if 'game_id' not in df.columns:
        # cannot safely join; skip
        continue
    df['game_id'] = df['game_id'].astype(str).str.replace(r'\\.0$','', regex=True)
    merged = df.merge(q, on='game_id', how='left')
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
      if (Test-Path $stakeBase) { Copy-Item $stakeBase (Join-Path $OutDir ("stake_sheet_" + $todayIso + "_base.csv")) -Force }
      if (Test-Path $stakeCal)  { Copy-Item $stakeCal  (Join-Path $OutDir ("stake_sheet_" + $todayIso + "_cal.csv")) -Force }
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

    if (-not $SkipGitPush) {
      Write-Section "11) Commit and push updated data files"
      # Add a small, curated set of whitelisted artifacts per .gitignore
      $toStage = @()
      $resPath = Join-Path $OutDir ("daily_results/results_" + $prevDate + ".csv")
      if (Test-Path $resPath) { $toStage += $resPath }
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

  # Newly produced meta/stability/calibration artifacts
  $metaMetrics = Join-Path $OutDir 'meta_probs_metrics.json'
  if (Test-Path $metaMetrics) { $toStage += $metaMetrics }
  $metaMetricsLgbm = Join-Path $OutDir 'meta_probs_metrics_lgbm.json'
  if (Test-Path $metaMetricsLgbm) { $toStage += $metaMetricsLgbm }
  # Quantile model + proper scoring for today
  $quantModel = Join-Path $OutDir 'quantile_model.json'
  if (Test-Path $quantModel) { $toStage += $quantModel }
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

  # Newly produced aligned and stake artifacts
  $alignCsv = Join-Path $OutDir ("align_period_" + $todayIso + ".csv")
  if (Test-Path $alignCsv) { $toStage += $alignCsv }
  $alignEdges = Join-Path $OutDir ("align_period_" + $todayIso + "_edges.csv")
  if (Test-Path $alignEdges) { $toStage += $alignEdges }
  $stakeBase = Join-Path $OutDir 'stake_sheet_today.csv'
  if (Test-Path $stakeBase) { $toStage += $stakeBase }
  $stakeCal = Join-Path $OutDir 'stake_sheet_today_cal.csv'
  if (Test-Path $stakeCal) { $toStage += $stakeCal }

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
  $predDisplay = Join-Path $OutDir ("predictions_display_" + $todayIso + ".csv")
  if (Test-Path $predDisplay) { $toStage += $predDisplay }
  $predEnriched = Join-Path $OutDir ("predictions_unified_enriched_" + $todayIso + ".csv")
  if (Test-Path $predEnriched) { $toStage += $predEnriched }

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

      if ($toStage.Count -gt 0) {
        foreach ($p in $toStage) { git add $p }
        # Also stage core frontend/backend files if modified today
        $codePaths = @(
          (Join-Path $RepoRoot 'app.py'),
          (Join-Path $RepoRoot 'templates\index.html'),
          (Join-Path $RepoRoot 'static\css\app.css')
        )
        foreach ($cp in $codePaths) { if (Test-Path $cp) { git add $cp } }
        # Optionally stage variance diagnostics if produced today
        $varTotalPath = Join-Path $OutDir ("variance/variance_total_" + $todayIso + ".json")
        $varMarginPath = Join-Path $OutDir ("variance/variance_margin_" + $todayIso + ".json")
        if (Test-Path $varTotalPath) { git add $varTotalPath }
        if (Test-Path $varMarginPath) { git add $varMarginPath }
  # Inference variance summary (produced earlier in step 5d) if present
  $infVarSummaryPath = Join-Path $OutDir ("variance/inference_variance_" + $todayIso + ".json")
  if (Test-Path $infVarSummaryPath) { git add $infVarSummaryPath }
        $msg = if ($GitCommitMessage) { $GitCommitMessage } else { "chore(data+ui): update outputs and UI for $prevDate (today $todayIso)" }
        $status = git status --porcelain
        if ($status) {
          git commit -m $msg
          git push
        }
        else {
          Write-Host "No data changes to commit." -ForegroundColor Yellow
        }
      }
      else {
        Write-Host "No whitelisted data artifacts found to stage." -ForegroundColor Yellow
      }
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
