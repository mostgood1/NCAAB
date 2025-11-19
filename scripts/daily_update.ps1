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
  [switch]$SkipFinalizePrev,
  [switch]$SkipStakeSheets,
  [switch]$SkipGitPush,
  [string]$GitCommitMessage
)

$ErrorActionPreference = 'Stop'
$script:StartTime = Get-Date

function Write-Section($msg) {
  Write-Host "`n==== $msg ====\n"
}

# Resolve paths
$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $VenvPython)) {
  throw "Python venv not found at $VenvPython. Create and install deps first."
}
$OutDir = Join-Path $RepoRoot 'outputs'
$LogsDir = Join-Path $OutDir 'logs'
New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null

$LogStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$LogPath = Join-Path $LogsDir "daily_update_$LogStamp.log"
Start-Transcript -Path $LogPath -Append | Out-Null

try {
  Set-Location $RepoRoot

  # Compute dates
  $todayDate = [DateTime]::ParseExact($Today, 'yyyy-MM-dd', $null)
  $prevDate = $todayDate.AddDays(-1).ToString('yyyy-MM-dd')
  $todayIso = $todayDate.ToString('yyyy-MM-dd')

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

  # Optional: model-first LightGBM/XGBoost training artifacts (independent from baseline) – only if not skipped and artifacts appear stale
  $modelDir = Join-Path $OutDir 'models'
  $latestModelTotal = Get-ChildItem -Path $modelDir -Filter 'total_model.pkl' -Recurse -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  $latestModelMargin = Get-ChildItem -Path $modelDir -Filter 'margin_model.pkl' -Recurse -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $SkipRetrain -and (-not $latestModelTotal -or -not $latestModelMargin)) {
    Write-Section '5b) Train model-first total & margin predictors (LightGBM)'
    try {
      & $VenvPython -m src.modeling.train_total --algo auto
    } catch { Write-Warning "train_total failed: $($_)" }
    try {
      & $VenvPython -m src.modeling.train_margin --algo auto
    } catch { Write-Warning "train_margin failed: $($_)" }
  } else {
    Write-Host 'Model-first artifacts present; skipping explicit retrain.'
  }

  # Ensure deterministic per-day feature rows exist for inference (lightweight placeholder ratings)
  $featuresCurr = Join-Path $OutDir 'features_curr.csv'
  if (-not (Test-Path $featuresCurr)) {
    Write-Section "5c) Generate today's placeholder features (features_curr.csv)"
    try {
      & $VenvPython -m src.modeling.gen_features_today --date $todayIso
    } catch { Write-Warning "gen_features_today failed: $($_)" }
  } else {
    Write-Host 'features_curr.csv already exists; skipping generation.'
  }

  # Run model inference to produce predictions_model_<date>.csv (used for independent edges in UI)
  $modelPredPath = Join-Path $OutDir ("predictions_model_" + $todayIso + ".csv")
  if (-not (Test-Path $modelPredPath)) {
    Write-Section '5d) Run model inference harness'
    try {
      & $VenvPython -m src.modeling.infer --date $todayIso
    } catch { Write-Warning "model inference failed: $($_)" }
  } else {
    Write-Host "Model predictions already exist for $todayIso; skipping inference." -ForegroundColor DarkGray
  }

  Write-Section "6) Fetch today's schedule, odds, and run predictions/picks"
  $dailyArgs = @('daily-run', '--date', $todayIso, '--season', $todayDate.Year, '--region', $Region, '--provider', $Provider, '--segment', 'team', '--preseason-weight', '0.4', '--threshold', '1.5', '--default-price', '-110')
  if ($NoCache.IsPresent) { $dailyArgs += '--no-use-cache' }
  & $VenvPython -m ncaab_model.cli @dailyArgs

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
      $pyFilter = @"
import pandas as pd, sys
inp = r'$mergedAll'
outp = r'$mergedToday'
target = '$todayIso'
try:
    df = pd.read_csv(inp)
except Exception as e:
    print(f'[read-fail] {e}')
    sys.exit(1)
if 'date' in df.columns:
    df['date'] = df['date'].astype(str)
    df_today = df[df['date'] == target].copy()
else:
    df_today = df.copy()
df_today.to_csv(outp, index=False)
print(f'Filtered games_with_last.csv -> {len(df_today)} rows for {target}')
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
  $distributionalArgs = @('--merged-csv', $alignEdges, '--out', $stakeCal, '--bankroll', '1000', '--kelly-fraction', '0.5', '--include-markets', 'totals,spreads', '--use-distributional', '--calibrate-probabilities')
    if (Test-Path $calArtifact) { $distributionalArgs += @('--calibration-artifact', $calArtifact) }
    try {
      & $VenvPython -m ncaab_model.cli bankroll-optimize @distributionalArgs
    }
    catch {
      Write-Warning "bankroll-optimize distributional failed: $($_)"
    }

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

  # Newly produced aligned and stake artifacts
  $alignCsv = Join-Path $OutDir ("align_period_" + $todayIso + ".csv")
  if (Test-Path $alignCsv) { $toStage += $alignCsv }
  $alignEdges = Join-Path $OutDir ("align_period_" + $todayIso + "_edges.csv")
  if (Test-Path $alignEdges) { $toStage += $alignEdges }
  $stakeBase = Join-Path $OutDir 'stake_sheet_today.csv'
  if (Test-Path $stakeBase) { $toStage += $stakeBase }
  $stakeCal = Join-Path $OutDir 'stake_sheet_today_cal.csv'
  if (Test-Path $stakeCal) { $toStage += $stakeCal }

      # Allowlist per-date odds snapshots so historical odds persist on Render
      $oddsPrev = Join-Path $OutDir ("odds_history/odds_" + $prevDate + ".csv")
      if (Test-Path $oddsPrev) { $toStage += $oddsPrev }
      $oddsTodayHist = Join-Path $OutDir ("odds_history/odds_" + $todayIso + ".csv")
      if (Test-Path $oddsTodayHist) { $toStage += $oddsTodayHist }

      if ($toStage.Count -gt 0) {
        foreach ($p in $toStage) { git add $p }
        $msg = if ($GitCommitMessage) { $GitCommitMessage } else { "chore(data): update results and odds for $prevDate (today $todayIso)" }
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
}
catch {
  Write-Error $_
  exit 1
}
finally {
  Stop-Transcript | Out-Null
}
