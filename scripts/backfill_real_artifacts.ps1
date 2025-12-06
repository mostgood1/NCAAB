param(
    [int]$Days = 30,
    [string]$PythonExe = ".venv/Scripts/python.exe",
    [switch]$SkipQuantile,
    [switch]$SkipMeta,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-Info($msg){
    Write-Host $msg -ForegroundColor Cyan
}
function Write-Warn($msg){
    Write-Host $msg -ForegroundColor Yellow
}
function Write-Ok($msg){
    Write-Host $msg -ForegroundColor Green
}
function Write-Err($msg){
    Write-Host $msg -ForegroundColor Red
}

$OUT = Join-Path $PSScriptRoot "..\outputs" | Resolve-Path
$ROOT = Resolve-Path (Join-Path $PSScriptRoot "..")

Write-Info "== Real artifact backfill start (Days=$Days) =="

# Verify Python exe
$py = Join-Path $ROOT $PythonExe
if (-not (Test-Path $py)) {
    $py = Resolve-Path $PythonExe -ErrorAction SilentlyContinue
}
if (-not $py) {
    Write-Err "Python executable not found: $PythonExe"
    exit 1
}
Write-Ok "Using Python: $py"

# Optional upfront quantile refresh to ensure residuals available
if (-not $SkipQuantile) {
    Write-Info "Refreshing quantile residual model & today's scoring"
    if (-not $DryRun) {
        # Invoke the PowerShell script directly instead of via Python
        powershell -File (Join-Path $ROOT "scripts\daily_quantile_scoring.ps1") | Out-Null
    }
}

# Optional upfront meta trainer refresh (recent window)
if (-not $SkipMeta) {
    Write-Info "Training meta probabilities (LGBM) over recent window"
    if (-not $DryRun) {
        & $py (Join-Path $ROOT "scripts\train_meta_probs_lgbm.py") --limit-days 45  | Out-Null
    }
}

# Iterate dates
for ($i = 0; $i -lt $Days; $i++) {
    $date = (Get-Date).AddDays(-$i).ToString('yyyy-MM-dd')
    $disp = Join-Path $OUT "predictions_display_$date.csv"
    $enr = Join-Path $OUT "predictions_unified_enriched_$date.csv"

    Write-Info "-- $date --"
    $needDisp = -not (Test-Path $disp)
    $needEnr = -not (Test-Path $enr)
    if (-not $needDisp -and -not $needEnr) {
        Write-Warn "[skip] Existing artifacts present"
        continue
    }

    # Run the daily pipeline pieces that produce enriched + display outputs
    # Assumptions: verify_today.py or an equivalent CLI can emit enriched; app.py writes enriched during index route for a selected date.
    # Strategy: call app index in headless mode with ?stable=1&date=<date> to force enrichment write; then generate display CSV via scripts/evaluate_scoring.py or a dedicated exporter if present.

    # Enriched: hit app index in a lightweight local run or invoke a script that writes enriched.
    try {
        Write-Info "Generating enriched artifact via app index headless"
        if (-not $DryRun) {
            # Use python to run app with environment to target date; app writes enriched_unified_enriched_<date>.csv
            $env:TARGET_DATE = $date
            $env:STABLE_DISPLAY = "1"
            # Use a short-lived run to trigger write. The app should not block; ensure it exits after producing outputs.
            & $py (Join-Path $ROOT "verify_today.py") | Out-Null
            $env:TARGET_DATE = $null
            $env:STABLE_DISPLAY = $null
        }
        if (Test-Path $enr) {
            Write-Ok "[ok] Enriched ready: $(Split-Path $enr -Leaf)"
        } else {
            Write-Warn "[warn] Enriched not found post-run; will rely on placeholder if necessary"
        }
    } catch {
        Write-Err "Error generating enriched: $_"
    }

    # Display: generate from enriched using existing display writer fallback in app or a dedicated script.
    try {
        Write-Info "Generating display artifact"
        if (-not $DryRun) {
            # Use evaluate_scoring to at least ensure scoring JSON exists; templates often consume display file.
            & $py (Join-Path $ROOT "scripts\evaluate_scoring.py") --window-days 1 --end-date $date | Out-Null
            # If a dedicated display writer exists, call it; otherwise app renders it when serving.
        }
        if (Test-Path $disp) {
            Write-Ok "[ok] Display ready: $(Split-Path $disp -Leaf)"
        } else {
            Write-Warn "[warn] Display not found post-run; consider running app to emit display for $date"
        }
    } catch {
        Write-Err "Error generating display: $_"
    }
}

Write-Ok "Backfill complete. Replace any remaining placeholders as needed."