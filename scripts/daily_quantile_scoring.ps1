Param(
  [int]$WindowDays = 45,
  [string]$PythonPath = ".venv/\Scripts/\python.exe"
)

Write-Host "Starting daily quantile + scoring refresh..." -ForegroundColor Cyan

# Ensure python path exists
if (!(Test-Path -LiteralPath $PythonPath)) {
  Write-Host "Python not found at '$PythonPath'." -ForegroundColor Red
  Write-Host "Please ensure the venv is created and path is correct." -ForegroundColor Yellow
  exit 1
}

$today = Get-Date -Format 'yyyy-MM-dd'

try {
  Write-Host "Training residual quantiles (window=$WindowDays days)..." -ForegroundColor Cyan
  & $PythonPath "scripts/train_quantiles.py" --window-days $WindowDays
  if ($LASTEXITCODE -ne 0) { throw "train_quantiles.py failed with exit code $LASTEXITCODE" }

  Write-Host "Evaluating scoring for $today..." -ForegroundColor Cyan
  & $PythonPath "scripts/evaluate_scoring.py" --date $today
  if ($LASTEXITCODE -ne 0) { throw "evaluate_scoring.py failed with exit code $LASTEXITCODE" }

  Write-Host "Done. Wrote 'outputs/quantile_model.json' and 'outputs/scoring_$today.json'." -ForegroundColor Green
}
catch {
  Write-Host "Daily refresh failed: $_" -ForegroundColor Red
  exit 1
}
