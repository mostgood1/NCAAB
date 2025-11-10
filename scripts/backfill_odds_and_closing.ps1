# Backfill odds history and rebuild strict last odds + heuristic closing lines, then join and validate coverage
param(
  [string]$Start = (Get-Date).AddDays(-30).ToString('yyyy-MM-dd'),
  [string]$End = (Get-Date).ToString('yyyy-MM-dd'),
  [string]$Region = 'us',
  [string]$Markets = 'h2h,spreads,totals,spreads_1st_half,totals_1st_half,spreads_2nd_half,totals_2nd_half'
)
$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $RepoRoot '.venv/Spcripts/python.exe'
if (-not (Test-Path $Py)) { $Py = Join-Path $RepoRoot '.venv/Scripts/python.exe' }
$Out = Join-Path $RepoRoot 'outputs'
$HistDir = Join-Path $Out 'odds_history'

Write-Host "Fetching odds history from $Start to $End..." -ForegroundColor Cyan
& $Py -m ncaab_model.cli fetch-odds-history --start $Start --end $End --region $Region --markets $Markets --out-dir $HistDir --mode current

Write-Host "Building strict last odds and closing lines..." -ForegroundColor Cyan
& $Py -m ncaab_model.cli make-last-odds --in-dir $HistDir --out (Join-Path $Out 'last_odds.csv') --tolerance-seconds 60
& $Py -m ncaab_model.cli make-closing-lines --in-dir $HistDir --out (Join-Path $Out 'closing_lines.csv')

# Choose a comprehensive games file for join (prefer fused history)
$gamesPath = Join-Path $Out 'games_hist_fused.csv'
if (-not (Test-Path $gamesPath)) { $gamesPath = Join-Path $Out 'games_all.csv' }
Write-Host "Joining last odds and closing lines to $gamesPath..." -ForegroundColor Cyan
& $Py -m ncaab_model.cli join-last-odds $gamesPath (Join-Path $Out 'last_odds.csv') --out (Join-Path $Out 'games_with_last.csv')
& $Py -m ncaab_model.cli join-closing $gamesPath (Join-Path $Out 'closing_lines.csv') --out (Join-Path $Out 'games_with_closing.csv')

Write-Host "Validating coverage (closing + last)..." -ForegroundColor Cyan
& $Py -m ncaab_model.cli validate-closing-coverage $gamesPath (Join-Path $Out 'games_with_closing.csv') --out (Join-Path $Out 'closing_coverage.csv') --verbose
& $Py -m ncaab_model.cli validate-last-coverage $gamesPath (Join-Path $Out 'games_with_last.csv') --out (Join-Path $Out 'last_coverage.csv') --out-books (Join-Path $Out 'last_coverage_books.csv') --verbose

Write-Host "Ingesting outputs (including last_odds)..." -ForegroundColor Cyan
& $Py -m ncaab_model.cli ingest-outputs

Write-Host "Done." -ForegroundColor Green
