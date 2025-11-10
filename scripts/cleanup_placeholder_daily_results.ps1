# Removes daily_results files that have zero scores and no predictions (to reduce confusion for upcoming slates)
param(
  [string]$Dir = "outputs/daily_results"
)
$dirPath = Join-Path (Get-Location) $Dir
if (-not (Test-Path $dirPath)) {
  Write-Host "Directory not found: $dirPath" -ForegroundColor Yellow
  exit 0
}
Get-ChildItem -Path $dirPath -Filter "results_*.csv" | ForEach-Object {
  try {
    $rows = Import-Csv $_.FullName
    if ($rows.Count -eq 0) { return }
    $hasPred = $false
    $anyScore = $false
    foreach ($r in $rows) {
      # Check for non-empty pred_total
      if ($r.pred_total -and $r.pred_total -ne '') { $hasPred = $true }
      # Check if any home+away score > 0
      $hs = 0; $as = 0
      [void][int]::TryParse(($r.home_score), [ref]$hs)
      [void][int]::TryParse(($r.away_score), [ref]$as)
      if (($hs + $as) -gt 0) { $anyScore = $true }
      if ($hasPred -or $anyScore) { break }
    }
    if (-not $hasPred -and -not $anyScore) {
      Write-Host "Removing placeholder results: $($_.Name)" -ForegroundColor Yellow
      Remove-Item $_.FullName -Force
    }
  } catch {
    Write-Warning "Failed to process $($_.FullName): $_"
  }
}
