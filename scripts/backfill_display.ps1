#Requires -Version 5.1
param(
  [int]$Days = 30
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$out = Join-Path $repo 'outputs'
New-Item -ItemType Directory -Path $out -Force | Out-Null

function Ensure-File($path, $headers) {
  if (-not (Test-Path $path)) {
    $dir = Split-Path -Parent $path
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    $csv = ($headers -join ',')
    Set-Content -Path $path -Value $csv -Encoding UTF8
    Write-Host "[created] $path" -ForegroundColor Green
  } else {
    Write-Host "[exists]  $path" -ForegroundColor DarkGray
  }
}

$displayHeaders = @(
  'game_id','home_team','away_team','date','start_time','commence_time','market_total',
  'pred_total','pred_total_basis','pred_margin','pred_margin_basis','edge','edge_closing',
  'closing_total','closing_spread_home','p_over','p_home_cover'
)

$enrichedHeaders = @(
  'game_id','home_team','away_team','date','start_time','commence_time','venue',
  'market_total','closing_total','spread_home','closing_spread_home',
  'pred_total','pred_total_basis','pred_margin','pred_margin_basis',
  'p_over','p_home_cover','start_dt_utc'
)

for ($i=0; $i -lt $Days; $i++) {
  $d = (Get-Date).AddDays(-$i).ToString('yyyy-MM-dd')
  $dispPath = Join-Path $out ("predictions_display_" + $d + ".csv")
  $enrPath  = Join-Path $out ("predictions_unified_enriched_" + $d + ".csv")
  Write-Host "\n== Backfill check for $d ==" -ForegroundColor Cyan
  Ensure-File -path $dispPath -headers $displayHeaders
  Ensure-File -path $enrPath  -headers $enrichedHeaders
}

Write-Host "\nBackfill placeholders ready. Replace with real artifacts as available." -ForegroundColor Yellow