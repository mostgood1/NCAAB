param(
    [Parameter(Mandatory=$false)][string]$SummaryJson = "outputs/backtest_summary.json",
    [Parameter(Mandatory=$false)][string]$WebhookUrl = $env:NCAAB_WEBHOOK_URL,
    [Parameter(Mandatory=$false)][string]$Title = "NCAAB Backtest Summary"
)
$ErrorActionPreference = 'Stop'
if (-not $WebhookUrl) { Write-Error "NCAAB_WEBHOOK_URL not set"; exit 1 }
if (-not (Test-Path $SummaryJson)) { Write-Error "Summary JSON not found: $SummaryJson"; exit 1 }
try {
  $j = Get-Content $SummaryJson -Raw | ConvertFrom-Json
  $bets = [int]$j.total_bets
  $hit = if ($j.hit_rate) { [math]::Round(100*[double]$j.hit_rate,1) } else { $null }
  $pnl = if ($j.total_pnl) { [math]::Round([double]$j.total_pnl,2) } else { $null }
  $mae = if ($j.mae_total) { [math]::Round([double]$j.mae_total,2) } else { $null }
  $bA = if ($j.brier_ats) { [math]::Round([double]$j.brier_ats,3) } else { $null }
  $bO = if ($j.brier_ou) { [math]::Round([double]$j.brier_ou,3) } else { $null }
  $aA = if ($j.auc_ats) { [math]::Round([double]$j.auc_ats,3) } else { $null }
  $aO = if ($j.auc_ou) { [math]::Round([double]$j.auc_ou,3) } else { $null }
  $crps = if ($j.crps_total) { [math]::Round([double]$j.crps_total,3) } else { $null }
  $lines = @()
  $lines += "Range: $($j.start) → $($j.end)"
  $lines += "Bets: $bets | Hit: $hit% | PnL: $pnl"
  if ($mae) { $lines += "MAE Total: $mae" }
  if ($bA) { $lines += "Brier ATS: $bA" }
  if ($bO) { $lines += "Brier OU: $bO" }
  if ($aA) { $lines += "AUC ATS: $aA" }
  if ($aO) { $lines += "AUC OU: $aO" }
  if ($crps) { $lines += "CRPS Total: $crps" }
  $text = "**$Title**`n" + ($lines -join "`n")
  $payload = @{ text = $text } | ConvertTo-Json -Depth 3
  # Try Slack first (simple webhook), fallback to generic JSON post
  try {
    Invoke-RestMethod -Uri $WebhookUrl -Method Post -ContentType 'application/json' -Body $payload | Out-Null
  } catch {
    # Teams often expects a different shape; send card-lite
    $teams = @{ "@type" = "MessageCard"; "@context" = "https://schema.org/extensions"; "summary" = $Title; "text" = $text } | ConvertTo-Json -Depth 5
    Invoke-RestMethod -Uri $WebhookUrl -Method Post -ContentType 'application/json' -Body $teams | Out-Null
  }
  Write-Host "Webhook notification sent"
} catch {
  Write-Error $_
  exit 1
}
