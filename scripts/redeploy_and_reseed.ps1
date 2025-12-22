param(
  [string]$Date = (Get-Date).ToString('yyyy-MM-dd'),
  [int]$DelaySeconds = 90,
  [string]$BaseUrl = 'https://ncaab.onrender.com'
)

Write-Host "[Redeploy] Date=$Date BaseUrl=$BaseUrl" -ForegroundColor Cyan

function Get-DeployHookUrl {
  try {
    if ($env:RENDER_DEPLOY_HOOK_URL -and -not [string]::IsNullOrWhiteSpace($env:RENDER_DEPLOY_HOOK_URL)) {
      return $env:RENDER_DEPLOY_HOOK_URL
    }
    $repoRoot = (Resolve-Path "$PSScriptRoot/..").Path
    $envPath = Join-Path $repoRoot '.env'
    if (Test-Path -LiteralPath $envPath) {
      $lines = Get-Content -LiteralPath $envPath
      foreach ($line in $lines) {
        if ($line -match '^\s*RENDER_DEPLOY_HOOK_URL\s*=\s*(.+)\s*$') {
          $val = $Matches[1].Trim()
          if (-not [string]::IsNullOrWhiteSpace($val)) { return $val }
        }
      }
    }
    $txtPath = Join-Path $repoRoot 'scripts/deploy_hook_url.txt'
    if (Test-Path -LiteralPath $txtPath) {
      $txt = Get-Content -LiteralPath $txtPath -TotalCount 1
      if ($txt -and -not [string]::IsNullOrWhiteSpace($txt)) { return $txt.Trim() }
    }
  } catch {}
  return $null
}

$hookUrl = Get-DeployHookUrl
if ([string]::IsNullOrWhiteSpace($hookUrl)) {
  Write-Host "[Error] Missing deploy hook URL. Run scripts/set_render_deploy_hook.ps1 -Url <URL> first." -ForegroundColor Red
  exit 1
}

Write-Host "[Redeploy] Triggering deploy via hook" -ForegroundColor Cyan
try {
  $resp = Invoke-RestMethod -Uri $hookUrl -Method Post
  Write-Host "[OK] Hook responded." -ForegroundColor Green
} catch {
  Write-Host "[Error] Deploy hook failed: $($_.Exception.Message)" -ForegroundColor Red
  exit 1
}

Write-Host "[Redeploy] Waiting $DelaySeconds seconds for deploy..." -ForegroundColor Cyan
Start-Sleep -Seconds $DelaySeconds

Write-Host "[Redeploy] Re-seeding artifacts for $Date" -ForegroundColor Cyan
& "$PSScriptRoot/upload_artifacts_to_render.ps1" -Date $Date -Verify

Write-Host "[Redeploy] Done." -ForegroundColor Cyan
