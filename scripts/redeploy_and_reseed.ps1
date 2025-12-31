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
    if ($env:RENDER_CODE_DEPLOY_HOOK_URL -and -not [string]::IsNullOrWhiteSpace($env:RENDER_CODE_DEPLOY_HOOK_URL)) {
      # Prefer explicit code deploy hook if provided
      return $env:RENDER_CODE_DEPLOY_HOOK_URL
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
        if ($line -match '^\s*RENDER_CODE_DEPLOY_HOOK_URL\s*=\s*(.+)\s*$') {
          $val2 = $Matches[1].Trim()
          if (-not [string]::IsNullOrWhiteSpace($val2)) { return $val2 }
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
# Capture baseline app version before triggering
$baselineSha = $null
$baselineBuildTime = $null
try {
  $ver0 = Invoke-RestMethod -Uri "$BaseUrl/api/version" -Method Get
  if ($ver0) { $baselineSha = $ver0.app_sha; $baselineBuildTime = $ver0.build_time_utc }
  Write-Host ("[Check] Baseline version: sha={0} build_time={1}" -f $baselineSha, $baselineBuildTime) -ForegroundColor White
} catch { Write-Host "[Warn] Baseline version check failed: $($_.Exception.Message)" -ForegroundColor Yellow }
try {
  $resp = Invoke-RestMethod -Uri $hookUrl -Method Post
  Write-Host "[OK] Hook responded." -ForegroundColor Green
} catch {
  Write-Host "[Error] Deploy hook failed: $($_.Exception.Message)" -ForegroundColor Red
  exit 1
}

Write-Host "[Redeploy] Polling version up to $DelaySeconds seconds..." -ForegroundColor Cyan
$deadline = (Get-Date).AddSeconds($DelaySeconds)
$changed = $false
while ((Get-Date) -lt $deadline) {
  Start-Sleep -Milliseconds 3000
  try {
    $ver = Invoke-RestMethod -Uri "$BaseUrl/api/version" -Method Get
    if ($ver) {
      $sha = $ver.app_sha
      $bt = $ver.build_time_utc
      Write-Host ("[Poll] Version: sha={0} build_time={1}" -f $sha, $bt) -ForegroundColor Gray
      if ($baselineSha -and $sha -and $sha -ne $baselineSha) { $changed = $true; break }
      if (-not $baselineBuildTime -and $bt) { $changed = $true; break }
    }
  } catch { Write-Host "[Warn] Version poll failed: $($_.Exception.Message)" -ForegroundColor Yellow }
}
if ($changed) { Write-Host "[OK] Detected new deployment version." -ForegroundColor Green }
else { Write-Host "[Warn] Version unchanged after polling; reseeding anyway." -ForegroundColor Yellow }

Write-Host "[Redeploy] Re-seeding artifacts for $Date" -ForegroundColor Cyan
try {
  & "$PSScriptRoot/upload_artifacts_to_render.ps1" -Date $Date -BaseUrl $BaseUrl | Out-Null
  Write-Host "[OK] Reseed completed." -ForegroundColor Green
} catch {
  Write-Host "[Error] Reseed failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "[Redeploy] Done." -ForegroundColor Cyan
