param(
    [Parameter(Mandatory=$true)][string]$Url
)

Write-Host "[Config] Setting Render deploy hook URL" -ForegroundColor Cyan
try {
    # Set for current session
    $env:RENDER_DEPLOY_HOOK_URL = $Url
    # Persist for user environment
    [System.Environment]::SetEnvironmentVariable('RENDER_DEPLOY_HOOK_URL', $Url, 'User')
    # Update .env
    $repoRoot = (Resolve-Path "$PSScriptRoot/..").Path
    $envPath = Join-Path $repoRoot '.env'
    if (Test-Path -LiteralPath $envPath) {
      $lines = Get-Content -LiteralPath $envPath
      $updated = $false
      $outLines = @()
      foreach ($line in $lines) {
        if ($line -match '^\s*RENDER_DEPLOY_HOOK_URL\s*=') {
          $outLines += "RENDER_DEPLOY_HOOK_URL=$Url"
          $updated = $true
        } else {
          $outLines += $line
        }
      }
      if (-not $updated) { $outLines += "RENDER_DEPLOY_HOOK_URL=$Url" }
      Set-Content -LiteralPath $envPath -Value $outLines -Encoding UTF8
    } else {
      Set-Content -LiteralPath $envPath -Value @("RENDER_DEPLOY_HOOK_URL=$Url") -Encoding UTF8
    }
    # Write helper text file as backup
    $txtPath = Join-Path $repoRoot 'scripts/deploy_hook_url.txt'
    Set-Content -LiteralPath $txtPath -Value $Url -Encoding UTF8
    Write-Host "[OK] Deploy hook URL configured." -ForegroundColor Green
} catch {
    Write-Host "[Error] Failed to set deploy hook URL: $($_.Exception.Message)" -ForegroundColor Red
}
