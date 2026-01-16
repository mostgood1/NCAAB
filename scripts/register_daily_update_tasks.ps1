#Requires -Version 5.1
<#
.SYNOPSIS
  Register Windows Task Scheduler tasks to run the NCAAB daily pipeline.

.DESCRIPTION
  Creates/updates a scheduled task that runs scripts/auto_daily_update.ps1.
  The underlying scripts/daily_update.ps1 pipeline will commit+push outputs to git
  and upload artifacts to Render as part of its normal flow.

  Notes:
  - Task runs under the current user context (InteractiveToken).
  - Ensure git auth is configured so git push works without prompts.
  - Trigger time is interpreted in your machine's local timezone.

.EXAMPLE
  powershell.exe -ExecutionPolicy Bypass -File scripts/register_daily_update_tasks.ps1 -At "08:00" -Mode Morning -TaskName "NCAAB Daily Update (8AM Central)"

.EXAMPLE
  powershell.exe -ExecutionPolicy Bypass -File scripts/register_daily_update_tasks.ps1 -At "16:30" -Mode Pretip -TaskName "NCAAB Pretip Refresh (4:30PM Central)"
#>

[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [ValidatePattern('^([01][0-9]|2[0-3]):[0-5][0-9]$')]
  [string]$At,

  [Parameter(Mandatory = $true)]
  [ValidateSet('Morning', 'Pretip', 'Custom')]
  [string]$Mode,

  [Parameter(Mandatory = $true)]
  [string]$TaskName
)

if ($TaskName.IndexOf(':') -ge 0) {
  throw "TaskName cannot contain ':' (Task Scheduler restriction). Use e.g. '4-30PM' instead of '4:30PM'."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $repoRoot 'scripts\auto_daily_update.ps1'
if (-not (Test-Path -LiteralPath $runner)) {
  throw "Runner script not found at: $runner"
}

# Build action (robust quoting)
$runnerQuoted = '"' + $runner + '"'
$actionArgs = "-NoProfile -ExecutionPolicy Bypass -File $runnerQuoted -Mode $Mode"
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $actionArgs -WorkingDirectory $repoRoot

# Daily trigger at local time
$parts = $At.Split(':')
$hour = [int]$parts[0]
$minute = [int]$parts[1]
$startAt = Get-Date -Hour $hour -Minute $minute -Second 0
$trigger = New-ScheduledTaskTrigger -Daily -At $startAt

# Settings
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 6)

# Run under current user (no password prompt); will only run when user is logged in
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited

try {
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue | Out-Null
} catch {}

try {
  Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description "Run NCAAB daily pipeline ($Mode) and push outputs to git" -ErrorAction Stop | Out-Null
} catch {
  Write-Error $_
  exit 1
}

Write-Host "Registered scheduled task '$TaskName' at $At (Mode=$Mode)." -ForegroundColor Green
Write-Host "To run immediately: Start-ScheduledTask -TaskName `"$TaskName`"" -ForegroundColor Gray
