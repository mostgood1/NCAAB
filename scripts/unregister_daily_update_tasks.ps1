#Requires -Version 5.1
<##
.SYNOPSIS
  Unregister Windows Task Scheduler tasks related to this NCAAB repo.

.DESCRIPTION
  Finds tasks that appear to run this repo's daily pipeline (e.g. scripts/auto_daily_update.ps1)
  and unregisters them.

  By default it matches either:
  - TaskName contains 'NCAAB'
  - Any task action references auto_daily_update.ps1 or daily_update.ps1 AND the task's
    working directory or arguments contain the repo root.

  Use -TaskName to explicitly remove named tasks.
  Use -WhatIf to preview removals.

.EXAMPLE
  powershell.exe -ExecutionPolicy Bypass -File scripts/unregister_daily_update_tasks.ps1 -WhatIf

.EXAMPLE
  powershell.exe -ExecutionPolicy Bypass -File scripts/unregister_daily_update_tasks.ps1

.EXAMPLE
  powershell.exe -ExecutionPolicy Bypass -File scripts/unregister_daily_update_tasks.ps1 -TaskName "NCAAB Daily Update (8AM Central)","NCAAB Pretip Refresh (4-30PM Central)"
##>

[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'High')]
param(
  [Parameter(Mandatory = $false)]
  [string]$RepoRoot,

  [Parameter(Mandatory = $false)]
  [string[]]$TaskName
)

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
  $scriptDir = Split-Path -Parent $PSCommandPath
  $RepoRoot = Split-Path -Parent $scriptDir
}

function Test-TaskMatchesRepo {
  param(
    [Parameter(Mandatory = $true)]
    $Task,

    [Parameter(Mandatory = $true)]
    [string]$RepoRoot
  )

  if ($Task.TaskName -match 'NCAAB') {
    return $true
  }

  foreach ($action in @($Task.Actions)) {
    $args = [string]$action.Arguments
    $wd = [string]$action.WorkingDirectory

    $mentionsRunner = ($args -match 'auto_daily_update\.ps1') -or ($args -match 'daily_update\.ps1')
    if (-not $mentionsRunner) { continue }

    $mentionsRepo = ($args -like "*$RepoRoot*") -or ($wd -like "*$RepoRoot*")
    if ($mentionsRepo) { return $true }
  }

  return $false
}

$allTasks = @()
try {
  $allTasks = Get-ScheduledTask -ErrorAction Stop
} catch {
  Write-Error "Failed to enumerate scheduled tasks. Try running PowerShell as Administrator. Error: $($_.Exception.Message)"
  exit 1
}

$candidates = @()
if ($TaskName -and $TaskName.Count -gt 0) {
  $nameSet = @{}
  foreach ($n in $TaskName) { $nameSet[$n] = $true }
  $candidates = $allTasks | Where-Object { $nameSet.ContainsKey($_.TaskName) }
} else {
  $candidates = $allTasks | Where-Object { Test-TaskMatchesRepo -Task $_ -RepoRoot $RepoRoot }
}

if (-not $candidates -or $candidates.Count -eq 0) {
  Write-Host "No matching scheduled tasks found." -ForegroundColor Yellow
  exit 0
}

Write-Host "Found $($candidates.Count) task(s) to unregister:" -ForegroundColor Cyan
$candidates | ForEach-Object { Write-Host ("- {0}{1}" -f $_.TaskPath, $_.TaskName) }

foreach ($t in $candidates) {
  $display = "$($t.TaskPath)$($t.TaskName)"
  if ($PSCmdlet.ShouldProcess($display, 'Unregister-ScheduledTask')) {
    try {
      Unregister-ScheduledTask -TaskName $t.TaskName -TaskPath $t.TaskPath -Confirm:$false -ErrorAction Stop | Out-Null
      Write-Host "Unregistered: $display" -ForegroundColor Green
    } catch {
      Write-Warning "Failed to unregister: $display ($($_.Exception.Message))"
    }
  }
}
