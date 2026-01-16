#Requires -Version 5.1
<#
.SYNOPSIS
  Back-compat wrapper for task registration.

.DESCRIPTION
  The scheduled task registration logic was moved to scripts/register_daily_update_tasks.ps1.
  This file remains to keep older command lines working.
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

$script = Join-Path $PSScriptRoot 'register_daily_update_tasks.ps1'
& $script @PSBoundParameters
exit $LASTEXITCODE


