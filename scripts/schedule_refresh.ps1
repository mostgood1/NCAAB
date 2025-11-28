param(
    [string]$Date = (Get-Date -Format 'yyyy-MM-dd')
)

# Runs ESPN fetch -> patch TBD -> build ESPN subset -> parity probe -> regenerate enriched
# Requires project venv at .\.venv and Python scripts in src\data

$ErrorActionPreference = 'Stop'

function Invoke-VenvPython {
    param([string[]]$ArgList)
    $py = Join-Path $PSScriptRoot '..' | Join-Path -ChildPath '.venv\Scripts\python.exe'
    if (-not (Test-Path $py)) { throw ".venv Python not found at $py" }
    & $py @ArgList
}

Write-Host "[schedule_refresh] Using date: $Date"

try {
    # 1) Fetch latest ESPN scoreboard cache
    Invoke-VenvPython -ArgList @('src/data/fetch_espn_scoreboard.py', '--date', $Date)
}
catch {
    Write-Warning "Fetch ESPN failed: $_"
}

try {
    # 2) Patch TBD teams in outputs/games_curr.csv when ESPN participants exist
    Invoke-VenvPython -ArgList @('src/data/patch_games_from_espn_cache.py', '--date', $Date)
}
catch {
    Write-Warning "Patch TBD failed: $_"
}

try {
    # 3) Build ESPN subset as intersection with local schedule
    Invoke-VenvPython -ArgList @('src/data/espn_subset_build.py', '--date', $Date)
}
catch {
    Write-Warning "Subset build failed: $_"
}

try {
    # 4) Run parity probe
    Invoke-VenvPython -ArgList @('src/data/schedule_alignment_probe.py', '--date', $Date)
}
catch {
    Write-Warning "Parity probe failed: $_"
}

try {
    # 5) Regenerate enriched artifacts via Flask app
    Invoke-VenvPython -ArgList @('src/data/regenerate_enriched_for_date.py', '--date', $Date)
}
catch {
    Write-Warning "Regenerate enriched failed: $_"
}

# 6) Print short summary from probe and pipeline outputs if available
try {
    $root = Split-Path $PSScriptRoot -Parent
    $probe = Join-Path $root ("outputs/schedule_alignment_probe_" + $Date + ".json")
    if (Test-Path $probe) {
        $p = Get-Content -Raw -Path $probe | ConvertFrom-Json
        Write-Host "[schedule_refresh] Parity OK: $($p.parity_ok) Placeholders: $($p.placeholder_count) Missing: $($p.missing_ids_count) LocalExtra: $($p.local_extra_ids_count)"
    } else {
        Write-Host "[schedule_refresh] Probe file not found: $probe"
    }
}
catch {
    Write-Warning "Summary read failed: $_"
}

Write-Host "[schedule_refresh] Done."
