param(
    [string]$Date = (Get-Date).ToString('yyyy-MM-dd'),
    [string]$BaseUrl = 'https://ncaab.onrender.com',
    [string]$OutputsDir = "$PSScriptRoot/../outputs",
    [switch]$TriggerRedeploy,
    [switch]$SlimSimOnly,
    [string]$DeployHookUrl = $env:RENDER_DEPLOY_HOOK_URL,
    [string]$CodeDeployHookUrl = $env:RENDER_CODE_DEPLOY_HOOK_URL,
    [int]$VersionPollSeconds = 240,
    [int]$VersionPollIntervalMs = 3000
)

function Write-Step {
    param([string]$Message)
    Write-Host "[Uploader] $Message" -ForegroundColor Cyan
}

function Upload-File {
    param(
        [string]$Uri,
        [string]$FilePath,
        [hashtable]$Query = @{}
    )
    if (-not (Test-Path -LiteralPath $FilePath)) {
        Write-Host "[Skip] Missing file: $FilePath" -ForegroundColor Yellow
        return $null
    }
    $q = if ($Query.Count -gt 0) { '?' + ((($Query.GetEnumerator() | ForEach-Object { "{0}={1}" -f $_.Key, $_.Value }) -join '&')) } else { '' }
    $target = "$Uri$q"
    Write-Step "POST $target with $(Split-Path -Leaf $FilePath)"
    try {
        try { Add-Type -AssemblyName System.Net.Http } catch {}
        $client = New-Object System.Net.Http.HttpClient
        $content = New-Object System.Net.Http.MultipartFormDataContent
        $fs = [System.IO.File]::OpenRead($FilePath)
        $fileContent = New-Object System.Net.Http.StreamContent($fs)
        $ext = ([System.IO.Path]::GetExtension($FilePath) | ForEach-Object { $_.ToLowerInvariant() })
        $ct = if ($ext -eq '.json') { 'application/json' } elseif ($ext -eq '.csv') { 'text/csv' } else { 'application/octet-stream' }
        $fileContent.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::Parse($ct)
        $fileName = [System.IO.Path]::GetFileName($FilePath)
        $content.Add($fileContent, 'file', $fileName)
        $resp = $client.PostAsync($target, $content).Result
        $text = $resp.Content.ReadAsStringAsync().Result
        try { $fs.Dispose() } catch {}
        try { $client.Dispose() } catch {}
        if (-not $resp.IsSuccessStatusCode) {
            # Optional endpoints may not exist on older deployments; treat as skip.
            $code = $null
            try { $code = [int]$resp.StatusCode } catch { $code = $null }
            if (($code -eq 404) -and ($Uri -match 'upload_sim_inputs_diagnostic|upload_sim_calibration|upload_sim_segments')) {
                Write-Host "[Skip] Endpoint not available yet: $Uri" -ForegroundColor Yellow
                return @{ status = 'skipped'; code = 404; uri = $Uri }
            }
            Write-Host "[Error] Upload HTTP $($resp.StatusCode): $text" -ForegroundColor Red
            return $null
        }
        # Try JSON parse, else return raw text
        try { return ($text | ConvertFrom-Json) } catch { return @{ status = 'ok'; raw = $text } }
    } catch {
        Write-Host "[Error] Upload failed: $($_.Exception.Message)" -ForegroundColor Red
        return $null
    }
}

# Resolve artifact paths
$picksPath     = Join-Path -Path $OutputsDir -ChildPath 'picks_raw.csv'
$picksAtsPath  = Join-Path -Path $OutputsDir -ChildPath ("picks/ats_picks_{0}.csv" -f $Date)
$edgesPath     = Join-Path -Path $OutputsDir -ChildPath ("align_period_{0}_edges.csv" -f $Date)
$displayPath   = Join-Path -Path $OutputsDir -ChildPath ("predictions_display_{0}.csv" -f $Date)
$enrichedPath  = Join-Path -Path $OutputsDir -ChildPath ("predictions_unified_enriched_{0}.csv" -f $Date)
$resultsPath   = Join-Path -Path $OutputsDir -ChildPath ("daily_results/results_{0}.csv" -f $Date)
$simQuantilesPath = Join-Path -Path $OutputsDir -ChildPath ("sim_quantiles_{0}.csv" -f $Date)
$simBlendPath     = Join-Path -Path $OutputsDir -ChildPath ("sim_blend_{0}.csv" -f $Date)
$simSegmentsPath  = Join-Path -Path $OutputsDir -ChildPath ("sim_segments_{0}.csv" -f $Date)
$simDiagPath      = Join-Path -Path $OutputsDir -ChildPath ("sim_inputs_diagnostic_{0}.json" -f $Date)
$simCalibPath     = Join-Path -Path $OutputsDir -ChildPath 'sim_calibration.json'
${needSimRetry} = $false

Write-Step "Using date=$Date, baseUrl=$BaseUrl"
Write-Step "Outputs dir: $OutputsDir"

function Get-DeployHookUrl {
    param()
    try {
        # Prefer explicit code deploy hook if provided
        if ($CodeDeployHookUrl -and -not [string]::IsNullOrWhiteSpace($CodeDeployHookUrl)) {
            return $CodeDeployHookUrl
        }
        if ($DeployHookUrl -and -not [string]::IsNullOrWhiteSpace($DeployHookUrl)) {
            return $DeployHookUrl
        }
        $repoRoot = (Resolve-Path "$PSScriptRoot/..").Path
        $envPath = Join-Path $repoRoot '.env'
        if (Test-Path -LiteralPath $envPath) {
            $lines = Get-Content -LiteralPath $envPath
            foreach ($line in $lines) {
                if ($line -match '^\s*RENDER_CODE_DEPLOY_HOOK_URL\s*=\s*(.+)\s*$') {
                    $val = $Matches[1].Trim()
                    if (-not [string]::IsNullOrWhiteSpace($val)) { return $val }
                }
                if ($line -match '^\s*RENDER_DEPLOY_HOOK_URL\s*=\s*(.+)\s*$') {
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

function Invoke-RedeployAndWait {
    param(
        [string]$HookUrl,
        [int]$PollSeconds,
        [int]$PollIntervalMs
    )
    if ([string]::IsNullOrWhiteSpace($HookUrl)) { return $false }
    $baselineSha = $null
    $baselineBuildTime = $null
    try {
        $ts0 = [int](Get-Date -UFormat %s)
        $ver0 = Invoke-RestMethod -Uri ("{0}/api/version?t={1}" -f $BaseUrl, $ts0) -Method Get
        if ($ver0) {
            $baselineSha = $ver0.app_sha
            $baselineBuildTime = $ver0.build_time_utc
        }
        Write-Host ("[Check] Baseline version: sha={0} build_time={1}" -f $baselineSha, $baselineBuildTime) -ForegroundColor White
    } catch {
        Write-Host "[Warn] Baseline version check failed; proceeding." -ForegroundColor Yellow
    }

    Write-Step "Triggering redeploy via deploy hook"
    try {
        $null = Invoke-RestMethod -Uri $HookUrl -Method Post
        Write-Host "[OK] Deploy hook accepted." -ForegroundColor Green
    } catch {
        Write-Host ("[Error] Deploy hook POST failed: {0}" -f $_.Exception.Message) -ForegroundColor Red
        return $false
    }

    $deadline = (Get-Date).AddSeconds([double]$PollSeconds)
    $startedAt = Get-Date
    $pollCount = 0
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds $PollIntervalMs
        $pollCount += 1
        try {
            $pollTs = [int](Get-Date -UFormat %s)
            $ver = Invoke-RestMethod -Uri ("{0}/api/version?t={1}" -f $BaseUrl, $pollTs) -Method Get
            if ($ver) {
                $sha = $ver.app_sha
                $bt = $ver.build_time_utc
                if (($pollCount % 5) -eq 0) {
                    $elapsed = [math]::Round(((Get-Date) - $startedAt).TotalSeconds)
                    $remaining = [math]::Max(0, [math]::Round(($deadline - (Get-Date)).TotalSeconds))
                    Write-Host ("[Poll] Version: sha={0} build_time={1} (elapsed={2}s remaining~{3}s)" -f $sha, $bt, $elapsed, $remaining) -ForegroundColor Gray
                }
                if ($baselineSha -and $sha -and $sha -ne $baselineSha) { return $true }
                if ($baselineBuildTime -and $bt -and $bt -ne $baselineBuildTime) { return $true }
                if (-not $baselineBuildTime -and $bt) { return $true }
            }
        } catch {
            Write-Host ("[Warn] Version poll failed: {0}" -f $_.Exception.Message) -ForegroundColor Yellow
        }
    }
    Write-Host ("[Warn] Gave up waiting for version change after ~{0}s; continuing." -f $PollSeconds) -ForegroundColor Yellow
    return $false
}

# If requested, redeploy FIRST (redeploy wipes ephemeral disk, so uploads must come after)
if ($TriggerRedeploy.IsPresent) {
    $hook = Get-DeployHookUrl
    if ([string]::IsNullOrWhiteSpace($hook)) {
        Write-Host "[Warn] TriggerRedeploy set but no deploy hook URL found (env/.env/scripts). Continuing without redeploy." -ForegroundColor Yellow
    } else {
        $changed = Invoke-RedeployAndWait -HookUrl $hook -PollSeconds $VersionPollSeconds -PollIntervalMs $VersionPollIntervalMs
        if ($changed) { Write-Host "[OK] Detected new deployment version; proceeding with uploads." -ForegroundColor Green }
        else { Write-Host "[Warn] Version unchanged after polling; deploy may still be in progress. Proceeding with uploads anyway." -ForegroundColor Yellow }
    }
}

function Get-CsvRowCount {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return 0 }
    try {
        $rows = Import-Csv -LiteralPath $Path -ErrorAction Stop
        return ($rows | Measure-Object).Count
    } catch {
        try {
            $lines = Get-Content -LiteralPath $Path -ErrorAction Stop
            if ($lines -and $lines.Count -gt 1) { return ($lines.Count - 1) } else { return 0 }
        } catch { return 0 }
    }
}

# Create a sanitized display CSV by removing placeholder rows (AWAY/HOME teams) and synthetic/odds ids
function Sanitize-DisplayCsv {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try {
        $rows = Import-Csv -LiteralPath $Path -ErrorAction Stop
        if (-not $rows) { return $null }
        $filtered = @()
        foreach ($r in $rows) {
            $gid = ("" + $r.game_id).Trim()
            $ht = ("" + $r.home_team).Trim().ToLower()
            $at = ("" + $r.away_team).Trim().ToLower()
            $isSynthetic = ($gid.StartsWith('synthetic:') -or $gid.StartsWith('odds:'))
            $isPlaceholder = ($ht -in @('home','away') -or $at -in @('home','away'))
            if (-not ($isSynthetic -or $isPlaceholder)) { $filtered += $r }
        }
        $tmp = [System.IO.Path]::Combine([System.IO.Path]::GetDirectoryName($Path), (".tmp_sanitized_{0}" -f [System.IO.Path]::GetFileName($Path)))
        if ($filtered.Count -gt 0) { $filtered | Export-Csv -LiteralPath $tmp -NoTypeInformation -Encoding UTF8 }
        else {
            # If sanitization removes all rows, fall back to original path to avoid empty uploads
            return $Path
        }
        return $tmp
    } catch {
        return $null
    }
}

# If local display mirrors market across all rows, build a model-first display by
# overriding pred_total/pred_total_basis from enriched (cal -> model -> blend -> seg).
function Build-ModelFirstDisplayFromEnriched {
    param(
        [string]$DisplayCsv,
        [string]$EnrichedCsv
    )
    if (-not (Test-Path -LiteralPath $DisplayCsv)) { return $null }
    if (-not (Test-Path -LiteralPath $EnrichedCsv)) { return $DisplayCsv }
    try {
        $disp = Import-Csv -LiteralPath $DisplayCsv -ErrorAction Stop
        if (-not $disp) { return $DisplayCsv }
        $rows = @()
        # Index enriched by game_id for quick lookup
        $enr = Import-Csv -LiteralPath $EnrichedCsv -ErrorAction Stop
        $emap = @{}
        foreach ($e in $enr) {
            $gid = ("" + $e.game_id).Trim()
            if (-not [string]::IsNullOrWhiteSpace($gid)) { $emap[$gid] = $e }
        }
        foreach ($r in $disp) {
            $gid = ("" + $r.game_id).Trim()
            $origPred = $r.pred_total
            $origBasis = $r.pred_total_basis
            if ($emap.ContainsKey($gid)) {
                $e = $emap[$gid]
                $choice = $null; $basis = $null
                $cals = @('pred_total_calibrated','pred_total_model','pred_total_blend','pred_total_seg')
                foreach ($c in $cals) {
                    if ($e.PSObject.Properties.Name -contains $c) {
                        $val = $e.$c
                        if ($null -ne $val -and -not [string]::IsNullOrWhiteSpace("" + $val)) {
                            $dv = $null; [void][double]::TryParse(("" + $val), [ref]$dv)
                            if ($dv -ne $null) { $choice = $dv; $basis = ($c -replace '^pred_total_',''); break }
                        }
                    }
                }
                if ($null -ne $choice) {
                    $r.pred_total = $choice
                    $r.pred_total_basis = $basis
                }
            }
            $rows += $r
        }
        $tmp = [System.IO.Path]::Combine([System.IO.Path]::GetDirectoryName($DisplayCsv), (".tmp_model_first_{0}" -f [System.IO.Path]::GetFileName($DisplayCsv)))
        $rows | Export-Csv -LiteralPath $tmp -NoTypeInformation -Encoding UTF8
        return $tmp
    } catch { return $DisplayCsv }
}

# Build model-first display using predictions_model(_calibrated)_<date>.csv if present.
function Build-ModelFirstDisplayFromModelPreds {
    param(
        [string]$DisplayCsv,
        [string]$OutputsDir,
        [string]$Date
    )
    if (-not (Test-Path -LiteralPath $DisplayCsv)) { return $null }
    $predModelPath = Join-Path -Path $OutputsDir -ChildPath ("predictions_model_{0}.csv" -f $Date)
    $predCalPath = Join-Path -Path $OutputsDir -ChildPath ("predictions_model_calibrated_{0}.csv" -f $Date)
    if (-not (Test-Path -LiteralPath $predModelPath) -and -not (Test-Path -LiteralPath $predCalPath)) { return $DisplayCsv }
    try {
        $disp = Import-Csv -LiteralPath $DisplayCsv -ErrorAction Stop
        if (-not $disp) { return $DisplayCsv }
        $model = $null
        $cal = $null
        if (Test-Path -LiteralPath $predModelPath) { $model = Import-Csv -LiteralPath $predModelPath -ErrorAction Stop }
        if (Test-Path -LiteralPath $predCalPath) { $cal = Import-Csv -LiteralPath $predCalPath -ErrorAction Stop }
        $mmap = @{}
        if ($model) {
            foreach ($m in $model) {
                $gid = ("" + $m.game_id).Trim(); if (-not [string]::IsNullOrWhiteSpace($gid)) { $mmap[$gid] = $m }
            }
        }
        $cmap = @{}
        if ($cal) {
            foreach ($c in $cal) {
                $gid = ("" + $c.game_id).Trim(); if (-not [string]::IsNullOrWhiteSpace($gid)) { $cmap[$gid] = $c }
            }
        }
        $rows = @()
        foreach ($r in $disp) {
            $gid = ("" + $r.game_id).Trim()
            $chosen = $null; $basis = $null
            if ($cmap.ContainsKey($gid)) {
                $cr = $cmap[$gid]
                $pv = $null; [void][double]::TryParse(("" + $cr.pred_total), [ref]$pv)
                if ($pv -ne $null) { $chosen = $pv; $basis = 'cal' }
            }
            if ($null -eq $chosen -and $mmap.ContainsKey($gid)) {
                $mr = $mmap[$gid]
                $pv2 = $null; [void][double]::TryParse(("" + $mr.pred_total), [ref]$pv2)
                if ($pv2 -ne $null) { $chosen = $pv2; $basis = 'model' }
            }
            if ($null -ne $chosen) {
                $r.pred_total = $chosen
                $r.pred_total_basis = $basis
            }
            $rows += $r
        }
        $tmp = [System.IO.Path]::Combine([System.IO.Path]::GetDirectoryName($DisplayCsv), (".tmp_modelpred_first_{0}" -f [System.IO.Path]::GetFileName($DisplayCsv)))
        $rows | Export-Csv -LiteralPath $tmp -NoTypeInformation -Encoding UTF8
        return $tmp
    } catch { return $DisplayCsv }
}

# Build a model-first enriched CSV by overriding pred_total and pred_total_basis
# with precedence: pred_total_calibrated -> pred_total_model -> pred_total_blend -> pred_total_seg.
function Build-ModelFirstEnriched {
    param(
        [string]$EnrichedCsv
    )
    if (-not (Test-Path -LiteralPath $EnrichedCsv)) { return $null }
    try {
        $rows = Import-Csv -LiteralPath $EnrichedCsv -ErrorAction Stop
        if (-not $rows) { return $null }
        $out = @()
        foreach ($r in $rows) {
            $choice = $null; $basis = $null
            $cals = @('pred_total_calibrated','pred_total_model','pred_total_blend','pred_total_seg')
            foreach ($c in $cals) {
                if ($r.PSObject.Properties.Name -contains $c) {
                    $val = $r.$c
                    if ($null -ne $val -and -not [string]::IsNullOrWhiteSpace("" + $val)) {
                        $dv = $null; [void][double]::TryParse(("" + $val), [ref]$dv)
                        if ($dv -ne $null) { $choice = $dv; $basis = ($c -replace '^pred_total_',''); break }
                    }
                }
            }
            if ($null -ne $choice) {
                $r.pred_total = $choice
                $r.pred_total_basis = $basis
            }
            $out += $r
        }
        $tmp = [System.IO.Path]::Combine([System.IO.Path]::GetDirectoryName($EnrichedCsv), (".tmp_modelfirst_{0}" -f [System.IO.Path]::GetFileName($EnrichedCsv)))
        $out | Export-Csv -LiteralPath $tmp -NoTypeInformation -Encoding UTF8
        return $tmp
    } catch { return $null }
}

# Upload in preferred order: picks_raw -> ATS picks(date) -> edges(date) -> display(date) -> enriched(date)
$picksRows = Get-CsvRowCount -Path $picksPath
if ($picksRows -gt 0) {
    $u1 = Upload-File -Uri "$BaseUrl/api/upload_picks_raw" -FilePath $picksPath
    if ($u1) { Write-Host "[OK] picks_raw uploaded: $($u1.path) rows=$($u1.rows)" -ForegroundColor Green }
} else {
    Write-Host "[Skip] picks_raw.csv has 0 rows; preserving remote non-empty file." -ForegroundColor Yellow
}

# Upload ATS picks if present
$picksAtsRows = Get-CsvRowCount -Path $picksAtsPath
if ($picksAtsRows -gt 0) {
    $u1b = Upload-File -Uri "$BaseUrl/api/upload_ats_picks" -FilePath $picksAtsPath -Query @{ date = $Date }
    if ($u1b) { Write-Host "[OK] ats_picks uploaded: rows=$($u1b.rows)" -ForegroundColor Green }
} else {
    Write-Host "[Skip] ats_picks missing or empty for $Date" -ForegroundColor Yellow
}

$u2 = Upload-File -Uri "$BaseUrl/api/upload_align_edges" -FilePath $edgesPath -Query @{ date = $Date }
if ($u2) {
    $rv = if ($u2.rows_verified) { $u2.rows_verified } elseif ($u2.rows) { $u2.rows } else { $null }
    $ru = if ($u2.rows_uploaded) { $u2.rows_uploaded } else { $null }
    $sha = if ($u2.sha) { $u2.sha } else { $null }
    $shaSuffix = if ($sha) { " sha=$sha" } else { "" }
    Write-Host ("[OK] edges uploaded: rows_uploaded={0} rows_verified={1}{2}" -f $ru, $rv, $shaSuffix) -ForegroundColor Green
}

$sanitizedDisplayPath = Sanitize-DisplayCsv -Path $displayPath
if ($sanitizedDisplayPath) {
    Write-Step ("Sanitized display for upload: {0}" -f (Split-Path -Leaf $sanitizedDisplayPath))
}
$displayToUpload = $null
if ($sanitizedDisplayPath -and -not [string]::IsNullOrWhiteSpace($sanitizedDisplayPath)) { $displayToUpload = $sanitizedDisplayPath } else { $displayToUpload = $displayPath }
# Detect if display mirrors market; if so, rebuild from enriched with model-first precedence
try {
    $drows = Import-Csv -LiteralPath $displayToUpload -ErrorAction Stop
    if ($drows) {
        $eq = 0; $n = 0
        foreach ($r in $drows) {
            $pt = $null; $mt = $null
            [void][double]::TryParse(("" + $r.pred_total), [ref]$pt)
            [void][double]::TryParse(("" + $r.market_total), [ref]$mt)
            if ($pt -ne $null -and $mt -ne $null) {
                $n += 1
                if ([Math]::Abs($pt - $mt) -lt 1e-9) { $eq += 1 }
            }
        }
        if ($n -gt 0 -and $eq -eq $n) {
            Write-Step "Display mirrors market across all rows; rebuilding model-first display"
            # Prefer model predictions files; fallback to enriched if needed
            $rebuilt = Build-ModelFirstDisplayFromModelPreds -DisplayCsv $displayToUpload -OutputsDir $OutputsDir -Date $Date
            if (-not $rebuilt -or -not (Test-Path -LiteralPath $rebuilt)) {
                if (Test-Path -LiteralPath $enrichedPath) {
                    $rebuilt = Build-ModelFirstDisplayFromEnriched -DisplayCsv $displayToUpload -EnrichedCsv $enrichedPath
                }
            }
            if ($rebuilt -and (Test-Path -LiteralPath $rebuilt)) { $displayToUpload = $rebuilt }
        }
    }
} catch {}

# If local display has fewer rows than edges, regenerate display from edges to publish all game cards
try {
    $edgesRowsLocal = Get-CsvRowCount -Path $edgesPath
    $localDispRowsPre = Get-CsvRowCount -Path $displayToUpload
    if ($edgesRowsLocal -gt 0 -and ($localDispRowsPre -lt $edgesRowsLocal)) {
        Write-Step ("Local display has {0} rows < edges {1}; rebuilding from edges" -f $localDispRowsPre, $edgesRowsLocal)
        $repoRoot = (Resolve-Path "$PSScriptRoot/..").Path
        $pyExe = Join-Path $repoRoot ".venv/Scripts/python.exe"
        $genScript = Join-Path $PSScriptRoot "generate_display_from_edges.py"
        if (Test-Path -LiteralPath $pyExe) {
            & $pyExe $genScript $Date | Out-Null
        } else {
            python $genScript $Date | Out-Null
        }
        $regeneratedPath = $displayPath
        $san2 = Sanitize-DisplayCsv -Path $regeneratedPath
        $usePath = if ($san2 -and (Get-CsvRowCount -Path $san2) -gt 0) { $san2 } else { $regeneratedPath }
        $displayToUpload = $usePath
    }
} catch {}
if (-not $SlimSimOnly.IsPresent) {
    $u3 = Upload-File -Uri "$BaseUrl/api/upload_predictions_display" -FilePath $displayToUpload -Query @{ date = $Date }
    if ($u3) {
        $rv = if ($u3.rows_verified) { $u3.rows_verified } elseif ($u3.rows) { $u3.rows } else { $null }
        $ru = if ($u3.rows_uploaded) { $u3.rows_uploaded } else { $null }
        $sha = if ($u3.sha) { $u3.sha } else { $null }
        $shaSuffix = if ($sha) { " sha=$sha" } else { "" }
        Write-Host ("[OK] display uploaded: rows_uploaded={0} rows_verified={1}{2}" -f $ru, $rv, $shaSuffix) -ForegroundColor Green
    }
} else {
    Write-Host "[Skip] SlimSimOnly: skipping predictions_display upload" -ForegroundColor Yellow
}

# Upload enriched predictions snapshot for recommendations parity
# If enriched shows basis entirely NaN or equals market for all rows, rebuild to model-first
$enrichedToUpload = $enrichedPath
try {
    if (Test-Path -LiteralPath $enrichedPath) {
        $erows = Import-Csv -LiteralPath $enrichedPath -ErrorAction Stop
        if ($erows) {
            $n = 0; $eq = 0; $nanBasis = 0
            foreach ($r in $erows) {
                $pt = $null; $mt = $null
                [void][double]::TryParse(("" + $r.pred_total), [ref]$pt)
                [void][double]::TryParse(("" + $r.market_total), [ref]$mt)
                $b = ("" + $r.pred_total_basis)
                if (-not [string]::IsNullOrWhiteSpace($b)) { }
                else { $nanBasis += 1 }
                if ($pt -ne $null -and $mt -ne $null) {
                    $n += 1
                    if ([Math]::Abs($pt - $mt) -lt 1e-9) { $eq += 1 }
                }
            }
            if (($n -gt 0 -and $eq -eq $n) -or ($nanBasis -ge $erows.Count)) {
                Write-Step "Enriched appears synthetic or basis NaN; rebuilding model-first enriched"
                $rebuiltEnr = Build-ModelFirstEnriched -EnrichedCsv $enrichedPath
                if ($rebuiltEnr -and (Test-Path -LiteralPath $rebuiltEnr)) { $enrichedToUpload = $rebuiltEnr }
            }
        }
    }
} catch {}

if (-not $SlimSimOnly.IsPresent) {
    $u3b = Upload-File -Uri "$BaseUrl/api/upload_predictions_enriched" -FilePath $enrichedToUpload -Query @{ date = $Date }
    if ($u3b) {
        $rv = if ($u3b.rows_verified) { $u3b.rows_verified } elseif ($u3b.rows) { $u3b.rows } else { $null }
        $ru = if ($u3b.rows_uploaded) { $u3b.rows_uploaded } else { $null }
        $sha = if ($u3b.sha) { $u3b.sha } else { $null }
        $shaSuffix = if ($sha) { " sha=$sha" } else { "" }
        Write-Host ("[OK] enriched uploaded: rows_uploaded={0} rows_verified={1}{2}" -f $ru, $rv, $shaSuffix) -ForegroundColor Green
    }
} else {
    Write-Host "[Skip] SlimSimOnly: skipping predictions_enriched upload" -ForegroundColor Yellow
}

# Upload simulation artifacts if present
if (Test-Path -LiteralPath $simQuantilesPath) {
    $simQRows = Get-CsvRowCount -Path $simQuantilesPath
    if ($simQRows -gt 0) {
        $uSimQ = Upload-File -Uri "$BaseUrl/api/upload_sim_quantiles" -FilePath $simQuantilesPath -Query @{ date = $Date }
        if ($uSimQ) { Write-Host "[OK] sim_quantiles uploaded: rows=$($uSimQ.rows)" -ForegroundColor Green }
        else { ${needSimRetry} = $true }
    } else {
        Write-Host "[Skip] sim_quantiles_$Date.csv empty (header-only); not uploading" -ForegroundColor Yellow
    }
} else {
    Write-Host "[Skip] sim_quantiles_$Date.csv missing" -ForegroundColor Yellow
}

# Upload simulation 5-min segments if present
if (Test-Path -LiteralPath $simSegmentsPath) {
    $segRows = Get-CsvRowCount -Path $simSegmentsPath
    if ($segRows -gt 0) {
        $uSimS = Upload-File -Uri "$BaseUrl/api/upload_sim_segments" -FilePath $simSegmentsPath -Query @{ date = $Date }
        if ($uSimS) { Write-Host "[OK] sim_segments uploaded: rows=$($uSimS.rows)" -ForegroundColor Green }
        else { ${needSimRetry} = $true }
    } else {
        Write-Host "[Skip] sim_segments_$Date.csv empty (header-only); not uploading" -ForegroundColor Yellow
    }
} else {
    Write-Host "[Skip] sim_segments_$Date.csv missing" -ForegroundColor Yellow
}
if (Test-Path -LiteralPath $simBlendPath) {
    $uSimB = Upload-File -Uri "$BaseUrl/api/upload_sim_blend" -FilePath $simBlendPath -Query @{ date = $Date }
    if ($uSimB) { Write-Host "[OK] sim_blend uploaded: rows=$($uSimB.rows)" -ForegroundColor Green }
    else { ${needSimRetry} = $true }
} else {
    Write-Host "[Skip] sim_blend_$Date.csv missing" -ForegroundColor Yellow
}

# Upload simulation input diagnostics JSON if present
if (Test-Path -LiteralPath $simDiagPath) {
    $uSimD = Upload-File -Uri "$BaseUrl/api/upload_sim_inputs_diagnostic" -FilePath $simDiagPath -Query @{ date = $Date }
    if ($uSimD -and ($uSimD.status -eq 'ok')) { Write-Host "[OK] sim_inputs_diagnostic uploaded" -ForegroundColor Green }
    elseif ($uSimD -and ($uSimD.status -eq 'skipped')) { Write-Host "[Skip] sim_inputs_diagnostic endpoint unavailable" -ForegroundColor Yellow }
    else { Write-Host "[Warn] sim_inputs_diagnostic upload failed" -ForegroundColor Yellow }
} else {
    Write-Host "[Skip] sim_inputs_diagnostic_$Date.json missing" -ForegroundColor Yellow
}

# Upload global simulation calibration JSON if present
if (Test-Path -LiteralPath $simCalibPath) {
    $uSimC = Upload-File -Uri "$BaseUrl/api/upload_sim_calibration" -FilePath $simCalibPath
    if ($uSimC -and ($uSimC.status -eq 'ok')) { Write-Host "[OK] sim_calibration uploaded" -ForegroundColor Green }
    elseif ($uSimC -and ($uSimC.status -eq 'skipped')) { Write-Host "[Skip] sim_calibration endpoint unavailable" -ForegroundColor Yellow }
    else { Write-Host "[Warn] sim_calibration upload failed" -ForegroundColor Yellow }
} else {
    Write-Host "[Skip] sim_calibration.json missing" -ForegroundColor Yellow
}

# Upload quantile artifacts if present
$qselPath = Join-Path -Path $OutputsDir -ChildPath 'quantiles_selected.csv'
$qhistPath = Join-Path -Path $OutputsDir -ChildPath 'quantiles_history.csv'
$qmodelPath = Join-Path -Path $OutputsDir -ChildPath 'quantile_model.json'

if (Test-Path -LiteralPath $qselPath) {
    $uQsel = Upload-File -Uri "$BaseUrl/api/upload_quantiles_selected" -FilePath $qselPath
    if ($uQsel) { Write-Host "[OK] quantiles_selected uploaded: rows=$($uQsel.rows)" -ForegroundColor Green }
} else {
    Write-Host "[Skip] quantiles_selected.csv missing" -ForegroundColor Yellow
}
if (Test-Path -LiteralPath $qhistPath) {
    $uQhist = Upload-File -Uri "$BaseUrl/api/upload_quantiles_history" -FilePath $qhistPath
    if ($uQhist) { Write-Host "[OK] quantiles_history uploaded: rows=$($uQhist.rows)" -ForegroundColor Green }
} else {
    Write-Host "[Skip] quantiles_history.csv missing" -ForegroundColor Yellow
}
if (Test-Path -LiteralPath $qmodelPath) {
    try {
        $json = Get-Content -LiteralPath $qmodelPath -Raw
        Write-Step "POST $BaseUrl/api/upload_quantile_model with quantile_model.json"
        $resp = Invoke-RestMethod -Uri "$BaseUrl/api/upload_quantile_model" -Method Post -Body $json -ContentType 'application/json'
        if ($resp) { Write-Host "[OK] quantile_model uploaded: keys=$($resp.keys -join ',')" -ForegroundColor Green }
    } catch {
        Write-Host "[Error] quantile_model upload failed: $($_.Exception.Message)" -ForegroundColor Red
    }
} else {
    Write-Host "[Skip] quantile_model.json missing" -ForegroundColor Yellow
}

# Upload daily results if present
$resRows = Get-CsvRowCount -Path $resultsPath
if ($resRows -gt 0) {
    $u4 = Upload-File -Uri "$BaseUrl/api/upload_daily_results" -FilePath $resultsPath -Query @{ date = $Date }
    if ($u4) { Write-Host "[OK] results uploaded: rows=$($u4.rows)" -ForegroundColor Green }
} else {
    Write-Host "[Skip] daily results missing or empty for $Date" -ForegroundColor Yellow
}

# Proactive: persist display snapshot after uploads (upload-only mode)
try {
    Write-Step "Persist display for date $Date (upload-only)"
    $tsPersist = [int](Get-Date -UFormat %s)
    $pd0 = Invoke-RestMethod -Uri ("{0}/api/persist_display?date={1}&t={2}" -f $BaseUrl, $Date, $tsPersist) -Method Get
    if ($pd0 -and $pd0.ok) {
        Write-Host ("[OK] persist_display wrote {0} rows path={1}" -f $pd0.rows, $pd0.path) -ForegroundColor Green
    } else {
        Write-Host "[Warn] persist_display did not return ok=true (upload-only)" -ForegroundColor Yellow
    }

    # IMPORTANT: On older deployments, server-side persist_display can overwrite the
    # previously uploaded predictions_display_<date>.csv, dropping locally-derived
    # columns (e.g., edge_total/edge_ats). Re-upload display after persist_display
    # so the artifact on disk matches the local snapshot.
    if (-not $SlimSimOnly.IsPresent) {
        $sanPost = Sanitize-DisplayCsv -Path $displayPath
        if (-not $sanPost) { $sanPost = $displayPath }
        if (Test-Path -LiteralPath $sanPost) {
            $null = Upload-File -Uri "$BaseUrl/api/upload_predictions_display" -FilePath $sanPost -Query @{ date = $Date }
        }
    }
} catch {
    Write-Host "[Warn] persist_display (upload-only) call failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Verify artifacts and recommendations presence
Write-Step "Verifying debug_artifacts and recommendations"
$debug = $null
$recs = $null
$local_picks_rows = Get-CsvRowCount -Path $picksPath
$local_ats_rows = Get-CsvRowCount -Path $picksAtsPath
$local_edges_rows = Get-CsvRowCount -Path $edgesPath
$local_display_rows = Get-CsvRowCount -Path $displayPath
try {
    $debug = Invoke-RestMethod -Uri "$BaseUrl/api/debug_artifacts?date=$Date" -Method Get
    $art = $debug.artifacts
    function Get-ArtifactValue {
        param([object]$obj, [string]$name)
        if (-not $obj) { return $null }
        # PSCustomObject safe property access
        try {
            $prop = $obj.PSObject.Properties[$name]
            if ($prop) { return $prop.Value } else { return $null }
        } catch { return $null }
    }
    $p_val = Get-ArtifactValue -obj $art -name 'picks_raw.csv'
    $p_rows = if ($p_val) { $p_val.rows } else { $null }
    $e_key = "align_period_${Date}_edges.csv"
    $d_key = "predictions_display_${Date}.csv"
    $ap_key = "picks/ats_picks_${Date}.csv"
    $e_val = Get-ArtifactValue -obj $art -name $e_key
    $d_val = Get-ArtifactValue -obj $art -name $d_key
    $ap_val = Get-ArtifactValue -obj $art -name $ap_key
    $e_rows = if ($e_val) { $e_val.rows } else { $null }
    $d_rows = if ($d_val) { $d_val.rows } else { $null }
    $ap_rows = if ($ap_val) { $ap_val.rows } else { $null }
    Write-Host "[Debug] picks_raw_rows=$p_rows ats_picks_rows=$ap_rows edges_rows=$e_rows display_rows=$d_rows" -ForegroundColor White

    # If a redeploy is still in progress, uploads can land on the old instance and then get wiped.
    # Compare local row counts to remote and retry once for critical artifacts.
    $remote_edges_rows = if ($null -ne $e_rows) { [int]$e_rows } else { 0 }
    $remote_display_rows = if ($null -ne $d_rows) { [int]$d_rows } else { 0 }
    $remote_ats_rows = if ($null -ne $ap_rows) { [int]$ap_rows } else { 0 }
    $needsRetry = $false
    if ($local_edges_rows -gt 0 -and $remote_edges_rows -ne $local_edges_rows) { $needsRetry = $true }
    if ($local_display_rows -gt 0 -and $remote_display_rows -ne $local_display_rows) { $needsRetry = $true }
    if ($local_ats_rows -gt 0 -and $remote_ats_rows -ne $local_ats_rows) { $needsRetry = $true }

    if ($needsRetry) {
        Write-Host ("[Warn] Remote artifacts don't match local rows (local edges={0} display={1} ats={2}; remote edges={3} display={4} ats={5}). Retrying uploads once..." -f $local_edges_rows, $local_display_rows, $local_ats_rows, $remote_edges_rows, $remote_display_rows, $remote_ats_rows) -ForegroundColor Yellow
        Start-Sleep -Seconds 10
        if ($local_ats_rows -gt 0 -and $remote_ats_rows -ne $local_ats_rows) {
            $null = Upload-File -Uri "$BaseUrl/api/upload_ats_picks" -FilePath $picksAtsPath -Query @{ date = $Date }
        }
        if ($local_edges_rows -gt 0 -and $remote_edges_rows -ne $local_edges_rows) {
            $null = Upload-File -Uri "$BaseUrl/api/upload_align_edges" -FilePath $edgesPath -Query @{ date = $Date }
        }
        if ($local_display_rows -gt 0 -and $remote_display_rows -ne $local_display_rows) {
            $san = Sanitize-DisplayCsv -Path $displayPath
            if (-not $san) { $san = $displayPath }
            $null = Upload-File -Uri "$BaseUrl/api/upload_predictions_display" -FilePath $san -Query @{ date = $Date }
            try {
                $tsPersist2 = [int](Get-Date -UFormat %s)
                $null = Invoke-RestMethod -Uri ("{0}/api/persist_display?date={1}&t={2}" -f $BaseUrl, $Date, $tsPersist2) -Method Get
            } catch {}

            # Re-upload display again in case persist_display overwrote it.
            try {
                $sanPost2 = Sanitize-DisplayCsv -Path $displayPath
                if (-not $sanPost2) { $sanPost2 = $displayPath }
                if (Test-Path -LiteralPath $sanPost2) {
                    $null = Upload-File -Uri "$BaseUrl/api/upload_predictions_display" -FilePath $sanPost2 -Query @{ date = $Date }
                }
            } catch {}
        }
        # Re-check debug after retry
        $debug2 = Invoke-RestMethod -Uri "$BaseUrl/api/debug_artifacts?date=$Date" -Method Get
        $art2 = $debug2.artifacts
        $e_val2 = Get-ArtifactValue -obj $art2 -name $e_key
        $d_val2 = Get-ArtifactValue -obj $art2 -name $d_key
        $ap_val2 = Get-ArtifactValue -obj $art2 -name $ap_key
        $e_rows2 = if ($e_val2) { [int]$e_val2.rows } else { 0 }
        $d_rows2 = if ($d_val2) { [int]$d_val2.rows } else { 0 }
        $ap_rows2 = if ($ap_val2) { [int]$ap_val2.rows } else { 0 }
        Write-Host "[Debug] (after retry) ats_picks_rows=$ap_rows2 edges_rows=$e_rows2 display_rows=$d_rows2" -ForegroundColor White
        $stillBad = $false
        if ($local_edges_rows -gt 0 -and $e_rows2 -ne $local_edges_rows) { $stillBad = $true }
        if ($local_display_rows -gt 0 -and $d_rows2 -ne $local_display_rows) { $stillBad = $true }
        if ($local_ats_rows -gt 0 -and $ap_rows2 -ne $local_ats_rows) { $stillBad = $true }
        if ($stillBad) {
            Write-Host "[Error] Remote artifacts still out of sync after retry; failing upload script." -ForegroundColor Red
            exit 1
        }
    }
} catch {
    Write-Host "[Warn] debug_artifacts check failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

try {
    $tsRecs = [int](Get-Date -UFormat %s)
    $recs = Invoke-RestMethod -Uri ("{0}/api/recommendations?date={1}&t={2}" -f $BaseUrl, $Date, $tsRecs) -Method Get
    $rowCount = 0
    if ($recs) {
        $rowsArr = $null
        if ($recs.data) { $rowsArr = $recs.data }
        elseif ($recs.recommendations) { $rowsArr = $recs.recommendations }
        elseif ($recs.rows -is [System.Collections.IEnumerable]) { $rowsArr = $recs.rows }
        if ($rowsArr) { $rowCount = ($rowsArr | Measure-Object).Count }
        elseif ($recs.rows -is [int]) { $rowCount = [int]$recs.rows }
        else { $rowCount = 0 }
    }
    Write-Host "[Check] recommendations rows=$rowCount" -ForegroundColor White
    if ($rowCount -eq 0) {
        Write-Host "[Alert] Recommendations are empty. Check uploads and server fallbacks." -ForegroundColor Yellow
    }
    # Preflight: verify OU recommendations include numeric totals in labels and a non-empty line
    try {
        $rows = $null
        if ($recs.data) { $rows = $recs.data }
        elseif ($recs.recommendations) { $rows = $recs.recommendations }
        elseif ($recs.rows -is [System.Collections.IEnumerable]) { $rows = $recs.rows }
        if ($rows) {
            $ouRows = @($rows | Where-Object { (("" + $_.code).ToUpper() -eq 'OU') -or (("" + $_.rec_code).ToUpper() -eq 'OU') })
            if ($ouRows.Count -gt 0) {
                $bad = @()
                foreach ($r in $ouRows) {
                    $lbl = ("" + $r.bet_label).Trim()
                    if ([string]::IsNullOrWhiteSpace($lbl)) { $lbl = ("" + $r.bet).Trim() }
                    $ln = ("" + $r.line).Trim()
                    $hasDigit = ($lbl -match '\d')
                    if (-not $hasDigit -or [string]::IsNullOrWhiteSpace($ln)) {
                        $bad += $r
                    }
                }
                if ($bad.Count -gt 0) {
                    Write-Host ("[Error] OU label preflight failed: {0} rows missing numeric totals or line" -f $bad.Count) -ForegroundColor Red
                    throw "OU labels missing numeric totals or line after upload"
                } else {
                    Write-Host "[OK] OU preflight: all labels include numeric totals and line" -ForegroundColor Green
                }
            } else {
                Write-Host "[Warn] No OU rows found in recommendations to preflight" -ForegroundColor Yellow
            }
        }
    } catch {
        Write-Host ("[Error] OU preflight validation failed: {0}" -f $_.Exception.Message) -ForegroundColor Red
        throw
    }
} catch {
    Write-Host "[Warn] recommendations check failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Decide whether redeploy is necessary: skip if recommendations already include ATS
# and row count meets expected parity (ATS + OU at minimum).
${shouldSkipRedeploy} = $false
try {
    $rowsForSkip = $null
    if ($recs) {
        if ($recs.data) { $rowsForSkip = $recs.data }
        elseif ($recs.recommendations) { $rowsForSkip = $recs.recommendations }
        elseif ($recs.rows -is [System.Collections.IEnumerable]) { $rowsForSkip = $recs.rows }
    }
    $hasATS = $false
    if ($rowsForSkip) {
        foreach ($rr in $rowsForSkip) {
            $codeVal = ("" + ($rr.code))
            $recCodeVal = ("" + ($rr.rec_code))
            if ($codeVal.ToUpper() -eq 'ATS' -or $recCodeVal.ToUpper() -eq 'ATS') { $hasATS = $true; break }
        }
    }
    $minExpected = 0
    if ($ap_rows -is [int]) { $minExpected += [int]$ap_rows }
    if ($d_rows -is [int]) { $minExpected += [int]$d_rows }
    if ($rowCount -is [int]) {
        # If ATS is present and we have at least ATS + OU rows, skip redeploy
        if ($hasATS -and ([int]$rowCount) -ge ([Math]::Max(35, $minExpected))) {
            ${shouldSkipRedeploy} = $true
            Write-Host ("[Info] Redeploy not needed: ATS present and rows={0} >= expected={1}" -f $rowCount, $minExpected) -ForegroundColor Gray
        }
    }
} catch {}

# Verify display predictions parity vs local CSV
try {
    $localDisplayPath = $displayPath
    if ($sanitizedDisplayPath -and -not [string]::IsNullOrWhiteSpace($sanitizedDisplayPath)) { $localDisplayPath = $sanitizedDisplayPath }
    $localDisplayRows = Get-CsvRowCount -Path $localDisplayPath
    $tsDisp = [int](Get-Date -UFormat %s)
    $dispResp = Invoke-RestMethod -Uri ("{0}/api/display_predictions?date={1}&t={2}" -f $BaseUrl, $Date, $tsDisp) -Method Get
    $remoteDisplayRows = if ($dispResp -and $dispResp.rows) { ($dispResp.rows | Measure-Object).Count } elseif ($dispResp -and $dispResp.count) { [int]$dispResp.count } else { 0 }
    $note = if ($localDisplayRows -eq $remoteDisplayRows) { 'match' } else { 'mismatch' }
    Write-Host "[Check] display parity: local=$localDisplayRows remote=$remoteDisplayRows ($note)" -ForegroundColor White
} catch {
    Write-Host "[Warn] display parity check failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Verify results archive for the date
Write-Step "Verifying results archive"
try {
    $tsRes = [int](Get-Date -UFormat %s)
    $res = Invoke-RestMethod -Uri ("{0}/api/results?date={1}&t={2}" -f $BaseUrl, $Date, $tsRes) -Method Get
    $n = if ($res.rows) { ($res.rows | Measure-Object).Count } else { 0 }
    Write-Host "[Check] results rows=$n (date=$Date)" -ForegroundColor White
} catch {
    Write-Host "[Warn] results check failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Step "Done."