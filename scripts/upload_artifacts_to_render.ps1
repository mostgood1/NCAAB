param(
    [string]$Date = (Get-Date).ToString('yyyy-MM-dd'),
    [string]$BaseUrl = 'https://ncaab.onrender.com',
    [string]$OutputsDir = "$PSScriptRoot/../outputs",
    [switch]$TriggerRedeploy,
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
        $fileContent.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::Parse('text/csv')
        $fileName = [System.IO.Path]::GetFileName($FilePath)
        $content.Add($fileContent, 'file', $fileName)
        $resp = $client.PostAsync($target, $content).Result
        $text = $resp.Content.ReadAsStringAsync().Result
        try { $fs.Dispose() } catch {}
        try { $client.Dispose() } catch {}
        if (-not $resp.IsSuccessStatusCode) {
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

Write-Step "Using date=$Date, baseUrl=$BaseUrl"
Write-Step "Outputs dir: $OutputsDir"

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
            # Write header-only file to preserve schema
            $rows | Select-Object -First 0 | Export-Csv -LiteralPath $tmp -NoTypeInformation -Encoding UTF8
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
$u3 = Upload-File -Uri "$BaseUrl/api/upload_predictions_display" -FilePath $displayToUpload -Query @{ date = $Date }
if ($u3) {
    $rv = if ($u3.rows_verified) { $u3.rows_verified } elseif ($u3.rows) { $u3.rows } else { $null }
    $ru = if ($u3.rows_uploaded) { $u3.rows_uploaded } else { $null }
    $sha = if ($u3.sha) { $u3.sha } else { $null }
    $shaSuffix = if ($sha) { " sha=$sha" } else { "" }
    Write-Host ("[OK] display uploaded: rows_uploaded={0} rows_verified={1}{2}" -f $ru, $rv, $shaSuffix) -ForegroundColor Green
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

$u3b = Upload-File -Uri "$BaseUrl/api/upload_predictions_enriched" -FilePath $enrichedToUpload -Query @{ date = $Date }
if ($u3b) {
    $rv = if ($u3b.rows_verified) { $u3b.rows_verified } elseif ($u3b.rows) { $u3b.rows } else { $null }
    $ru = if ($u3b.rows_uploaded) { $u3b.rows_uploaded } else { $null }
    $sha = if ($u3b.sha) { $u3b.sha } else { $null }
    $shaSuffix = if ($sha) { " sha=$sha" } else { "" }
    Write-Host ("[OK] enriched uploaded: rows_uploaded={0} rows_verified={1}{2}" -f $ru, $rv, $shaSuffix) -ForegroundColor Green
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

# Verify artifacts and recommendations presence
Write-Step "Verifying debug_artifacts and recommendations"
$debug = $null
$recs = $null
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
} catch {
    Write-Host "[Warn] debug_artifacts check failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

try {
    $recs = Invoke-RestMethod -Uri "$BaseUrl/api/recommendations?date=$Date" -Method Get
    $rowCount = 0
    if ($recs) {
        if ($recs.rows) { $rowCount = ($recs.rows | Measure-Object).Count }
        elseif ($recs.data) { $rowCount = ($recs.data | Measure-Object).Count }
        elseif ($recs.recommendations) { $rowCount = ($recs.recommendations | Measure-Object).Count }
    }
    Write-Host "[Check] recommendations rows=$rowCount" -ForegroundColor White
    if ($rowCount -eq 0) {
        Write-Host "[Alert] Recommendations are empty. Check uploads and server fallbacks." -ForegroundColor Yellow
    }
} catch {
    Write-Host "[Warn] recommendations check failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Verify display predictions parity vs local CSV
try {
    $localDisplayPath = $displayPath
    if ($sanitizedDisplayPath -and -not [string]::IsNullOrWhiteSpace($sanitizedDisplayPath)) { $localDisplayPath = $sanitizedDisplayPath }
    $localDisplayRows = Get-CsvRowCount -Path $localDisplayPath
    $dispResp = Invoke-RestMethod -Uri "$BaseUrl/api/display_predictions?date=$Date" -Method Get
    $remoteDisplayRows = if ($dispResp -and $dispResp.rows) { ($dispResp.rows | Measure-Object).Count } elseif ($dispResp -and $dispResp.count) { [int]$dispResp.count } else { 0 }
    $note = if ($localDisplayRows -eq $remoteDisplayRows) { 'match' } else { 'mismatch' }
    Write-Host "[Check] display parity: local=$localDisplayRows remote=$remoteDisplayRows ($note)" -ForegroundColor White
} catch {
    Write-Host "[Warn] display parity check failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Verify results archive for the date
Write-Step "Verifying results archive"
try {
    $res = Invoke-RestMethod -Uri "$BaseUrl/api/results?date=$Date" -Method Get
    $n = if ($res.rows) { ($res.rows | Measure-Object).Count } else { 0 }
    Write-Host "[Check] results rows=$n (date=$Date)" -ForegroundColor White
} catch {
    Write-Host "[Warn] results check failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Optional redeploy trigger via Render deploy hook
if ($TriggerRedeploy.IsPresent) {
    function Get-DeployHookUrl {
        param()
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
    if ([string]::IsNullOrWhiteSpace($DeployHookUrl)) { $DeployHookUrl = Get-DeployHookUrl }
    # Capture both hook URLs if available: prefer explicit code hook, but fall back to service hook as well
    $codeHook = $null
    $serviceHook = $null
    if ($env:RENDER_CODE_DEPLOY_HOOK_URL -and -not [string]::IsNullOrWhiteSpace($env:RENDER_CODE_DEPLOY_HOOK_URL)) { $codeHook = $env:RENDER_CODE_DEPLOY_HOOK_URL }
    if ($env:RENDER_DEPLOY_HOOK_URL -and -not [string]::IsNullOrWhiteSpace($env:RENDER_DEPLOY_HOOK_URL)) { $serviceHook = $env:RENDER_DEPLOY_HOOK_URL }
    if ([string]::IsNullOrWhiteSpace($DeployHookUrl)) { $DeployHookUrl = if ($codeHook) { $codeHook } else { $serviceHook } }
    if ([string]::IsNullOrWhiteSpace($DeployHookUrl)) {
        Write-Host "[Skip] TriggerRedeploy set but no DeployHookUrl provided, env var set, or .env/scripts fallback found." -ForegroundColor Yellow
    } else {
        Write-Step "Triggering redeploy via deploy hook"
        # Capture baseline app version before triggering
        $baselineSha = $null
        $baselineBuildTime = $null
        try {
            $ver0 = Invoke-RestMethod -Uri "$BaseUrl/api/version" -Method Get
            if ($ver0) { $baselineSha = $ver0.app_sha; $baselineBuildTime = $ver0.build_time_utc }
            Write-Host ("[Check] Baseline version: sha={0} build_time={1}" -f $baselineSha, $baselineBuildTime) -ForegroundColor White
        } catch { Write-Host "[Warn] Baseline version check failed: $($_.Exception.Message)" -ForegroundColor Yellow }
        function Invoke-DeployHookAndPoll {
            param([string]$HookUrl, [datetime]$Deadline, [string]$Label)
            if ([string]::IsNullOrWhiteSpace($HookUrl)) { return $false }
            try {
                $hookResp = Invoke-RestMethod -Uri $HookUrl -Method Post
                Write-Host ("[OK] {0} hook response received." -f $Label) -ForegroundColor Green
            } catch {
                Write-Host ("[Error] {0} hook failed: {1}" -f $Label, $_.Exception.Message) -ForegroundColor Red
            }
            $changedLocal = $false
            while ((Get-Date) -lt $Deadline) {
                Start-Sleep -Milliseconds $VersionPollIntervalMs
                try {
                    $ver = Invoke-RestMethod -Uri "$BaseUrl/api/version" -Method Get
                    if ($ver) {
                        $sha = $ver.app_sha
                        $bt = $ver.build_time_utc
                        Write-Host ("[Poll] Version: sha={0} build_time={1}" -f $sha, $bt) -ForegroundColor Gray
                        if ($baselineSha -and $sha -and $sha -ne $baselineSha) { $changedLocal = $true; break }
                        if ($baselineBuildTime -and $bt -and $bt -ne $baselineBuildTime) { $changedLocal = $true; break }
                        if (-not $baselineBuildTime -and $bt) { $changedLocal = $true; break }
                    }
                } catch { Write-Host "[Warn] Version poll failed: $($_.Exception.Message)" -ForegroundColor Yellow }
            }
            return $changedLocal
        }

        $deadline1 = (Get-Date).AddSeconds([double]$VersionPollSeconds * 0.6)
        $deadline2 = (Get-Date).AddSeconds([double]$VersionPollSeconds)
        $changed = $false
        # Try code hook first if available
        if ($codeHook) { $changed = Invoke-DeployHookAndPoll -HookUrl $codeHook -Deadline $deadline1 -Label 'Code deploy' }
        if (-not $changed) {
            # Fall back to provided DeployHookUrl (may equal serviceHook) and allow more time
            $changed = Invoke-DeployHookAndPoll -HookUrl $DeployHookUrl -Deadline $deadline2 -Label 'Service deploy'
        }
        if (-not $changed -and $serviceHook -and $DeployHookUrl -ne $serviceHook) {
            # As a last resort, try the explicit service hook if different
            $deadline3 = (Get-Date).AddSeconds([double]$VersionPollSeconds * 1.25)
            $changed = Invoke-DeployHookAndPoll -HookUrl $serviceHook -Deadline $deadline3 -Label 'Service deploy (fallback)'
        }
        if ($changed) { Write-Host "[OK] Detected new deployment version." -ForegroundColor Green }
        else { Write-Host "[Warn] Version unchanged after polling; deployment may still be in progress or using previous image." -ForegroundColor Yellow }

        # Post-deploy: run backtest totals to compute calibration offset, then persist display
        try {
            Write-Step "Post-deploy backtest_totals for date $Date"
            $bt = Invoke-RestMethod -Uri "$BaseUrl/api/backtest-totals?date=$Date" -Method Get
            if ($bt) {
                $bias = if ($bt.bias) { $bt.bias } else { $null }
                $mae = if ($bt.mae) { $bt.mae } else { $null }
                $rmse = if ($bt.rmse) { $bt.rmse } else { $null }
                Write-Host ("[OK] backtest_totals: bias={0} mae={1} rmse={2}" -f $bias, $mae, $rmse) -ForegroundColor Green
            } else {
                Write-Host "[Warn] backtest_totals returned empty response" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "[Warn] backtest_totals call failed: $($_.Exception.Message)" -ForegroundColor Yellow
        }

        # After backtest, proactively persist display for the date to enrich snapshot with odds and calibrated totals
        try {
            Write-Step "Post-deploy persist_display for date $Date"
            $pd = Invoke-RestMethod -Uri "$BaseUrl/api/persist_display?date=$Date" -Method Get
            if ($pd -and $pd.ok) {
                Write-Host ("[OK] persist_display wrote {0} rows path={1}" -f $pd.rows, $pd.path) -ForegroundColor Green
            } else {
                Write-Host "[Warn] persist_display did not return ok=true" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "[Warn] persist_display call failed: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
}

Write-Step "Done."