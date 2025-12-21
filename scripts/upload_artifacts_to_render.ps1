param(
    [string]$Date = (Get-Date).ToString('yyyy-MM-dd'),
    [string]$BaseUrl = 'https://ncaab.onrender.com',
    [string]$OutputsDir = "$PSScriptRoot/../outputs",
    [switch]$TriggerRedeploy,
    [string]$DeployHookUrl = $env:RENDER_DEPLOY_HOOK_URL
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
if ($u2) { Write-Host "[OK] edges uploaded: rows=$($u2.rows)" -ForegroundColor Green }

$u3 = Upload-File -Uri "$BaseUrl/api/upload_predictions_display" -FilePath $displayPath -Query @{ date = $Date }
if ($u3) { Write-Host "[OK] display uploaded: rows=$($u3.rows)" -ForegroundColor Green }

# Upload enriched predictions snapshot for recommendations parity
$u3b = Upload-File -Uri "$BaseUrl/api/upload_predictions_enriched" -FilePath $enrichedPath -Query @{ date = $Date }
if ($u3b) { Write-Host "[OK] enriched uploaded: rows=$($u3b.rows)" -ForegroundColor Green }

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
    Write-Host "[Debug] exist(picks)=$($debug.global_picks_raw.exists) exist(edges)=$($debug.edges.exists) display_rows=$($debug.display.rows)" -ForegroundColor White
} catch {
    Write-Host "[Warn] debug_artifacts check failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

try {
    $recs = Invoke-RestMethod -Uri "$BaseUrl/api/recommendations?date=$Date" -Method Get
    $rowCount = if ($recs -is [System.Collections.IEnumerable]) { ($recs | Measure-Object).Count } else { 0 }
    Write-Host "[Check] recommendations rows=$rowCount" -ForegroundColor White
    if ($rowCount -eq 0) {
        Write-Host "[Alert] Recommendations are empty. Check uploads and server fallbacks." -ForegroundColor Yellow
    }
} catch {
    Write-Host "[Warn] recommendations check failed: $($_.Exception.Message)" -ForegroundColor Yellow
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
    if ([string]::IsNullOrWhiteSpace($DeployHookUrl)) {
        Write-Host "[Skip] TriggerRedeploy set but no DeployHookUrl provided or env var set." -ForegroundColor Yellow
    } else {
        Write-Step "Triggering redeploy via deploy hook"
        try {
            $hookResp = Invoke-RestMethod -Uri $DeployHookUrl -Method Post
            Write-Host "[OK] Redeploy hook response received." -ForegroundColor Green
        } catch {
            Write-Host "[Error] Redeploy hook failed: $($_.Exception.Message)" -ForegroundColor Red
        }
    }
}

Write-Step "Done."