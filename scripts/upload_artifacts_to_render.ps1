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
$picksPath   = Join-Path -Path $OutputsDir -ChildPath 'picks_raw.csv'
$edgesPath   = Join-Path -Path $OutputsDir -ChildPath ("align_period_{0}_edges.csv" -f $Date)
$displayPath = Join-Path -Path $OutputsDir -ChildPath ("predictions_display_{0}.csv" -f $Date)

Write-Step "Using date=$Date, baseUrl=$BaseUrl"
Write-Step "Outputs dir: $OutputsDir"

function Get-CsvRowCount {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return 0 }
    try {
        $lines = Get-Content -LiteralPath $Path -ErrorAction Stop
        if (-not $lines) { return 0 }
        # assume header present; count data rows
        return [Math]::Max(0, $lines.Length - 1)
    } catch { return 0 }
}

# Upload in preferred order: picks_raw -> edges(date) -> display(date)
$picksRows = Get-CsvRowCount -Path $picksPath
if ($picksRows -gt 0) {
    $u1 = Upload-File -Uri "$BaseUrl/api/upload_picks_raw" -FilePath $picksPath
    if ($u1) { Write-Host "[OK] picks_raw uploaded: $($u1.path) rows=$($u1.rows)" -ForegroundColor Green }
} else {
    Write-Host "[Skip] picks_raw.csv has 0 rows; preserving remote non-empty file." -ForegroundColor Yellow
}

$u2 = Upload-File -Uri "$BaseUrl/api/upload_align_edges" -FilePath $edgesPath -Query @{ date = $Date }
if ($u2) { Write-Host "[OK] edges uploaded: rows=$($u2.rows)" -ForegroundColor Green }

$u3 = Upload-File -Uri "$BaseUrl/api/upload_predictions_display" -FilePath $displayPath -Query @{ date = $Date }
if ($u3) { Write-Host "[OK] display uploaded: rows=$($u3.rows)" -ForegroundColor Green }

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