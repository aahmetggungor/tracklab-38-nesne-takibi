$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    $python = Join-Path (Split-Path -Parent $projectRoot) '.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Python ortamı bulunamadı. README.md içindeki kurulum adımlarını uygulayın.'
}

$result = Join-Path $projectRoot 'results\summary.csv'
if (-not (Test-Path -LiteralPath $result)) {
    & $python (Join-Path $projectRoot 'download_assets.py')
    if ($LASTEXITCODE -ne 0) { throw 'Örnek veri/model indirme başarısız.' }
    & $python (Join-Path $projectRoot 'run_experiments.py')
    if ($LASTEXITCODE -ne 0) { throw 'Deney çalıştırma başarısız.' }
}

$ready = $false
try { $ready = (Invoke-WebRequest -Uri 'http://127.0.0.1:8503/' -TimeoutSec 2 -UseBasicParsing).StatusCode -eq 200 } catch {}
if (-not $ready) {
    Start-Process -FilePath $python -ArgumentList @('-m','streamlit','run','app.py',
        '--server.address','127.0.0.1','--server.port','8503','--browser.gatherUsageStats','false') `
        -WorkingDirectory $projectRoot -WindowStyle Hidden | Out-Null
}
for ($attempt = 0; $attempt -lt 25; $attempt++) {
    try {
        if ((Invoke-WebRequest -Uri 'http://127.0.0.1:8503/' -TimeoutSec 2 -UseBasicParsing).StatusCode -eq 200) {
            $ready = $true
            break
        }
    } catch {}
    Start-Sleep -Milliseconds 500
}
if (-not $ready) { throw 'TRACKLAB 38 arayüzü başlatılamadı. 8503 portunu kontrol edin.' }
Write-Host 'TRACKLAB 38 hazır: http://127.0.0.1:8503/'
Start-Process -FilePath 'http://127.0.0.1:8503/' | Out-Null
