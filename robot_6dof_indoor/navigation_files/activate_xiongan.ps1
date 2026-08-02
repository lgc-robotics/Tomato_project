$ErrorActionPreference = "Stop"

$XionganEnv = "C:\Users\Administrator\.virtualenvs\xiongan"
$ProjectRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$ActivateScript = Join-Path $XionganEnv "Scripts\Activate.ps1"

if (-not (Test-Path -LiteralPath $ActivateScript)) {
    throw "xiongan environment was not found: $XionganEnv"
}

Set-Location $ProjectRoot
& $ActivateScript

Write-Host ""
Write-Host "xiongan environment activated." -ForegroundColor Green
Write-Host "Project: $ProjectRoot" -ForegroundColor Green
Write-Host "Python:  $XionganEnv\Scripts\python.exe" -ForegroundColor Green
Write-Host ""
