$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

# Local preparation only. Does not install CUDA, download weights or rent compute.
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Install Python 3.11 from python.org and retry." }
}
$OvrPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"
& $OvrPython -m pip install -r requirements-local.txt
if ($LASTEXITCODE -ne 0) { throw "Local dependency installation failed." }
& $OvrPython scripts/check_windows_host.py
if ($LASTEXITCODE -ne 0) { throw "Shared pilot validation failed." }
Write-Host "Preparation finished. Read WINDOWS_START_HERE.md before model inference."
