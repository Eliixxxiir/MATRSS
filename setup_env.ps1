# One-shot environment setup: creates .venv (if missing) and installs requirements.txt.
# Usage (from the MATRSS folder):   .\setup_env.ps1
# If PowerShell blocks scripts:     powershell -ExecutionPolicy Bypass -File .\setup_env.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating .venv ..."
    python -m venv .venv
}

$py = ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip --quiet
& $py -m pip install -r requirements.txt

Write-Host "`nVerifying imports ..."
& $py -c "import numpy, pandas, yaml, matplotlib, matrss; print('OK - matrss', matrss.__version__)"
