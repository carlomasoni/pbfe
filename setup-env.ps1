

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Info($Message) {
    Write-Host "[INFO] $Message"
}

function Write-Warn($Message) {
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-ErrorAndExit($Message) {
    Write-Error $Message
    exit 1
}

Write-Info "Checking for Python..."
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-ErrorAndExit "Python was not found on PATH. Install Python 3.10+ and rerun this script."
}

$venvPath = ".venv"
if (-not (Test-Path $venvPath)) {
    Write-Info "Creating virtual environment at '$venvPath'..."
    python -m venv $venvPath
} else {
    Write-Info "Virtual environment already exists at '$venvPath'."
}

$activate = Join-Path $venvPath "Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
    Write-ErrorAndExit "Activation script not found at '$activate'. The virtual environment setup may have failed."
}

Write-Info "Upgrading pip and installing dependencies..."
& (Join-Path $venvPath "Scripts\python.exe") -m pip install --upgrade pip

$requirements = "pbfe\requirements.txt"
if (Test-Path $requirements) {
    & (Join-Path $venvPath "Scripts\python.exe") -m pip install -r $requirements
} else {
    Write-Warn "No requirements file found at '$requirements'; skipping dependency installation."
}

Write-Info "Setup complete."
Write-Info "Activate the environment in new shells with:"
Write-Host "    .\.venv\Scripts\Activate.ps1"


