# Style Studio - one-shot setup for Windows PowerShell.
# Run from the project root:
#   cd C:\Users\Asus\Desktop\style-studio
#   .\setup.ps1

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== Style Studio setup ===" -ForegroundColor Cyan
Write-Host ""

if (-Not (Test-Path "requirements.txt")) {
    Write-Host "ERROR: run this from the style-studio folder." -ForegroundColor Red
    exit 1
}

if (-Not (Test-Path ".venv")) {
    Write-Host "Creating Python venv (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "Venv already exists, skipping create." -ForegroundColor Green
}

Write-Host "Activating venv..." -ForegroundColor Yellow
& ".\.venv\Scripts\Activate.ps1"

Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

Write-Host "Installing dependencies (this takes 2-4 min the first time)..." -ForegroundColor Yellow
pip install -r requirements.txt

if (-Not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host ""
    Write-Host "Created .env from template. Open it and fill in REPLICATE_API_TOKEN later." -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Put a selfie at:  tests\selfies\me.jpg"
Write-Host "  2. Run face analysis test:"
Write-Host "       python tests\run_local_test.py tests\selfies\me.jpg"
Write-Host "  3. Or run with visual debug overlay (saves an annotated image):"
Write-Host "       python tests\run_local_test.py tests\selfies\me.jpg --debug-out tests\debug_out\me_annotated.jpg"
Write-Host "  4. Start the API server:"
Write-Host "       uvicorn backend.main:app --reload"
Write-Host "     Then open http://127.0.0.1:8000/docs"
Write-Host ""
