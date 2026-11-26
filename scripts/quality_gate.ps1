$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "Quality gate: lint"
ruff check . --exclude .venv --exclude results --exclude tmp --exclude models --exclude .pytest_cache
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Quality gate: type checking"
mypy
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Quality gate: tests"
python -m pytest -q --durations=10
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Quality gate: golden references"
python -m pytest -q tests/test_golden_references.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Quality gate: number audit"
python scripts/audit_numbers.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Quality gate: documentation links"
python scripts/check_links.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Quality gate: results integrity"
python scripts/results_integrity.py --check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "QUALITY GATE PASSED"
