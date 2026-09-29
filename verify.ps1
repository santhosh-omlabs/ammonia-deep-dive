Set-Location $PSScriptRoot
python -m pytest tests -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
