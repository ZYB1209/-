$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$mflPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (Test-Path -LiteralPath $mflPython) { & $mflPython app.py }
else { python app.py }
exit $LASTEXITCODE
