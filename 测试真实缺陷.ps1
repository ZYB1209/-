$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$mflEvaluationPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (Test-Path -LiteralPath $mflEvaluationPython) {
    & $mflEvaluationPython evaluate_agent.py @args
} else {
    python evaluate_agent.py @args
}
exit $LASTEXITCODE
