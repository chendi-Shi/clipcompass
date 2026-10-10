$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPythonCandidates = @(
    (Join-Path $PSScriptRoot '.venv/Scripts/python.exe'),
    (Join-Path $PSScriptRoot '.research-venv/Scripts/python.exe'),
    'python'
)
$taskPython = $null
foreach ($taskCandidate in $taskPythonCandidates) {
    if ($taskCandidate -ne 'python' -and -not (Test-Path -LiteralPath $taskCandidate)) { continue }
    & $taskCandidate -c 'import numpy, torch' 2>$null
    if ($LASTEXITCODE -eq 0) { $taskPython = $taskCandidate; break }
}
if (-not $taskPython) { throw 'Install dependencies first: python -m pip install -e .' }
$env:PYTHONPATH = $PSScriptRoot
& $taskPython -m clipcompass.cli demo-data
if ($LASTEXITCODE -ne 0) { throw 'Data generation failed' }
& $taskPython -m clipcompass.cli train --epochs 80
if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
& $taskPython -m clipcompass.cli recommend --user 0 --output artifacts/recommendations.json
if ($LASTEXITCODE -ne 0) { throw 'Recommendation failed' }
Write-Output 'Done. Read artifacts/REPORT.md and artifacts/recommendations.json.'
