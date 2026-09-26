<#
  One-click reproduction: unit tests -> batch run -> format validation ->
  structural diagnostic -> graph edit distance bounds.

  Usage (ASCII only so Windows PowerShell 5.1 parses the file regardless of the
  console code page):
    powershell -ExecutionPolicy Bypass -File run_all.ps1
    powershell -ExecutionPolicy Bypass -File run_all.ps1 -Start 1 -End 10 -Out output\demo
#>
param(
    [int]$Start = 1,
    [int]$End = 40,
    [string]$Out = "output\final",
    [string]$LabelsCache = "output\labels-cache",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "virtual environment not found: $python" }

function Invoke-Step {
    param([string]$Name, [scriptblock]$Body)
    Write-Host "== $Name" -ForegroundColor Cyan
    & $Body
    if ($LASTEXITCODE -ne 0) { throw "step failed: $Name" }
}

if (-not $SkipTests) {
    Invoke-Step -Name "unit tests" -Body { & $python test_topology.py }
}

Invoke-Step -Name "batch run ($Start-$End)" -Body {
    & $python run_eda_batch.py --start $Start --end $End --out $Out `
        --labels-cache $LabelsCache --aux-labels-cache $LabelsCache
}
Invoke-Step -Name "format validation" -Body {
    & $python validate_eda_outputs.py --output $Out
}
Invoke-Step -Name "structural diagnostic" -Body {
    & $python evaluate_connections.py --generated $Out
}
Invoke-Step -Name "graph edit distance bounds" -Body {
    & $python ged_metric.py --mode bounds --generated $Out
}

Write-Host ""
Write-Host "done. artifacts: $Out" -ForegroundColor Green
Write-Host "  review list : $Out\qa_summary.md"
Write-Host "  metrics     : $Out\connection_diagnostic.json and $Out\ged_bounds.json"
