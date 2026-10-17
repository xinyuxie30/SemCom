param([string]$Resume = "")
$ErrorActionPreference = "Stop"
$runner = Join-Path $PSScriptRoot "run_suite.py"
$python = "D:\PyChram_project\DCAT-SoM\.venv\Scripts\python.exe"
$arguments = @($runner, "--mode", "formal", "--live-api")
if ($Resume) { $arguments += @("--resume", $Resume) }
& $python @arguments
exit $LASTEXITCODE
