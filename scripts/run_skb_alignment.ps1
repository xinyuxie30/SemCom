param(
    [int]$Trials = 40,
    [double]$NodeKeepProb = 0.65,
    [double]$EdgeKeepProb = 0.55
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$Python = "D:\anconda\envs\pytorch\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = "python"
}

$ResultDir = Join-Path $ProjectRoot "experiments\results"
New-Item -ItemType Directory -Force -Path $ResultDir | Out-Null
$LogPath = Join-Path $ResultDir "e8_skb_alignment_terminal.log"

"=" * 72 | Tee-Object -FilePath $LogPath
"E8 KG-SKB cloud-edge alignment run" | Tee-Object -FilePath $LogPath -Append
"Project: $ProjectRoot" | Tee-Object -FilePath $LogPath -Append
"Python:  $Python" | Tee-Object -FilePath $LogPath -Append
"Trials:  $Trials" | Tee-Object -FilePath $LogPath -Append
"NodeKeepProb: $NodeKeepProb  EdgeKeepProb: $EdgeKeepProb" | Tee-Object -FilePath $LogPath -Append
"Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Tee-Object -FilePath $LogPath -Append
"=" * 72 | Tee-Object -FilePath $LogPath -Append

function Run-Step {
    param(
        [string]$Name,
        [string[]]$ArgsList
    )

    "" | Tee-Object -FilePath $LogPath -Append
    "---- $Name ----" | Tee-Object -FilePath $LogPath -Append
    $cmdLine = "$Python " + ($ArgsList -join " ")
    $cmdLine | Tee-Object -FilePath $LogPath -Append

    & $Python @ArgsList 2>&1 | Tee-Object -FilePath $LogPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

Run-Step "Unit tests" @("-m", "unittest", "discover", "-s", "tests")
Run-Step "Compile checks" @(
    "-m", "py_compile",
    "agents\slow_loop\skb_server.py",
    "utils\semantic_consistency.py",
    "experiments\eval_skb_alignment.py",
    "tests\test_transe_pipeline.py"
)
Run-Step "E8 SKB alignment experiment" @(
    "experiments\eval_skb_alignment.py",
    "--trials", "$Trials",
    "--node-keep-prob", "$NodeKeepProb",
    "--edge-keep-prob", "$EdgeKeepProb"
)

"" | Tee-Object -FilePath $LogPath -Append
"Completed: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Tee-Object -FilePath $LogPath -Append
"Raw CSV: experiments\results\skb_alignment.csv" | Tee-Object -FilePath $LogPath -Append
"Summary CSV: experiments\results\skb_alignment_summary.csv" | Tee-Object -FilePath $LogPath -Append
"Terminal log: $LogPath" | Tee-Object -FilePath $LogPath -Append
"=" * 72 | Tee-Object -FilePath $LogPath -Append
