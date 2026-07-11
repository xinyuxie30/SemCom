$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = "D:\anconda\envs\pytorch\python.exe"
$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$SaveDir = Join-Path $Root "checkpoints\e1_transe_fb15k237_$RunStamp"
$LogDir = Join-Path $Root "experiments\results\e1_transe_fb15k237_$RunStamp"

New-Item -ItemType Directory -Force -Path $SaveDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Seeds = @(42, 123, 2025)

foreach ($Seed in $Seeds) {
    $TagSuffix = "seed$Seed"
    $LogPath = Join-Path $LogDir "transe_fb15k237_$TagSuffix.log"
    Write-Host "Starting FB15K237 TransE seed=$Seed"
    & $Python (Join-Path $Root "scripts\train_transe_unified.py") `
        --dataset FB15K237 `
        --model transE `
        --epochs 3000 `
        --dim 128 `
        --seed $Seed `
        --device cuda `
        --nbatches 100 `
        --lr 1.0 `
        --eval-every 50 `
        --fast-eval-size 2000 `
        --eval-max 0 `
        --save-dir $SaveDir `
        --tag-suffix $TagSuffix 2>&1 | Tee-Object -FilePath $LogPath
}

$SummaryPath = Join-Path $LogDir "summary.csv"
& $Python -c @"
import csv, json, pathlib
save_dir = pathlib.Path(r'$SaveDir')
rows = []
for path in sorted(save_dir.glob('transe_fb15k237_transe_seed*_eval.json')):
    data = json.load(open(path, encoding='utf-8'))
    rows.append({
        'file': path.name,
        'seed': data.get('seed'),
        'MRR': data.get('MRR'),
        'Hits@1': data.get('Hits@1'),
        'Hits@3': data.get('Hits@3'),
        'Hits@10': data.get('Hits@10'),
        'epochs_recorded': len(data.get('history', [])),
        'final_train_mrr': data.get('history', [{}])[-1].get('train_mrr'),
        'final_test_mrr_fast': data.get('history', [{}])[-1].get('test_mrr'),
        'elapsed_sec': data.get('elapsed_sec'),
    })
summary = pathlib.Path(r'$SummaryPath')
with open(summary, 'w', newline='', encoding='utf-8') as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ['file'])
    writer.writeheader()
    writer.writerows(rows)
print(f'Summary saved -> {summary}')
"@

Write-Host "E1 complete"
Write-Host "Checkpoints: $SaveDir"
Write-Host "Logs: $LogDir"
