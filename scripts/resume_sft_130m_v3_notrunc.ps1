# Resume the 130M-v3 no-truncation SFT run (paused 2026-10-26 15:07 IST at step 750).
# Calls train.py directly: launch_sft_130m_v3.ps1's resume path sets target = step + Steps
# and shifts the cooldown, which would break the 2000-step / decay-at-1500 schedule.
# Optimizer state is in the checkpoint, so no warmup on resume.

param(
    [string]$ResumeFrom = "I:/parentheses/checkpoints/parentheses-0.9-130M-v3-sft-notrunc/step_750.pt"
)

Set-Location E:\Parentheses\parentheses-0.9
$env:PYTHONIOENCODING = 'utf-8'
$py = '.venv\Scripts\python.exe'
$log = 'out\launch_sft_130m_v3.log'
$trainLog = 'out\sft_130m_v3_notrunc.log'
$lockFile = 'out\locks\GPU.lock'

function Say($m) {
    $l = "$(Get-Date -f 'yyyy-MM-dd HH:mm:ss') $m"
    Write-Host $l
    $l | Out-File $log -Append -Encoding utf8
}

if (-not (Test-Path $ResumeFrom)) { Say "[ABORT] resume checkpoint not found: $ResumeFrom"; exit 1 }
if (Test-Path $lockFile) {
    Say "[ABORT] GPU.lock held by: $((Get-Content $lockFile -Raw).Trim())"
    exit 1
}
"parentheses-130m-v3-sft-notrunc-resume (PID $PID, started $(Get-Date -f 'yyyy-MM-dd HH:mm:ss'))" | Out-File $lockFile -Encoding ascii
Say "Resuming no-truncation SFT from $ResumeFrom (target step 1999, cooldown from 1500)"

$trainArgs = @(
    'train.py',
    '--preset', 'parentheses-0.9-130M-v3',
    '--data', 'data/processed_sft_stem_open_v2/train.bin',
    '--sft-spans', 'data/processed_sft_stem_open_v2/train.mask.notrunc.bin',
    '--boundaries', 'data/processed_sft_stem_open_v2/train_boundaries.npy',
    '--resume-from', $ResumeFrom,
    '--tokenizer', 'data/tokenizer/v3_32k',
    '--out-dir', 'checkpoints/parentheses-0.9-130M-v3-sft-notrunc',
    '--batch-size', 4, '--grad-accum', 4, '--ce-rows', 4096,
    '--lr', 3e-5, '--min-lr', 5e-6, '--decay-start', 1500, '--max-steps', 2000,
    '--warmup-steps', 0,
    '--differential-adamw', '--recurrent-lr-ratio', '0.1',
    '--agc', '--clip-factor', '0.05', '--decay-clip-factor', '0.01',
    '--weight-decay', '0.05', '--grad-checkpointing',
    '--ckpt-every', '250', '--log-every', '10',
    '--replay-data', 'data/processed_130m_v1/train.bin', '--replay-frac', 0.2
)

try {
    & $py -u $trainArgs 2>&1 | ForEach-Object { "$_" } | Out-File $trainLog -Append -Encoding utf8
    if ($LASTEXITCODE -eq 0) { Say "No-truncation SFT finished." } else { Say "[FAIL] train.py exited with code $LASTEXITCODE" }
} finally {
    if (Test-Path $lockFile) { Remove-Item $lockFile -Force; Say "Released GPU.lock." }
}
