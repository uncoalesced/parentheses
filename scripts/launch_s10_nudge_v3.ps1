# Engineered by uncoalesced
# Session 10 Option 3 Format Nudge v3 Launcher (50 Steps: 14124 -> 14174)
# Follows the EPOCH RULE: 500 bins for 50 steps x 16 visits = 1.6 epochs (<= 2.0 epochs)

$ErrorActionPreference = "Stop"
$RepoRoot = "E:\Parentheses\parentheses-0.9"
Set-Location $RepoRoot

$Lock = "$RepoRoot\out\locks\GPU.lock"
$PythonExe = "$RepoRoot\.venv\Scripts\python.exe"
$DataDir = "$RepoRoot\data\processed_sft_240m_s10_nudge_v3"
$ResumeCkpt = "$RepoRoot\checkpoints\parentheses-1.0-240M-s10-blend\sft-patch-winner.pt"
$OutDir = "I:\parentheses\checkpoints\parentheses-1.0-240M-s10-nudge-v3"

if (Test-Path $Lock) {
    Write-Error "GPU lock already exists: $(Get-Content $Lock)"
}

if (!(Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
}

$LogDir = "$RepoRoot\out\s10"
if (!(Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

$LogFile = "$LogDir\nudge_v3_train.log"
"antigravity session10-nudge-v3 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss IST')" | Out-File -FilePath $Lock -Encoding utf8
Write-Host "[*] Acquired GPU.lock for S10 Option 3 Nudge v3"

try {
    & $PythonExe train.py `
        --preset parentheses-1.0-240M `
        --resume-from $ResumeCkpt `
        --data "$DataDir/train.bin" `
        --boundaries "$DataDir/train_boundaries_flat.npy" `
        --sft-spans "$DataDir/train.mask.bin" `
        --container-sft `
        --tokenizer "data/tokenizer/v3_32k" `
        --out-dir $OutDir `
        --batch-size 1 `
        --grad-accum 16 `
        --ce-rows 512 `
        --lr 5.0e-6 `
        --min-lr 2.0e-6 `
        --warmup-steps 3 `
        --decay-start 14127 `
        --decay-end 14174 `
        --eos-weight 3.0 `
        --differential-adamw `
        --recurrent-lr-ratio 0.1 `
        --grad-checkpointing `
        --abort-below-loss 0.30 `
        --max-steps 14174 `
        --log-every 1 `
        --ckpt-every 25 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $LogFile
    
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        throw "train.py option 3 nudge v3 exited with code $code"
    }
    Write-Host "[+] S10 Option 3 Nudge v3 training completed successfully."
} finally {
    if (Test-Path $Lock) {
        Remove-Item $Lock -Force -ErrorAction SilentlyContinue
        Write-Host "[*] Released GPU.lock"
    }
}
