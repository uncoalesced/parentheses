# Engineered by uncoalesced
# Session 10 Micro-Patch Launcher (200 Steps: 14000 -> 14200)

$ErrorActionPreference = "Stop"
$RepoRoot = "E:\Parentheses\parentheses-0.9"
Set-Location $RepoRoot

$Lock = "$RepoRoot\out\locks\GPU.lock"
$PythonExe = "$RepoRoot\.venv\Scripts\python.exe"
$DataDir = "$RepoRoot\data\processed_sft_240m_s10_patch"
$ResumeCkpt = "$RepoRoot\checkpoints\parentheses-1.0-240M-s10-blend\sft-winner.pt"
$OutDir = "I:\parentheses\checkpoints\parentheses-1.0-240M-s10-patch"

if (Test-Path $Lock) {
    Write-Error "GPU lock already exists: $(Get-Content $Lock)"
}

if (!(Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
}

$LogFile = "$RepoRoot\out\s10\micropatch_train.log"
"antigravity session10-micropatch $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss IST')" | Out-File -FilePath $Lock -Encoding utf8
Write-Host "[*] Acquired GPU.lock for S10 Micro-Patch"

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
        --lr 1.5e-5 `
        --min-lr 5e-6 `
        --warmup-steps 10 `
        --decay-start 14010 `
        --decay-end 14200 `
        --eos-weight 3.0 `
        --differential-adamw `
        --recurrent-lr-ratio 0.1 `
        --grad-checkpointing `
        --max-steps 14200 `
        --log-every 1 `
        --ckpt-every 50 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $LogFile
    
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        throw "train.py micro-patch exited with code $code"
    }
    Write-Host "[+] S10 Micro-Patch training completed successfully."
} finally {
    if (Test-Path $Lock) {
        Remove-Item $Lock -Force -ErrorAction SilentlyContinue
        Write-Host "[*] Released GPU.lock"
    }
}
