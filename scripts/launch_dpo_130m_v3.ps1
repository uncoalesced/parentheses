# scripts/launch_dpo_130m_v3.ps1
# Automated Launch Wrapper for Path 1: Calibrated DPO Alignment on Parentheses 0.9-130M-v3

$ErrorActionPreference = "Stop"
$RepoRoot = "E:\Parentheses\parentheses-0.9"
Set-Location $RepoRoot

$LockFile = "$RepoRoot\out\locks\GPU.lock"
$LockDir = Split-Path $LockFile
if (-not (Test-Path $LockDir)) {
    New-Item -ItemType Directory -Path $LockDir -Force | Out-Null
}

$PyExe = "$RepoRoot\.venv\Scripts\python.exe"
if (-not (Test-Path $PyExe)) {
    $PyExe = "python.exe"
}

$PolicyCkpt = "$RepoRoot\checkpoints\parentheses-0.9-130M-v3-sft\step_2000_sft_swa3.pt"
$PairsFile = "$RepoRoot\out\dpo_pairs_balanced.jsonl"
$OutDir = "$RepoRoot\checkpoints\parentheses-0.9-130M-v3-dpo-calibrated"

Write-Host "=== Launching Path 1: Calibrated DPO for Parentheses 0.9-130M-v3 ==="
Write-Host "Policy:  $PolicyCkpt"
Write-Host "Pairs:   $PairsFile"
Write-Host "OutDir:  $OutDir"

# Acquire GPU.lock
"parentheses-130m-v3-dpo (PID $PID, started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))" | Set-Content -Path $LockFile -Encoding utf8
Write-Host "Acquired GPU.lock."

try {
    & $PyExe "scripts\train_dpo_130m_v3.py" `
        --policy $PolicyCkpt `
        --pairs $PairsFile `
        --out-dir $OutDir `
        --steps 100 `
        --batch-size 4 `
        --lr 5e-7 `
        --beta 0.1 `
        --sft-coef 0.3 `
        --max-len 1024 `
        --device cuda `
        --log-every 10
    Write-Host "=== DPO Alignment Sprint Finished Successfully! ==="
} finally {
    if (Test-Path $LockFile) {
        Remove-Item $LockFile -Force
        Write-Host "Released GPU.lock."
    }
}
