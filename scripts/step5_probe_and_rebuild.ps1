# Step 5: Probe base on NQ-open, generate rtuning.jsonl, and rebuild S10 SFT corpus
# Engineered by uncoalesced

$ErrorActionPreference = "Stop"
$RepoRoot = "E:\Parentheses\parentheses-0.9"
Set-Location $RepoRoot

$Lock = Join-Path $RepoRoot "out\locks\GPU.lock"
if (Test-Path $Lock) { throw "[FAIL] GPU.lock held: $(Get-Content $Lock)" }
"s10-probe-rtuning $(Get-Date -Format s) pid=$PID" | Out-File -Encoding utf8 $Lock

$outLog = Join-Path $RepoRoot "out\s10\step5_probe_and_rebuild.log"

try {
    Write-Host "[*] Probing base model on NQ-open questions (limit 5000)..." -ForegroundColor Cyan
    & .venv\Scripts\python.exe scripts\s10_probe_rtuning.py `
        --checkpoint "checkpoints\parentheses-1.0-240M-s8-golden\parentheses-1.0-240M-s6-s8-blend-best.pt" `
        --limit 5000 `
        --device cuda 2>&1 | Tee-Object -FilePath $outLog
} finally {
    Remove-Item $Lock -Force -ErrorAction SilentlyContinue
}

Write-Host "[*] Rebuilding S10 SFT 65M-token corpus with rtuning rows..." -ForegroundColor Cyan
& .venv\Scripts\python.exe scripts\build_s10_sft_240m.py 2>&1 | Tee-Object -FilePath $outLog -Append
Write-Host "[+] Step 5 Complete!" -ForegroundColor Green
