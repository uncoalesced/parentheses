$ErrorActionPreference = "Stop"
$RepoRoot = "E:\Parentheses\parentheses-0.9"
Set-Location $RepoRoot

$Python = "$RepoRoot\.venv\Scripts\python.exe"
$Ckpt = "$RepoRoot\checkpoints\parentheses-1.0-240M-s5\latest.pt"
$EvalDir = "$RepoRoot\out\eval_s5"
$Lock = "$RepoRoot\out\locks\GPU.lock"

if (Test-Path $Lock) {
    throw "[FAIL] GPU.lock is currently held: $(Get-Content $Lock)"
}

"s5-eval-512 $(Get-Date -Format s)" | Out-File -Encoding utf8 $Lock

try {
    Write-Host "[*] Running 60-Probe Adversarial Suite (max-tokens 512)..." -ForegroundColor Cyan
    & $Python scripts\eval_60p_adversarial_suite.py `
        --checkpoint $Ckpt `
        --output "$EvalDir\s5_chat_greedy_512.jsonl" `
        --mode chat --temp 0.0 --samples 1 --max-tokens 512 `
        *> "$EvalDir\eval_greedy_512.log"

    Write-Host "[*] Scoring 512-token outputs with score_60p.py..." -ForegroundColor Cyan
    & $Python scripts\score_60p.py `
        out\eval_s4\s4_chat_greedy_256.jsonl `
        "$EvalDir\s5_chat_greedy.jsonl" `
        "$EvalDir\s5_chat_greedy_512.jsonl" `
        --json "$EvalDir\scores_512.json" `
        *> "$EvalDir\scores_512.txt"

    Write-Host "[+] 512-token evaluation finished successfully." -ForegroundColor Green
} finally {
    Remove-Item $Lock -Force -ErrorAction SilentlyContinue
}
