# Engineered by uncoalesced
<#
.SYNOPSIS
    Automated Watcher and Pipeline Orchestrator from Session 7.5 to Session 8.
.DESCRIPTION
    1. Monitors active S7.5 training and built-in post-chain until completion.
    2. Runs scripts\eval_s7_5_gate_verdict.py to audit Gates G1-G5.
    3. If all gates pass:
         - Selects candidate S8 base
         - Logs approval to COORDINATION_240M.md
         - Executes S8 On-Policy DPO v2 + SWA-3 end-to-end
         - Runs S8 evaluations and checkpoint archival
         - Generates S8 morning handoff report
    4. If any gate fails:
         - Logs failure details and halts immediately
#>

$ErrorActionPreference = "Stop"
$RepoRoot = "E:\Parentheses\parentheses-0.9"
Set-Location $RepoRoot
$env:PYTHONIOENCODING = "utf-8"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Lock = Join-Path $RepoRoot "out\locks\GPU.lock"
$S75EvalDir = Join-Path $RepoRoot "out\eval_s7_5"
$S75BpbGate = Join-Path $S75EvalDir "bpb_gate.txt"
$S75Scores = Join-Path $S75EvalDir "scores.json"
$CoordLog = Join-Path $RepoRoot "out\swarm\COORDINATION_240M.md"
$S8OutDir = Join-Path $RepoRoot "checkpoints\parentheses-1.0-240M-s8-dpo-swa3"
$S8EvalDir = Join-Path $RepoRoot "out\eval_s8"

Write-Host "=== PARENTHESES 1.0-240M -- S7.5 TO S8 AUTOMATED SENTINEL ARMED ===" -ForegroundColor Cyan
Write-Host "[*] Waiting for S7.5 training and post-chain completion..."

# 1. Wait loop for S7.5 completion
$pollingInterval = 30
while ($true) {
    Start-Sleep -Seconds $pollingInterval
    $liveLog = Join-Path $RepoRoot "out\session7_5_240m_live.log"
    if (Test-Path $liveLog) {
        $lastLine = Get-Content $liveLog -Tail 1 -ErrorAction SilentlyContinue
        Write-Host "[S7.5 live] $lastLine" -ForegroundColor DarkGray
    }

    # S7.5 is complete when bpb_gate.txt exists and GPU.lock is cleared
    if ((Test-Path $S75BpbGate) -and (Test-Path $S75Scores) -and (-not (Test-Path $Lock))) {
        Write-Host "[+] Detected S7.5 post-chain completion and released GPU.lock!" -ForegroundColor Green
        break
    }
}

# 2. Gate Verification (G1 to G5)
Write-Host "[*] Evaluating Gates G1-G5 via scripts\eval_s7_5_gate_verdict.py..." -ForegroundColor Yellow
$gateExitCode = 1
try {
    & $Python scripts\eval_s7_5_gate_verdict.py 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath "$S75EvalDir\gate_verdict_run.log"
    $gateExitCode = $LASTEXITCODE
} catch {
    Write-Host "[!] Exception in gate evaluation: $_" -ForegroundColor Red
    $gateExitCode = 1
}

$verdictJsonPath = Join-Path $S75EvalDir "gate_verdict.json"
if (-not (Test-Path $verdictJsonPath) -or $gateExitCode -ne 0) {
    Write-Host "[FAIL] S7.5 did not pass all gates (Exit Code $gateExitCode). S8 trigger aborted per protocol." -ForegroundColor Red
    $now = Get-Date -Format "yyyy-MM-dd HH:mm"
    Add-Content -Path $CoordLog -Value "`n### $now IST -- Antigravity Sentinel: S7.5 GATE CHECK FAILED -- S8 HALTED`n- Gate check exit code: $gateExitCode`n- See $S75EvalDir\gate_verdict_run.log for details.`n"
    exit 1
}

$verdict = Get-Content $verdictJsonPath -Raw | ConvertFrom-Json
$candidateBase = $verdict.candidate_base

Write-Host "[PASS] All S7.5 Gates Passed! Candidate S8 Base: $candidateBase" -ForegroundColor Green
$now = Get-Date -Format "yyyy-MM-dd HH:mm"
Add-Content -Path $CoordLog -Value "`n### $now IST -- Antigravity Sentinel: ALL S7.5 GATES PASSED -- LAUNCHING S8 INSTANTLY 🚀`n- Candidate base: $candidateBase`n- Scores: Greedy $($verdict.scores.final_greedy), SWA-3 $($verdict.scores.swa3_greedy), Sampled pass@5 $($verdict.scores.sampled_pass5)`n- Launching On-Policy DPO v2 + SWA-3 pipeline on RTX 5050...`n"

# 3. Launch Session 8 Pipeline
Write-Host "=== LAUNCHING SESSION 8 ON-POLICY DPO v2 + SWA-3 ===" -ForegroundColor Cyan
"session8-dpo $(Get-Date -Format s) pid=$PID" | Out-File -Encoding utf8 $Lock

try {
    # 3.1 Pair Generation
    Write-Host "[S8 1/5] Generating on-policy pairs (--mode generate)..." -ForegroundColor Yellow
    & $Python scripts\train_dpo_v2_apache.py --mode generate --sft-ckpt $candidateBase 2>&1 | Tee-Object -FilePath "out\s8_generate.log"
    if ($LASTEXITCODE -ne 0) { throw "[FAIL] S8 generate failed with code $LASTEXITCODE" }

    # 3.2 Precompute Reference Log-Probs
    Write-Host "[S8 2/5] Precomputing reference log-probs (--mode precompute-ref)..." -ForegroundColor Yellow
    & $Python scripts\train_dpo_v2_apache.py --mode precompute-ref --sft-ckpt $candidateBase 2>&1 | Tee-Object -FilePath "out\s8_precompute.log"
    if ($LASTEXITCODE -ne 0) { throw "[FAIL] S8 precompute-ref failed with code $LASTEXITCODE" }

    # 3.3 Smoke Test (2 steps)
    Write-Host "[S8 3/5] Running smoke test (2 steps)..." -ForegroundColor Yellow
    & $Python scripts\train_dpo_v2_apache.py --mode train --sft-ckpt $candidateBase --max-steps 2 --save-every 1000 --out-dir "checkpoints\_s8_smoke" 2>&1 | Tee-Object -FilePath "out\s8_smoke.log"
    if ($LASTEXITCODE -ne 0) { throw "[FAIL] S8 smoke test failed with code $LASTEXITCODE" }

    # 3.4 Full DPO Training (150 steps)
    Write-Host "[S8 4/5] Training DPO (150 steps, lr=5e-7, beta=0.1, nll=0.1)..." -ForegroundColor Green
    & $Python scripts\train_dpo_v2_apache.py --mode train --sft-ckpt $candidateBase --max-steps 150 --save-every 50 --lr 5e-7 --beta 0.1 --nll-coeff 0.1 --out-dir $S8OutDir 2>&1 | Tee-Object -FilePath "out\s8_train.log"
    if ($LASTEXITCODE -ne 0) { throw "[FAIL] S8 training failed with code $LASTEXITCODE" }

    # 3.5 SWA-3 Checkpoint Merge
    Write-Host "[S8 5/5] Building SWA-3 golden checkpoint (--mode swa3)..." -ForegroundColor Green
    & $Python scripts\train_dpo_v2_apache.py --mode swa3 --sft-ckpt $candidateBase --out-dir $S8OutDir 2>&1 | Tee-Object -FilePath "out\s8_swa3.log"
    if ($LASTEXITCODE -ne 0) { throw "[FAIL] S8 swa3 failed with code $LASTEXITCODE" }

    # 4. S8 Evaluation Suite
    Write-Host "=== RUNNING S8 EVALUATION SUITE ===" -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path $S8EvalDir | Out-Null
    $goldenCkpt = Join-Path $S8OutDir "parentheses-1.0-240M-dpo-swa3-golden.pt"

    if (Test-Path $goldenCkpt) {
        Write-Host "  evaluating S8 golden greedy..."
        & $Python scripts\eval_60p_adversarial_suite.py --checkpoint $goldenCkpt --output "$S8EvalDir\s8_golden_chat_greedy.jsonl" --mode chat --temp 0.0 --samples 1 --max-tokens 256 2>&1 | Out-File -Encoding utf8 "$S8EvalDir\eval_greedy.log"

        Write-Host "  evaluating S8 golden sampled T=0.7 x 5..."
        & $Python scripts\eval_60p_adversarial_suite.py --checkpoint $goldenCkpt --output "$S8EvalDir\s8_golden_chat_t07x5.jsonl" --mode chat --temp 0.7 --samples 5 --seed 1234 --max-tokens 256 2>&1 | Out-File -Encoding utf8 "$S8EvalDir\eval_sampled.log"

        Write-Host "  scoring 60-probe..."
        & $Python scripts\score_60p.py "out\eval_s6_12k\s6_12000_chat_greedy.jsonl" "$S75EvalDir\s7_5_chat_greedy.jsonl" "$S8EvalDir\s8_golden_chat_greedy.jsonl" "$S8EvalDir\s8_golden_chat_t07x5.jsonl" --json "$S8EvalDir\scores.json" 2>&1 | Out-File -Encoding utf8 "$S8EvalDir\scores.txt"

        Write-Host "  evaluating BPB..."
        $S6Anchor = Join-Path $RepoRoot "checkpoints\parentheses-1.0-240M-s6\step_12000.pt"
        & $Python scripts\bpb_slab_v4.py --checkpoints $S6Anchor $candidateBase $goldenCkpt --out "$S8EvalDir\bpb_240m.json" --device cuda 2>&1 | Out-File -Encoding utf8 "$S8EvalDir\bpb.log"

        Write-Host "  evaluating BPB gate..."
        & $Python scripts\bpb_gate.py "$S8EvalDir\bpb_240m.json" --max-rise 0.06 2>&1 | Out-File -Encoding utf8 "$S8EvalDir\bpb_gate.txt"
    }

    # 5. Archive to I:
    $ArchiveDir = "I:\parentheses\checkpoints\parentheses-1.0-240M-s8"
    New-Item -ItemType Directory -Force -Path $ArchiveDir | Out-Null
    Write-Host "[*] Archiving S8 checkpoints to $ArchiveDir..."
    Get-ChildItem -Path $S8OutDir -Filter "*.pt" | ForEach-Object {
        $dest = Join-Path $ArchiveDir $_.Name
        if (-not (Test-Path $dest)) {
            Copy-Item $_.FullName -Destination $dest
            $hash = (Get-FileHash -Algorithm SHA256 $dest).Hash
            "$hash`t$($_.Name)`t$($_.Length)`t$(Get-Date -Format s)" | Out-File -Append -Encoding utf8 "$ArchiveDir\MANIFEST_s8_sha256.tsv"
        }
    }

    $now = Get-Date -Format "yyyy-MM-dd HH:mm"
    Add-Content -Path $CoordLog -Value "`n### $now IST -- Antigravity Sentinel: SESSION 8 COMPLETED & EVALUATED 🏆`n- Golden Checkpoint: $goldenCkpt`n- Evaluation results in $S8EvalDir`n- Ready for Joel's final morning certification review.`n"
    Write-Host "[+] Session 8 Pipeline Complete!" -ForegroundColor Green

} finally {
    Remove-Item $Lock -Force -ErrorAction SilentlyContinue
}
