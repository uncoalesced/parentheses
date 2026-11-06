<#
.SYNOPSIS
    Supervised orchestrator for Parentheses 1.0-240M Session 4 (steps 6,001 -> 8,000).
.DESCRIPTION
    1. Telemetry watchdog   (telemetry_watchdog_240m.py, background job)
    2. Checkpoint archiver  (checkpoint_manager_s3.py with S4 dirs/manifest/milestones, background job)
    3. Training             (launch_s4_240m.ps1; it holds out\locks\GPU.lock while training)
    4. Post chain on success, written to out\eval_s4\ :
         - 60-probe suite on the S4 checkpoint: chat greedy + chat T=0.7 x5 (seed 1234)
         - score_60p.py (mechanical scores) over S3 rebaseline + S4 runs -> scores.json
         - bpb_slab_v4.py on S3 and S4 checkpoints -> bpb_240m.json
    Run: powershell -ExecutionPolicy Bypass -File scripts\run_session4_supervised.ps1
#>

$ErrorActionPreference = "Stop"
$RepoRoot = "E:\Parentheses\parentheses-0.9"
Set-Location $RepoRoot
$env:PYTHONIOENCODING = "utf-8"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$LogPath = Join-Path $RepoRoot "out\session4_240m_live.log"
$EvalDir = Join-Path $RepoRoot "out\eval_s4"
$S3Ckpt = Join-Path $RepoRoot "checkpoints\parentheses-1.0-240M-s3\latest.pt"
$S4Ckpt = Join-Path $RepoRoot "checkpoints\parentheses-1.0-240M-s4\latest.pt"
$Lock = Join-Path $RepoRoot "out\locks\GPU.lock"

Write-Host "=== PARENTHESES 1.0-240M -- SESSION 4 SUPERVISED RUN ===" -ForegroundColor Cyan

Write-Host "[1/4] Telemetry watchdog..." -ForegroundColor Yellow
$telemetryJob = Start-Job -ScriptBlock {
    param($root, $py, $log)
    Set-Location $root
    & $py "scripts\telemetry_watchdog_240m.py" --log-path $log --max-steps 8001 --interval 15 *> "out\telemetry_watchdog_s4.log"
} -ArgumentList $RepoRoot, $Python, $LogPath

Write-Host "[2/4] Checkpoint archiver (E: -> I:, SHA256 parity before any prune)..." -ForegroundColor Yellow
$archiverJob = Start-Job -ScriptBlock {
    param($root, $py)
    Set-Location $root
    & $py "scripts\checkpoint_manager_s3.py" `
        --source-dir "checkpoints\parentheses-1.0-240M-s4" `
        --archive-dir "I:\parentheses\checkpoints\parentheses-1.0-240M-s4" `
        --manifest-name "MANIFEST_s4_sha256.tsv" `
        --milestones "7000,8000" `
        --interval 20 *> "out\checkpoint_manager_s4.log"
} -ArgumentList $RepoRoot, $Python

Write-Host "[3/4] Training (log: $LogPath)..." -ForegroundColor Green
$trainExitCode = 0
try {
    powershell -ExecutionPolicy Bypass -File "scripts\launch_s4_240m.ps1"
    $trainExitCode = $LASTEXITCODE
} catch {
    Write-Host "[!] Exception during training: $_" -ForegroundColor Red
    $trainExitCode = 1
}
Write-Host "[training] exit code $trainExitCode" -ForegroundColor Cyan

# Give the archiver time to copy + verify the final checkpoint before stopping it.
Start-Sleep -Seconds 300
foreach ($j in $telemetryJob, $archiverJob) {
    Stop-Job $j -ErrorAction SilentlyContinue | Out-Null
    Remove-Job $j -ErrorAction SilentlyContinue | Out-Null
}

if ($trainExitCode -ne 0) {
    Write-Host "[!] Training did not exit cleanly. Post chain skipped." -ForegroundColor Red
    exit $trainExitCode
}

Write-Host "[4/4] Post chain -> $EvalDir" -ForegroundColor Green
New-Item -ItemType Directory -Force -Path $EvalDir | Out-Null
if (Test-Path $Lock) { throw "[FAIL] GPU.lock held after training: $(Get-Content $Lock)" }
"session4-post-chain $(Get-Date -Format s)" | Out-File -Encoding utf8 $Lock
try {
    & $Python scripts\eval_60p_adversarial_suite.py --checkpoint $S4Ckpt --output "$EvalDir\s4_chat_greedy.jsonl" --mode chat --temp 0.0 --samples 1 --max-tokens 80 *> "$EvalDir\eval_greedy.log"
    & $Python scripts\eval_60p_adversarial_suite.py --checkpoint $S4Ckpt --output "$EvalDir\s4_chat_t07x5.jsonl" --mode chat --temp 0.7 --samples 5 --seed 1234 --max-tokens 80 *> "$EvalDir\eval_sampled.log"
    & $Python scripts\score_60p.py `
        out\eval_rebaseline_60p\s3_chat_greedy.jsonl out\eval_rebaseline_60p\s3_chat_t07x5.jsonl `
        "$EvalDir\s4_chat_greedy.jsonl" "$EvalDir\s4_chat_t07x5.jsonl" `
        --json "$EvalDir\scores.json" *> "$EvalDir\scores.txt"
    & $Python scripts\bpb_slab_v4.py --checkpoints $S3Ckpt $S4Ckpt --out "$EvalDir\bpb_240m.json" *> "$EvalDir\bpb.log"
} finally {
    Remove-Item $Lock -Force -ErrorAction SilentlyContinue
}
Write-Host "[+] Session 4 complete. Numbers: $EvalDir\scores.txt and $EvalDir\bpb.log" -ForegroundColor Green
