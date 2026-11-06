# Engineered by uncoalesced
<#
.SYNOPSIS
    Supervised orchestrator for Parentheses 1.0-240M Session 7.5 Precision Calibrated SFT (steps 12,001 -> 13,000).
.DESCRIPTION
    1. Telemetry watchdog   (telemetry_watchdog_240m.py, background job)
    2. Checkpoint archiver  (checkpoint_manager_s3.py with S7.5 dirs/manifest/milestones 12250,12500,12750,13000)
    3. Training             (launch_s7_5_240m.ps1; holds out\locks\GPU.lock while training)
    4. Archiver drain: wait until latest.pt is a complete copy of step_13000_final.pt.
    5. Post chain on success, written to out\eval_s7_5\ :
         - SWA-3 checkpoint merge across step_12500.pt, step_12750.pt, step_13000_final.pt -> step_13000_swa3.pt
         - 60-probe adversarial suite at --max-tokens 256 (greedy + sampled + SWA-3 greedy)
         - score_60p.py mechanical scoring against out\eval_s6_12k\s6_12000_chat_greedy.jsonl
         - bpb_slab_v4.py across S6 (12000), S7.5 (13000), and S7.5 SWA-3
         - bpb_gate.py drift check (<= +0.06)
    Run: powershell -ExecutionPolicy Bypass -File scripts\run_session7_5_supervised.ps1
#>

$ErrorActionPreference = "Stop"
$RepoRoot = "E:\Parentheses\parentheses-0.9"
Set-Location $RepoRoot
$env:PYTHONIOENCODING = "utf-8"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$LogPath = Join-Path $RepoRoot "out\session7_5_240m_live.log"
$EvalDir = Join-Path $RepoRoot "out\eval_s7_5"
$S6Anchor = Join-Path $RepoRoot "checkpoints\parentheses-1.0-240M-s6\step_12000.pt"
$S75CkptDir = Join-Path $RepoRoot "checkpoints\parentheses-1.0-240M-s7_5"
$S75Latest = Join-Path $S75CkptDir "latest.pt"
$S75Final = Join-Path $S75CkptDir "step_13000_final.pt"
$S75Swa3 = Join-Path $S75CkptDir "step_13000_swa3.pt"
$Lock = Join-Path $RepoRoot "out\locks\GPU.lock"

Write-Host "=== PARENTHESES 1.0-240M -- SESSION 7.5 PRECISION CALIBRATED SFT SUPERVISED RUN ===" -ForegroundColor Cyan

Write-Host "[1/4] Telemetry watchdog..." -ForegroundColor Yellow
$telemetryJob = Start-Job -ScriptBlock {
    param($root, $py, $log)
    Set-Location $root
    & $py "scripts\telemetry_watchdog_240m.py" --log-path $log --max-steps 13001 --interval 15 2>&1 | ForEach-Object { "$_" } | Out-File -Encoding utf8 "out\telemetry_watchdog_s7_5.log"
} -ArgumentList $RepoRoot, $Python, $LogPath

Write-Host "[2/4] Checkpoint archiver (E: -> I:, SHA256 parity before any prune)..." -ForegroundColor Yellow
$archiverJob = Start-Job -ScriptBlock {
    param($root, $py)
    Set-Location $root
    & $py "scripts\checkpoint_manager_s3.py" `
        --source-dir "checkpoints\parentheses-1.0-240M-s7_5" `
        --archive-dir "I:\parentheses\checkpoints\parentheses-1.0-240M-s7_5" `
        --manifest-name "MANIFEST_s7_5_sha256.tsv" `
        --milestones "12250,12500,12750,13000" `
        --interval 20 2>&1 | ForEach-Object { "$_" } | Out-File -Encoding utf8 "out\checkpoint_manager_s7_5.log"
} -ArgumentList $RepoRoot, $Python

Write-Host "[3/4] Training (log: $LogPath)..." -ForegroundColor Green
$trainExitCode = 0
try {
    powershell -ExecutionPolicy Bypass -File "scripts\launch_s7_5_240m.ps1"
    $trainExitCode = $LASTEXITCODE
} catch {
    Write-Host "[!] Exception during S7.5 training: $_" -ForegroundColor Red
    $trainExitCode = 1
}
Write-Host "[training] exit code $trainExitCode" -ForegroundColor Cyan

# Wait for archiver to copy and verify final checkpoint
$latestOk = $false
if ($trainExitCode -eq 0) {
    $deadline = (Get-Date).AddSeconds(1200)
    while ((Get-Date) -lt $deadline) {
        if ((Test-Path $S75Final) -and (Test-Path $S75Latest)) {
            $fin = Get-Item $S75Final
            $lat = Get-Item $S75Latest
            if ($lat.Length -eq $fin.Length -and $lat.LastWriteTimeUtc -eq $fin.LastWriteTimeUtc) { $latestOk = $true; break }
        }
        if ($archiverJob.State -ne "Running") { Write-Host "[!] Archiver job state $($archiverJob.State)" -ForegroundColor Red; break }
        Wait-Job $archiverJob -Timeout 30 | Out-Null
    }
}
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
"session7_5-post-chain $(Get-Date -Format s)" | Out-File -Encoding utf8 $Lock
try {
    # 0. Build SWA-3 checkpoint on CPU from milestones 12500, 12750, 13000_final
    $c12500 = Join-Path $S75CkptDir "step_12500.pt"
    $c12750 = Join-Path $S75CkptDir "step_12750.pt"
    if ((Test-Path $c12500) -and (Test-Path $c12750) -and (Test-Path $S75Final)) {
        Write-Host "  merging SWA-3 checkpoint (12500 + 12750 + 13000_final)..."
        & $Python scripts\merge_swa_checkpoints.py --ckpts $c12500 $c12750 $S75Final --out $S75Swa3 2>&1 | ForEach-Object { "$_" } | Out-File -Encoding utf8 "$EvalDir\swa3_merge.log"
    }

    # 1. Final checkpoint greedy, sampled T=0.7 x 5, and SWA-3 greedy
    Write-Host "  evaluating step 13000 final greedy..."
    & $Python scripts\eval_60p_adversarial_suite.py --checkpoint $S75Final --output "$EvalDir\s7_5_chat_greedy.jsonl" --mode chat --temp 0.0 --samples 1 --max-tokens 256 2>&1 | ForEach-Object { "$_" } | Out-File -Encoding utf8 "$EvalDir\eval_greedy.log"
    Write-Host "  evaluating step 13000 final sampled T=0.7 x 5..."
    & $Python scripts\eval_60p_adversarial_suite.py --checkpoint $S75Final --output "$EvalDir\s7_5_chat_t07x5.jsonl" --mode chat --temp 0.7 --samples 5 --seed 1234 --max-tokens 256 2>&1 | ForEach-Object { "$_" } | Out-File -Encoding utf8 "$EvalDir\eval_sampled.log"
    if (Test-Path $S75Swa3) {
        Write-Host "  evaluating step 13000 SWA-3 greedy..."
        & $Python scripts\eval_60p_adversarial_suite.py --checkpoint $S75Swa3 --output "$EvalDir\s7_5_swa3_chat_greedy.jsonl" --mode chat --temp 0.0 --samples 1 --max-tokens 256 2>&1 | ForEach-Object { "$_" } | Out-File -Encoding utf8 "$EvalDir\eval_swa3_greedy.log"
    }

    # 2. Mechanical scoring across S6 (12000), S7.5 (13000), and S7.5 SWA-3
    Write-Host "  running score_60p..."
    if (Test-Path $S75Swa3) {
        & $Python scripts\score_60p.py "out\eval_s6_12k\s6_12000_chat_greedy.jsonl" "$EvalDir\s7_5_chat_greedy.jsonl" "$EvalDir\s7_5_swa3_chat_greedy.jsonl" "$EvalDir\s7_5_chat_t07x5.jsonl" --json "$EvalDir\scores.json" 2>&1 | ForEach-Object { "$_" } | Out-File -Encoding utf8 "$EvalDir\scores.txt"
    } else {
        & $Python scripts\score_60p.py "out\eval_s6_12k\s6_12000_chat_greedy.jsonl" "$EvalDir\s7_5_chat_greedy.jsonl" "$EvalDir\s7_5_chat_t07x5.jsonl" --json "$EvalDir\scores.json" 2>&1 | ForEach-Object { "$_" } | Out-File -Encoding utf8 "$EvalDir\scores.txt"
    }

    # 3. BPB evaluation on S6, S7.5 final, and S7.5 SWA-3
    Write-Host "  running bpb_slab_v4..."
    if (Test-Path $S75Swa3) {
        & $Python scripts\bpb_slab_v4.py --checkpoints $S6Anchor $S75Final $S75Swa3 --out "$EvalDir\bpb_240m.json" --device cuda 2>&1 | ForEach-Object { "$_" } | Out-File -Encoding utf8 "$EvalDir\bpb.log"
    } else {
        & $Python scripts\bpb_slab_v4.py --checkpoints $S6Anchor $S75Final --out "$EvalDir\bpb_240m.json" --device cuda 2>&1 | ForEach-Object { "$_" } | Out-File -Encoding utf8 "$EvalDir\bpb.log"
    }

    # 4. BPB gate with step 12000 baseline
    Write-Host "  evaluating bpb_gate..."
    & $Python scripts\bpb_gate.py "$EvalDir\bpb_240m.json" --max-rise 0.06 2>&1 | ForEach-Object { "$_" } | Out-File -Encoding utf8 "$EvalDir\bpb_gate.txt"
} finally {
    Remove-Item $Lock -Force -ErrorAction SilentlyContinue
}
Write-Host "[+] Session 7.5 complete. Results: $EvalDir\scores.txt and $EvalDir\bpb_gate.txt" -ForegroundColor Green
