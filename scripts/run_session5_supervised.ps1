<#
.SYNOPSIS
    Supervised orchestrator for Parentheses 1.0-240M Session 5 (steps 8,001 -> 10,000).
.DESCRIPTION
    1. Telemetry watchdog   (telemetry_watchdog_240m.py, background job)
    2. Checkpoint archiver  (checkpoint_manager_s3.py with S5 dirs/manifest/milestones, background job)
    3. Training             (launch_s5_240m.ps1; it holds out\locks\GPU.lock while training)
    4. Archiver drain: wait (<= 1200 s) until latest.pt is a complete copy of step_10000_final.pt,
       then stop the jobs. S4 slept a fixed 300 s and Stop-Job killed the archiver mid-copy,
       leaving latest.pt partially written.
    5. Post chain on success, written to out\eval_s5\ :
         - 60-probe suite on the S5 checkpoint: chat greedy + chat T=0.7 x5 (seed 1234), 256 max tokens
         - score_60p.py (mechanical scores) over S4 + S5 runs -> scores.json
         - bpb_slab_v4.py on S4 and S5 checkpoints -> bpb_240m.json
    Run: powershell -ExecutionPolicy Bypass -File scripts\run_session5_supervised.ps1
#>

$ErrorActionPreference = "Stop"
$RepoRoot = "E:\Parentheses\parentheses-0.9"
Set-Location $RepoRoot
$env:PYTHONIOENCODING = "utf-8"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$LogPath = Join-Path $RepoRoot "out\session5_240m_live.log"
$EvalDir = Join-Path $RepoRoot "out\eval_s5"
$S4Ckpt = Join-Path $RepoRoot "checkpoints\parentheses-1.0-240M-s4\step_8000_final.pt"
$S5Ckpt = Join-Path $RepoRoot "checkpoints\parentheses-1.0-240M-s5\latest.pt"
$S5Final = Join-Path $RepoRoot "checkpoints\parentheses-1.0-240M-s5\step_10000_final.pt"
$Lock = Join-Path $RepoRoot "out\locks\GPU.lock"

Write-Host "=== PARENTHESES 1.0-240M -- SESSION 5 SUPERVISED RUN ===" -ForegroundColor Cyan

Write-Host "[1/4] Telemetry watchdog..." -ForegroundColor Yellow
$telemetryJob = Start-Job -ScriptBlock {
    param($root, $py, $log)
    Set-Location $root
    & $py "scripts\telemetry_watchdog_240m.py" --log-path $log --max-steps 10001 --interval 15 *> "out\telemetry_watchdog_s5.log"
} -ArgumentList $RepoRoot, $Python, $LogPath

Write-Host "[2/4] Checkpoint archiver (E: -> I:, SHA256 parity before any prune)..." -ForegroundColor Yellow
$archiverJob = Start-Job -ScriptBlock {
    param($root, $py)
    Set-Location $root
    & $py "scripts\checkpoint_manager_s3.py" `
        --source-dir "checkpoints\parentheses-1.0-240M-s5" `
        --archive-dir "I:\parentheses\checkpoints\parentheses-1.0-240M-s5" `
        --manifest-name "MANIFEST_s5_sha256.tsv" `
        --milestones "9000,10000" `
        --interval 20 *> "out\checkpoint_manager_s5.log"
} -ArgumentList $RepoRoot, $Python

# Warn-only replay gate (Joel, 2026-11-03): BPB may drift at most 0.06 above step 8000 per language.
# BPB can't run while the GPU trains, so the in-run tripwire is val_v5 (S4 end 3.3849) + 0.05 nats.
# A breach writes out\session5_gate_warn.txt and pings Antigravity once; training continues.
# The real per-language BPB check runs in the post chain (bpb_gate.txt).
Write-Host "[2b/4] val_v5 gate watcher (warn only, threshold 3.4349)..." -ForegroundColor Yellow
$gateJob = Start-Job -ScriptBlock {
    param($root, $log)
    $threshold = 3.4349
    $warnFile = Join-Path $root "out\session5_gate_warn.txt"
    $coord = Join-Path $root "out\swarm\COORDINATION_240M.md"
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    $warned = $false
    while ($true) {
        Start-Sleep -Seconds 300
        if (-not (Test-Path $log)) { continue }
        $last = Get-Content $log -Tail 400 | Select-String "val_v5 ([0-9.]+)" | Select-Object -Last 1
        if (-not $last) { continue }
        $v = [double]$last.Matches[0].Groups[1].Value
        if ($v -gt $threshold -and -not $warned) {
            $msg = "`n- $(Get-Date -Format s) [S5 GATE WARN] val_v5 $v > $threshold (S4 end 3.3849 + 0.05). Warn-only; training continues. Line: $($last.Line.Trim())`n"
            [IO.File]::AppendAllText($warnFile, $msg, $utf8)
            if (Test-Path $coord) { [IO.File]::AppendAllText($coord, $msg, $utf8) }
            $warned = $true
        }
    }
} -ArgumentList $RepoRoot, $LogPath

Write-Host "[3/4] Training (log: $LogPath)..." -ForegroundColor Green
$trainExitCode = 0
try {
    powershell -ExecutionPolicy Bypass -File "scripts\launch_s5_240m.ps1"
    $trainExitCode = $LASTEXITCODE
} catch {
    Write-Host "[!] Exception during training: $_" -ForegroundColor Red
    $trainExitCode = 1
}
Write-Host "[training] exit code $trainExitCode" -ForegroundColor Cyan

# Wait for the archiver to copy + verify the final checkpoint and refresh latest.pt before stopping it.
# checkpoint_manager_s3.py loops forever, so Wait-Job alone never returns early: wait in 30 s Wait-Job
# slices (deadline 1200 s) and stop as soon as latest.pt matches the final checkpoint. shutil.copy2
# stamps the source mtime only after the data copy, so size + mtime equal means the copy finished.
$latestOk = $false
if ($trainExitCode -eq 0) {
    $deadline = (Get-Date).AddSeconds(1200)
    while ((Get-Date) -lt $deadline) {
        if ((Test-Path $S5Final) -and (Test-Path $S5Ckpt)) {
            $fin = Get-Item $S5Final
            $lat = Get-Item $S5Ckpt
            if ($lat.Length -eq $fin.Length -and $lat.LastWriteTimeUtc -eq $fin.LastWriteTimeUtc) { $latestOk = $true; break }
        }
        if ($archiverJob.State -ne "Running") { Write-Host "[!] Archiver job state $($archiverJob.State)" -ForegroundColor Red; break }
        Wait-Job $archiverJob -Timeout 30 | Out-Null
    }
}
foreach ($j in $telemetryJob, $archiverJob, $gateJob) {
    Stop-Job $j -ErrorAction SilentlyContinue | Out-Null
    Remove-Job $j -ErrorAction SilentlyContinue | Out-Null
}

if ($trainExitCode -ne 0) {
    Write-Host "[!] Training did not exit cleanly. Post chain skipped." -ForegroundColor Red
    exit $trainExitCode
}
if (-not $latestOk) {
    $finLen = if (Test-Path $S5Final) { (Get-Item $S5Final).Length } else { "missing" }
    $latLen = if (Test-Path $S5Ckpt) { (Get-Item $S5Ckpt).Length } else { "missing" }
    Write-Host "[!] latest.pt ($latLen) does not match step_10000_final.pt ($finLen) after 1200 s. Post chain skipped; see out\checkpoint_manager_s5.log." -ForegroundColor Red
    exit 2
}
Write-Host "[archiver] latest.pt verified = step_10000_final.pt ($((Get-Item $S5Ckpt).Length) bytes)" -ForegroundColor Cyan

Write-Host "[4/4] Post chain -> $EvalDir" -ForegroundColor Green
New-Item -ItemType Directory -Force -Path $EvalDir | Out-Null
if (Test-Path $Lock) { throw "[FAIL] GPU.lock held after training: $(Get-Content $Lock)" }
"session5-post-chain $(Get-Date -Format s)" | Out-File -Encoding utf8 $Lock
try {
    & $Python scripts\eval_60p_adversarial_suite.py --checkpoint $S5Ckpt --output "$EvalDir\s5_chat_greedy.jsonl" --mode chat --temp 0.0 --samples 1 --max-tokens 512 *> "$EvalDir\eval_greedy.log"
    & $Python scripts\eval_60p_adversarial_suite.py --checkpoint $S5Ckpt --output "$EvalDir\s5_chat_t07x5.jsonl" --mode chat --temp 0.7 --samples 5 --seed 1234 --max-tokens 512 *> "$EvalDir\eval_sampled.log"
    # S4 greedy at 256 tokens matches S5's horizon; S4 sampled exists only at 80 tokens (not comparable).
    & $Python scripts\score_60p.py `
        out\eval_s4\s4_chat_greedy_256.jsonl `
        "$EvalDir\s5_chat_greedy.jsonl" "$EvalDir\s5_chat_t07x5.jsonl" `
        --json "$EvalDir\scores.json" *> "$EvalDir\scores.txt"
    & $Python scripts\bpb_slab_v4.py --checkpoints $S4Ckpt $S5Ckpt --out "$EvalDir\bpb_240m.json" *> "$EvalDir\bpb.log"
    # BPB gate, warn only: any language more than 0.06 above step 8000.
    & $Python scripts\bpb_gate.py "$EvalDir\bpb_240m.json" --max-rise 0.06 *> "$EvalDir\bpb_gate.txt"
} finally {
    Remove-Item $Lock -Force -ErrorAction SilentlyContinue
}
Write-Host "[+] Session 5 complete. Numbers: $EvalDir\scores.txt and $EvalDir\bpb.log" -ForegroundColor Green
