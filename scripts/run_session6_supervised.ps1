# Engineered by uncoalesced
<#
.SYNOPSIS
    Supervised orchestrator for Parentheses 1.0-240M Session 6 (steps 10,001 -> 13,000).
.DESCRIPTION
    Per vault Briefs/BRIEF_SESSION_6_240M_DRAFT.md + PLAN_SESSION_6_240M_EXECUTION.md §3.
    1. Telemetry watchdog   (telemetry_watchdog_240m.py, background job)
    2. Checkpoint archiver  (checkpoint_manager_s3.py with S6 dirs/manifest/milestones 11000,12000,13000, background job)
    3. Gate watcher         (val_v5 threshold 3.3887 warn-only, background job)
    4. Training             (launch_s6_240m.ps1; holds out\locks\GPU.lock while training)
    5. Archiver drain: wait (<= 1200 s) until latest.pt is a complete copy of step_13000_final.pt.
    6. Post chain on success, written to out\eval_s6\ :
         - 60-probe suite at --max-tokens 256:
             * greedy on 11000, 12000, and 13000
             * sampled T=0.7 x 5 (seed 1234) on 13000
         - score_60p.py over S4 256, S5, and S6 runs -> scores.json / scores.txt
         - bpb_slab_v4.py on S5 and S6 checkpoints -> bpb_240m.json / bpb.log
         - bpb_gate.py with step 10000 baseline -> bpb_gate.txt
         - All stdout/stderr piped via | ForEach-Object { "$_" } | Out-File -Encoding utf8 (never *>)
    Run: powershell -ExecutionPolicy Bypass -File scripts\run_session6_supervised.ps1
#>

$ErrorActionPreference = "Stop"
$RepoRoot = "E:\Parentheses\parentheses-0.9"
Set-Location $RepoRoot
$env:PYTHONIOENCODING = "utf-8"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$LogPath = Join-Path $RepoRoot "out\session6_240m_live.log"
$EvalDir = Join-Path $RepoRoot "out\eval_s6"
$S4Ckpt = Join-Path $RepoRoot "checkpoints\parentheses-1.0-240M-s4\step_8000_final.pt"
$S5Final = Join-Path $RepoRoot "checkpoints\parentheses-1.0-240M-s5\step_10000_final.pt"
$S6CkptDir = Join-Path $RepoRoot "checkpoints\parentheses-1.0-240M-s6"
$S6Latest = Join-Path $S6CkptDir "latest.pt"
$S6Final = Join-Path $S6CkptDir "step_13000_final.pt"
$S6_11k = Join-Path $S6CkptDir "step_11000.pt"
$S6_12k = Join-Path $S6CkptDir "step_12000.pt"
$Lock = Join-Path $RepoRoot "out\locks\GPU.lock"

Write-Host "=== PARENTHESES 1.0-240M -- SESSION 6 SUPERVISED RUN ===" -ForegroundColor Cyan

Write-Host "[1/4] Telemetry watchdog..." -ForegroundColor Yellow
$telemetryJob = Start-Job -ScriptBlock {
    param($root, $py, $log)
    Set-Location $root
    & $py "scripts\telemetry_watchdog_240m.py" --log-path $log --max-steps 13001 --interval 15 2>&1 | ForEach-Object { "$_" } | Out-File -Encoding utf8 "out\telemetry_watchdog_s6.log"
} -ArgumentList $RepoRoot, $Python, $LogPath

Write-Host "[2/4] Checkpoint archiver (E: -> I:, SHA256 parity before any prune)..." -ForegroundColor Yellow
$archiverJob = Start-Job -ScriptBlock {
    param($root, $py)
    Set-Location $root
    & $py "scripts\checkpoint_manager_s3.py" `
        --source-dir "checkpoints\parentheses-1.0-240M-s6" `
        --archive-dir "I:\parentheses\checkpoints\parentheses-1.0-240M-s6" `
        --manifest-name "MANIFEST_s6_sha256.tsv" `
        --milestones "11000,12000,13000" `
        --interval 20 2>&1 | ForEach-Object { "$_" } | Out-File -Encoding utf8 "out\checkpoint_manager_s6.log"
} -ArgumentList $RepoRoot, $Python

# Warn-only replay gate (Brief §5, Plan §3): val_v5 threshold 3.3887 (S5 end 3.3387 + 0.05).
Write-Host "[2b/4] val_v5 gate watcher (warn only, threshold 3.3887)..." -ForegroundColor Yellow
$gateJob = Start-Job -ScriptBlock {
    param($root, $log)
    $threshold = 3.3887
    $warnFile = Join-Path $root "out\session6_gate_warn.txt"
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
            $msg = "`n- $(Get-Date -Format s) [S6 GATE WARN] val_v5 $v > $threshold (S5 end 3.3387 + 0.05). Warn-only; training continues. Line: $($last.Line.Trim())`n"
            [IO.File]::AppendAllText($warnFile, $msg, $utf8)
            if (Test-Path $coord) { [IO.File]::AppendAllText($coord, $msg, $utf8) }
            $warned = $true
        }
    }
} -ArgumentList $RepoRoot, $LogPath

Write-Host "[3/4] Training (log: $LogPath)..." -ForegroundColor Green
$trainExitCode = 0
try {
    powershell -ExecutionPolicy Bypass -File "scripts\launch_s6_240m.ps1"
    $trainExitCode = $LASTEXITCODE
} catch {
    Write-Host "[!] Exception during training: $_" -ForegroundColor Red
    $trainExitCode = 1
}
Write-Host "[training] exit code $trainExitCode" -ForegroundColor Cyan

# Wait for the archiver to copy + verify the final checkpoint and refresh latest.pt before stopping it.
$latestOk = $false
if ($trainExitCode -eq 0) {
    $deadline = (Get-Date).AddSeconds(1200)
    while ((Get-Date) -lt $deadline) {
        if ((Test-Path $S6Final) -and (Test-Path $S6Latest)) {
            $fin = Get-Item $S6Final
            $lat = Get-Item $S6Latest
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
    $finLen = if (Test-Path $S6Final) { (Get-Item $S6Final).Length } else { "missing" }
    $latLen = if (Test-Path $S6Latest) { (Get-Item $S6Latest).Length } else { "missing" }
    Write-Host "[!] latest.pt ($latLen) does not match step_13000_final.pt ($finLen) after 1200 s. Post chain skipped; see out\checkpoint_manager_s6.log." -ForegroundColor Red
    exit 2
}
Write-Host "[archiver] latest.pt verified = step_13000_final.pt ($((Get-Item $S6Latest).Length) bytes)" -ForegroundColor Cyan

Write-Host "[4/4] Post chain -> $EvalDir" -ForegroundColor Green
New-Item -ItemType Directory -Force -Path $EvalDir | Out-Null
if (Test-Path $Lock) { throw "[FAIL] GPU.lock held after training: $(Get-Content $Lock)" }
"session6-post-chain $(Get-Date -Format s)" | Out-File -Encoding utf8 $Lock
try {
    # 1. Milestone checkpoints 60-probe greedy at max-tokens 256
    if (Test-Path $S6_11k) {
        Write-Host "  evaluating step 11000 greedy..."
        & $Python scripts\eval_60p_adversarial_suite.py --checkpoint $S6_11k --output "$EvalDir\s6_11000_chat_greedy.jsonl" --mode chat --temp 0.0 --samples 1 --max-tokens 256 2>&1 | ForEach-Object { "$_" } | Out-File -Encoding utf8 "$EvalDir\eval_11k_greedy.log"
    }
    if (Test-Path $S6_12k) {
        Write-Host "  evaluating step 12000 greedy..."
        & $Python scripts\eval_60p_adversarial_suite.py --checkpoint $S6_12k --output "$EvalDir\s6_12000_chat_greedy.jsonl" --mode chat --temp 0.0 --samples 1 --max-tokens 256 2>&1 | ForEach-Object { "$_" } | Out-File -Encoding utf8 "$EvalDir\eval_12k_greedy.log"
    }

    # 2. Final checkpoint (13000) greedy and sampled T=0.7 x 5
    Write-Host "  evaluating step 13000 final greedy..."
    & $Python scripts\eval_60p_adversarial_suite.py --checkpoint $S6Final --output "$EvalDir\s6_chat_greedy.jsonl" --mode chat --temp 0.0 --samples 1 --max-tokens 256 2>&1 | ForEach-Object { "$_" } | Out-File -Encoding utf8 "$EvalDir\eval_greedy.log"
    Write-Host "  evaluating step 13000 final sampled T=0.7 x 5..."
    & $Python scripts\eval_60p_adversarial_suite.py --checkpoint $S6Final --output "$EvalDir\s6_chat_t07x5.jsonl" --mode chat --temp 0.7 --samples 5 --seed 1234 --max-tokens 256 2>&1 | ForEach-Object { "$_" } | Out-File -Encoding utf8 "$EvalDir\eval_sampled.log"

    # 3. Mechanical scoring across S4, S5, and S6
    Write-Host "  running score_60p..."
    $scoreArgs = @("out\eval_s4\s4_chat_greedy_256.jsonl", "out\eval_s5\s5_chat_greedy.jsonl", "$EvalDir\s6_chat_greedy.jsonl", "$EvalDir\s6_chat_t07x5.jsonl")
    if (Test-Path "$EvalDir\s6_11000_chat_greedy.jsonl") { $scoreArgs += "$EvalDir\s6_11000_chat_greedy.jsonl" }
    if (Test-Path "$EvalDir\s6_12000_chat_greedy.jsonl") { $scoreArgs += "$EvalDir\s6_12000_chat_greedy.jsonl" }
    $scoreArgs += @("--json", "$EvalDir\scores.json")
    & $Python scripts\score_60p.py @scoreArgs 2>&1 | ForEach-Object { "$_" } | Out-File -Encoding utf8 "$EvalDir\scores.txt"

    # 4. BPB evaluation on S5 and S6 final checkpoints
    Write-Host "  running bpb_slab_v4..."
    & $Python scripts\bpb_slab_v4.py --checkpoints $S5Final $S6Final --out "$EvalDir\bpb_240m.json" 2>&1 | ForEach-Object { "$_" } | Out-File -Encoding utf8 "$EvalDir\bpb.log"

    # 5. BPB gate with step 10000 baseline
    Write-Host "  evaluating bpb_gate..."
    & $Python scripts\bpb_gate.py "$EvalDir\bpb_240m.json" --max-rise 0.06 2>&1 | ForEach-Object { "$_" } | Out-File -Encoding utf8 "$EvalDir\bpb_gate.txt"
} finally {
    Remove-Item $Lock -Force -ErrorAction SilentlyContinue
}
Write-Host "[+] Session 6 complete. Results: $EvalDir\scores.txt and $EvalDir\bpb.log" -ForegroundColor Green
