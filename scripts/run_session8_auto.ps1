# Engineered by uncoalesced
<#
.SYNOPSIS
    Unattended S7.5 -> S8 chain for Parentheses 1.0-240M (Joel, 2026-11-07 ~01:00: "when I'm asleep it
    all takes place automatically"). Replaces Antigravity's watch_and_run_s8_pipeline.ps1.
.DESCRIPTION
    0. Wait until the S7.5 supervisor and train.py have exited and GPU.lock is free.
    1. Fill any S7.5 post-chain output that is missing (same commands as run_session7_5_supervised.ps1).
    2. Gates G1-G5 (scripts\s8_gates.py s75) + output sanity + identity spot check. Fail -> report, stop.
    3. S8: generate pairs -> pair gate -> precompute ref -> 2-step smoke -> 150-step DPO -> SWA-3.
       Checkpoints go to I:\parentheses\checkpoints\parentheses-1.0-240M-s8 (E: would drop below 15 GiB).
    4. S8 eval: 60-probe greedy + sampled on golden, greedy on step 150, score_60p, BPB (S6/base/golden),
       S8 pass bar (s8_gates.py s8), identity spot check.
    5. Report to the vault handoff folder + coordination log. Never starts S9.
    Native calls run with ErrorActionPreference=Continue and are judged by $LASTEXITCODE: under "Stop",
    PS 5.1 throws on the first stderr line (tested 2026-11-07).
    Run detached: Start-Process powershell -ArgumentList '-ExecutionPolicy','Bypass','-File','scripts\run_session8_auto.ps1' -WindowStyle Hidden
    Dry run of the control flow (no GPU, no waiting): add -DryRun
    Halt switch: create out\locks\S8_HOLD (text = reason); checked before generate, precompute and train.
    Launch outside any agent's process tree (WMI parent), see the coordination log entry of 2026-11-07.
#>
param([switch]$DryRun)

$ErrorActionPreference = "Continue"
$RepoRoot = "E:\Parentheses\parentheses-0.9"
Set-Location $RepoRoot
$env:PYTHONIOENCODING = "utf-8"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Lock = Join-Path $RepoRoot "out\locks\GPU.lock"
$Coord = Join-Path $RepoRoot "out\swarm\COORDINATION_240M.md"
$Status = Join-Path $RepoRoot "out\session8_auto_status.txt"
$S75Eval = Join-Path $RepoRoot "out\eval_s7_5"
$S75Ck = Join-Path $RepoRoot "checkpoints\parentheses-1.0-240M-s7_5"
$S75Final = Join-Path $S75Ck "step_13000_final.pt"
$S75Swa3 = Join-Path $S75Ck "step_13000_swa3.pt"
$S6Anchor = Join-Path $RepoRoot "checkpoints\parentheses-1.0-240M-s6\step_12000.pt"
$S8Ck = "I:\parentheses\checkpoints\parentheses-1.0-240M-s8"
$S8Smoke = "I:\parentheses\checkpoints\_s8_smoke"
$S8Eval = Join-Path $RepoRoot "out\eval_s8"
$Golden = Join-Path $S8Ck "parentheses-1.0-240M-dpo-swa3-golden.pt"
$DpoDir = Join-Path $RepoRoot "data\dpo_240m_apache"
$Pairs = Join-Path $DpoDir "on_policy_pairs.jsonl"
$RefCache = Join-Path $DpoDir "ref_logprobs.npy"
$Vault = "C:\Users\Joel Anthony\Documents\Phile\Heafod\Heafod\Parentheses\parentheses-0.9-handoffs"
New-Item -ItemType Directory -Force -Path $S75Eval, $S8Eval | Out-Null

function Say([string]$m) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
    $line | Out-File -Append -Encoding utf8 $Status
    Write-Host $line
}
function Coord([string]$title, [string]$body) {
    if ($DryRun) { return }
    $t = Get-Date -Format "yyyy-MM-dd HH:mm"
    "`n---`n`n### $t IST -- Claude Code S8 orchestrator: $title`n$body`n" | Out-File -Append -Encoding utf8 $Coord
}
function Py([string]$log, [string[]]$pyArgs) {
    # Returns the Python exit code; stdout+stderr go to a UTF-8 log.
    Say "  run: python $($pyArgs -join ' ')  (log $log)"
    if ($DryRun) { return 0 }
    & $Python @pyArgs 2>&1 | ForEach-Object { "$_" } | Out-File -Encoding utf8 $log
    return $LASTEXITCODE
}
function FreeGB([string]$drive) { return [math]::Round((Get-PSDrive $drive).Free / 1GB, 1) }
function Report([string]$verdict, [string]$detail) {
    $ts = Get-Date -Format "yyyy-MM-dd_HHmm"
    $path = Join-Path $Vault "HANDOFF_${ts}_s8-auto-results.md"
    $parts = @("# S7.5 -> S8 unattended chain: $verdict", "", "Written by scripts\run_session8_auto.ps1 (Claude Code) at $(Get-Date -Format s).", "", $detail, "")
    foreach ($f in @("$S75Eval\gate_verdict_claude.json", "$S75Eval\scores_claude.txt", "$S75Eval\bpb_gate.txt", "$S75Eval\identity.jsonl",
                     "$S8Eval\pairs_check.json", "$S8Eval\s8_verdict.json", "$S8Eval\scores.txt", "$S8Eval\bpb_gate.txt", "$S8Eval\identity.jsonl")) {
        if (Test-Path $f) { $parts += @("## $f", '```', (Get-Content $f -Raw -Encoding UTF8), '```', "") }
    }
    $parts += @("## Status log", '```', (Get-Content $Status -Raw -Encoding UTF8), '```',
                "", "Not done automatically: the 20-pair human review (out\eval_s8\pairs_review_20.md) -- Hermes/Joel to read. S9 not started.")
    if (-not $DryRun) {
        try { New-Item -ItemType Directory -Force -Path $Vault | Out-Null; ($parts -join "`n") | Out-File -Encoding utf8 $path }
        catch { $path = Join-Path $RepoRoot "out\HANDOFF_${ts}_s8-auto-results.md"; ($parts -join "`n") | Out-File -Encoding utf8 $path }
    }
    Coord $verdict "- $detail`n- Report: $path`n- Status log: out\session8_auto_status.txt"
    Say "REPORT: $verdict -> $path"
}
function Stop-Chain([string]$why) {
    Say "STOP: $why"
    Report "STOPPED -- S8 not completed" $why
    if ((Test-Path $Lock) -and ((Get-Content $Lock -Raw) -match "session8-claude")) { Remove-Item $Lock -Force }
    exit 1
}

Say "=== run_session8_auto.ps1 armed (pid $PID, dryrun=$DryRun) ==="

# ---------------------------------------------------------------- 0. single orchestrator + wait for S7.5
function Stop-OtherSentinels {
    $others = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "watch_and_run_s8_pipeline" -and $_.ProcessId -ne $PID }
    foreach ($p in $others) {
        Say "stopping Antigravity sentinel pid $($p.ProcessId) (Joel approved replacement, 2026-11-07 ~01:00)"
        if (-not $DryRun) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    }
}
function Check-Hold([string]$phase) {
    # Hermes / Joel / Antigravity can halt the chain without killing processes: create out\locks\S8_HOLD.
    if (Test-Path $HoldFile) { Stop-Chain "S8_HOLD present before '$phase': $((Get-Content $HoldFile -Raw).Trim())" }
}
function Check-Frozen([string]$phase) {
    foreach ($k in @($Frozen.Keys)) {
        $now = (Get-FileHash -Algorithm SHA256 $k).Hash.ToLower()
        if ($now -ne $Frozen[$k]) { Stop-Chain "$k changed after launch (sha256 $($Frozen[$k]) -> $now) before '$phase'. Re-verify and relaunch." }
    }
}
$HoldFile = Join-Path $RepoRoot "out\locks\S8_HOLD"
$Frozen = @{}
foreach ($f in @("$DpoDir\prompts.jsonl", "$RepoRoot\scripts\train_dpo_v2_apache.py", "$RepoRoot\scripts\s8_gates.py")) {
    $Frozen[$f] = (Get-FileHash -Algorithm SHA256 $f).Hash.ToLower()
    Say "frozen sha256 $($Frozen[$f]) $f"
}
Stop-OtherSentinels

if (-not $DryRun) {
    $lastNote = Get-Date
    while ($true) {
        # Only the supervisor, its launcher and the trainer count. Leftover archiver/telemetry pythons
        # (Stop-Job can orphan them) do not use the GPU and must not block S8 forever.
        $sup = Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -notmatch "run_session8_auto" -and (
            $_.CommandLine -match "run_session7_5_supervised|launch_s7_5_240m" -or
            ($_.CommandLine -match "train\.py" -and $_.CommandLine -match "parentheses-1\.0-240M-s7_5")) }
        if (-not $sup) {
            if (-not (Test-Path $Lock)) { break }
            $lk = Get-Content $Lock -Raw
            # Supervisor gone but its lock left behind (killed mid post-chain): stale after 10 min with no GPU python.
            $gpuPy = (nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>$null) | Where-Object { $_ -match "\d" }
            if ($lk -match "session7_5" -and -not $gpuPy -and ((Get-Date) - (Get-Item $Lock).LastWriteTime).TotalMinutes -gt 10) {
                Say "removing stale S7.5 GPU.lock ($($lk.Trim())); no S7.5 process, no GPU compute app"
                Remove-Item $Lock -Force
                break
            }
        }
        if (((Get-Date) - $lastNote).TotalMinutes -ge 30) { Say "waiting for S7.5 (supervisor running: $([bool]$sup), lock: $(Test-Path $Lock))"; $lastNote = Get-Date }
        Start-Sleep -Seconds 60
    }
}
Say "S7.5 supervisor finished; GPU free"
Stop-OtherSentinels
if (-not $DryRun) { "session8-claude $(Get-Date -Format s) pid=$PID" | Out-File -Encoding utf8 $Lock }

# ---------------------------------------------------------------- 0b. backfill I: archive if the supervisor's drain cut the archiver off
$Arch75 = "I:\parentheses\checkpoints\parentheses-1.0-240M-s7_5"
$Man75 = Join-Path $Arch75 "MANIFEST_s7_5_sha256.tsv"
if (-not $DryRun -and (Test-Path $Man75)) {
    $listed = @{}
    Get-Content $Man75 -Encoding UTF8 | Select-Object -Skip 1 | ForEach-Object { $c = $_ -split "`t"; if ($c.Count -ge 2) { $listed[$c[1]] = $c[0] } }
    foreach ($n in @("step_12500.pt", "step_12750.pt", "step_13000_final.pt")) {
        $src = Join-Path $S75Ck $n; $dst = Join-Path $Arch75 $n
        if ((Test-Path $src) -and (-not $listed.ContainsKey($n) -or -not (Test-Path $dst))) {
            Say "backfilling $n to I: (not in manifest or missing)"
            Copy-Item $src $dst -Force
            $hs = (Get-FileHash -Algorithm SHA256 $src).Hash.ToLower(); $hd = (Get-FileHash -Algorithm SHA256 $dst).Hash.ToLower()
            if ($hs -ne $hd) { Stop-Chain "backfill hash mismatch for $n ($hs vs $hd)" }
            $stepNum = ([regex]::Match($n, "step_(\d+)")).Groups[1].Value
            "$hs`t$n`t$stepNum`t$((Get-Item $src).Length)`t$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -Append -Encoding utf8 $Man75
        }
    }
}

# ---------------------------------------------------------------- 1. fill missing S7.5 post-chain outputs
if (-not (Test-Path $S75Final) -and -not $DryRun) { Stop-Chain "step_13000_final.pt missing: S7.5 training did not finish. See out\session7_5_240m_live.log." }
$c12500 = Join-Path $S75Ck "step_12500.pt"; $c12750 = Join-Path $S75Ck "step_12750.pt"
function Set-Aside([string]$f) { if (Test-Path $f) { $to = "$f.partial_$(Get-Date -Format yyyyMMddHHmmss)"; Say "setting aside incomplete $f -> $to"; if (-not $DryRun) { Move-Item $f $to -Force } } }
function Rows-Ok([string]$f, [int]$n) {
    if (-not (Test-Path $f)) { return $false }
    $c = @(Get-Content $f -Encoding UTF8 | Where-Object { $_.Trim() }).Count
    if ($c -ne $n) { Say "$f has $c rows, expected $n"; return $false }
    return $true
}
function Ckpt-Ok([string]$f) {
    if (-not (Test-Path $f)) { return $false }
    if ($DryRun) { return $true }
    & $Python -c "import sys,torch; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False,mmap=True); assert 'model' in c and 'cfg' in c" $f 2>&1 | Out-Null
    return ($LASTEXITCODE -eq 0)
}
# latest.pt must equal the final (G1). The supervisor's drain gives up after 20 min; copy if it did.
if (-not $DryRun -and (Test-Path $S75Final)) {
    $lp = Join-Path $S75Ck "latest.pt"
    if (-not (Test-Path $lp) -or (Get-Item $lp).Length -ne (Get-Item $S75Final).Length) { Say "latest.pt != final; copying final -> latest.pt"; Copy-Item $S75Final $lp -Force }
}
# The supervisor's post-chain can die mid-step (PS 5.1 Stop + stderr). Check completeness, not existence.
if ((Test-Path $S75Swa3) -and -not (Ckpt-Ok $S75Swa3)) { Set-Aside $S75Swa3 }
if (-not (Test-Path $S75Swa3) -and (Test-Path $c12500) -and (Test-Path $c12750)) {
    $rc = Py "$S75Eval\swa3_merge_rerun.log" @("scripts\merge_swa_checkpoints.py", "--ckpts", $c12500, $c12750, $S75Final, "--out", $S75Swa3)
    if ((Test-Path $S75Swa3) -and -not (Ckpt-Ok $S75Swa3)) { Set-Aside $S75Swa3 }
}
if (-not (Rows-Ok "$S75Eval\s7_5_chat_greedy.jsonl" 60)) {
    Set-Aside "$S75Eval\s7_5_chat_greedy.jsonl"
    $rc = Py "$S75Eval\eval_greedy_rerun.log" @("scripts\eval_60p_adversarial_suite.py", "--checkpoint", $S75Final, "--output", "$S75Eval\s7_5_chat_greedy.jsonl", "--mode", "chat", "--temp", "0.0", "--samples", "1", "--max-tokens", "256")
}
if (-not (Rows-Ok "$S75Eval\s7_5_chat_t07x5.jsonl" 300)) {
    Set-Aside "$S75Eval\s7_5_chat_t07x5.jsonl"
    $rc = Py "$S75Eval\eval_sampled_rerun.log" @("scripts\eval_60p_adversarial_suite.py", "--checkpoint", $S75Final, "--output", "$S75Eval\s7_5_chat_t07x5.jsonl", "--mode", "chat", "--temp", "0.7", "--samples", "5", "--seed", "1234", "--max-tokens", "256")
}
if ((Test-Path $S75Swa3) -and -not (Rows-Ok "$S75Eval\s7_5_swa3_chat_greedy.jsonl" 60)) {
    Set-Aside "$S75Eval\s7_5_swa3_chat_greedy.jsonl"
    $rc = Py "$S75Eval\eval_swa3_greedy_rerun.log" @("scripts\eval_60p_adversarial_suite.py", "--checkpoint", $S75Swa3, "--output", "$S75Eval\s7_5_swa3_chat_greedy.jsonl", "--mode", "chat", "--temp", "0.0", "--samples", "1", "--max-tokens", "256")
}
$scoreFiles = @("out\eval_s6_12k\s6_12000_chat_greedy.jsonl", "$S75Eval\s7_5_chat_greedy.jsonl")
if (Test-Path "$S75Eval\s7_5_swa3_chat_greedy.jsonl") { $scoreFiles += "$S75Eval\s7_5_swa3_chat_greedy.jsonl" }
$scoreFiles += "$S75Eval\s7_5_chat_t07x5.jsonl"
$rc = Py "$S75Eval\scores_claude.txt" (@("scripts\score_60p.py") + $scoreFiles + @("--json", "$S75Eval\scores.json"))
$bpbOk = $false
if (Test-Path "$S75Eval\bpb_240m.json") {
    $b = @(Get-Content "$S75Eval\bpb_240m.json" -Raw | ConvertFrom-Json)
    $want = 2; if (Test-Path $S75Swa3) { $want = 3 }
    $bpbOk = ($b.Count -eq $want)
}
if (-not $bpbOk) {
    $ck = @($S6Anchor, $S75Final); if (Test-Path $S75Swa3) { $ck += $S75Swa3 }
    $rc = Py "$S75Eval\bpb_rerun.log" (@("scripts\bpb_slab_v4.py", "--checkpoints") + $ck + @("--out", "$S75Eval\bpb_240m.json", "--device", "cuda"))
    $rc = Py "$S75Eval\bpb_gate.txt" @("scripts\bpb_gate.py", "$S75Eval\bpb_240m.json", "--max-rise", "0.06")
}

# ---------------------------------------------------------------- 2. gates + sanity + identity
$gateRc = Py "$S75Eval\gate_claude.log" @("scripts\s8_gates.py", "s75")
$identCk = @($S75Final); if (Test-Path $S75Swa3) { $identCk += $S75Swa3 }
$null = Py "$S75Eval\identity.log" (@("scripts\s8_identity_check.py", "--checkpoints") + $identCk + @("--out", "$S75Eval\identity.jsonl"))
if ($gateRc -ne 0) { Stop-Chain "S7.5 gates did not pass (s8_gates.py s75 exit $gateRc). Details: out\eval_s7_5\gate_verdict_claude.json" }
if ($DryRun) { $base = $S75Final; $baseGreedy = "$S75Eval\s7_5_chat_greedy.jsonl" }
else {
    $v = Get-Content "$S75Eval\gate_verdict_claude.json" -Raw | ConvertFrom-Json
    $base = $v.candidate_base; $baseGreedy = $v.base_greedy_file
}
Say "S7.5 gates PASS. S8 base: $base"
Coord "S7.5 GATES PASS -- S8 STARTING" "- Base: $base`n- Verdict: out\eval_s7_5\gate_verdict_claude.json"

# ---------------------------------------------------------------- 3. S8
if ((FreeGB "I") -lt 12) { Stop-Chain "I: has $(FreeGB 'I') GB free; S8 needs about 6 GB plus headroom." }
if ((FreeGB "E") -lt 3) { Stop-Chain "E: has $(FreeGB 'E') GB free." }
if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path $S8Ck | Out-Null
    foreach ($f in @($Pairs, $RefCache)) { if (Test-Path $f) { Move-Item $f "$f.prev_$(Get-Date -Format yyyyMMddHHmm)" -Force } }
}
$dpo = "scripts\train_dpo_v2_apache.py"
Check-Hold "generate"; Check-Frozen "generate"
# 6 candidates per prompt (Joel, 2026-11-07 ~01:10): projected yield at 4 was ~72% (~860 pairs), too close to 800.
$rc = Py "out\s8_generate.log" @($dpo, "--mode", "generate", "--sft-ckpt", $base, "--num-candidates", "6")
if ($rc -ne 0) { Stop-Chain "pair generation failed (exit $rc), out\s8_generate.log" }
$rc = Py "$S8Eval\pairs_check.log" @("scripts\s8_gates.py", "pairs")
if ($rc -ne 0) { Stop-Chain "pair gate failed (need >= 800 pairs, all chosen > rejected): out\eval_s8\pairs_check.json" }
Check-Hold "precompute-ref"; Check-Frozen "precompute-ref"
$rc = Py "out\s8_precompute.log" @($dpo, "--mode", "precompute-ref", "--sft-ckpt", $base)
if ($rc -ne 0) { Stop-Chain "precompute-ref failed (exit $rc), out\s8_precompute.log" }
Check-Hold "train"
$rc = Py "out\s8_smoke.log" @($dpo, "--mode", "train", "--sft-ckpt", $base, "--max-steps", "2", "--save-every", "1000", "--out-dir", $S8Smoke, "--metrics-jsonl", "out\session8_smoke_metrics.jsonl", "--lr", "5e-7", "--beta", "0.1", "--nll-coeff", "0.1")
if ($rc -ne 0) { Stop-Chain "2-step smoke failed (exit $rc), out\s8_smoke.log" }
$rc = Py "out\s8_train.log" @($dpo, "--mode", "train", "--sft-ckpt", $base, "--max-steps", "150", "--save-every", "50", "--lr", "5e-7", "--beta", "0.1", "--nll-coeff", "0.1", "--out-dir", $S8Ck, "--metrics-jsonl", "out\session8_dpo_metrics.jsonl")
if ($rc -ne 0) { Stop-Chain "DPO training failed (exit $rc), out\s8_train.log" }
$rc = Py "out\s8_swa3.log" @($dpo, "--mode", "swa3", "--out-dir", $S8Ck)
if ($rc -ne 0 -or (-not $DryRun -and -not (Test-Path $Golden))) { Stop-Chain "SWA-3 merge failed (exit $rc), out\s8_swa3.log" }
if (-not $DryRun) {
    $man = Join-Path $S8Ck "MANIFEST_s8_sha256.tsv"
    "sha256`tfilename`tfile_size_bytes`ttimestamp" | Out-File -Encoding utf8 $man
    Get-ChildItem $S8Ck -Filter *.pt | ForEach-Object {
        "$((Get-FileHash -Algorithm SHA256 $_.FullName).Hash.ToLower())`t$($_.Name)`t$($_.Length)`t$(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $man
    }
}
Say "S8 training + SWA-3 done: $Golden"

# ---------------------------------------------------------------- 4. S8 eval
$step150 = Join-Path $S8Ck "step_0150.pt"
$null = Py "$S8Eval\eval_greedy.log" @("scripts\eval_60p_adversarial_suite.py", "--checkpoint", $Golden, "--output", "$S8Eval\s8_golden_chat_greedy.jsonl", "--mode", "chat", "--temp", "0.0", "--samples", "1", "--max-tokens", "256")
$null = Py "$S8Eval\eval_sampled.log" @("scripts\eval_60p_adversarial_suite.py", "--checkpoint", $Golden, "--output", "$S8Eval\s8_golden_chat_t07x5.jsonl", "--mode", "chat", "--temp", "0.7", "--samples", "5", "--seed", "1234", "--max-tokens", "256")
$null = Py "$S8Eval\eval_step150_greedy.log" @("scripts\eval_60p_adversarial_suite.py", "--checkpoint", $step150, "--output", "$S8Eval\s8_step150_chat_greedy.jsonl", "--mode", "chat", "--temp", "0.0", "--samples", "1", "--max-tokens", "256")
$null = Py "$S8Eval\scores.txt" @("scripts\score_60p.py", "out\eval_s6_12k\s6_12000_chat_greedy.jsonl", $baseGreedy, "$S8Eval\s8_step150_chat_greedy.jsonl", "$S8Eval\s8_golden_chat_greedy.jsonl", "$S8Eval\s8_golden_chat_t07x5.jsonl", "--json", "$S8Eval\scores.json")
$null = Py "$S8Eval\bpb.log" @("scripts\bpb_slab_v4.py", "--checkpoints", $S6Anchor, $base, $Golden, "--out", "$S8Eval\bpb_240m.json", "--device", "cuda")
$null = Py "$S8Eval\bpb_gate.txt" @("scripts\bpb_gate.py", "$S8Eval\bpb_240m.json", "--max-rise", "0.06")
$null = Py "$S8Eval\identity.log" @("scripts\s8_identity_check.py", "--checkpoints", $Golden, "--out", "$S8Eval\identity.jsonl")
$rc = Py "$S8Eval\s8_gate.log" @("scripts\s8_gates.py", "s8", "--eval-dir", $S8Eval, "--base-greedy", $baseGreedy)

if ((Test-Path $Lock) -and ((Get-Content $Lock -Raw) -match "session8-claude")) { Remove-Item $Lock -Force }
if ($rc -eq 0) { Report "S8 PASSED its pass bar" "Golden: $Golden. Base: $base. Wait for Joel before S9." }
else { Report "S8 FINISHED but FAILED its pass bar (exit $rc)" "Golden: $Golden. See out\eval_s8\s8_verdict.json. Wait for Joel." }
Say "=== done ==="
exit 0
