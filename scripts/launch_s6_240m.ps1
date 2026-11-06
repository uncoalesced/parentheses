# Engineered by uncoalesced
# Parentheses 1.0-240M (slab_v4) - Session 6 Launch Script (Steps 10,001 -> 13,000)
# Per vault Briefs/BRIEF_SESSION_6_240M_DRAFT.md + PLAN_SESSION_6_240M_EXECUTION.md (approved 2026-11-04).
# Primary: processed_s6_anneal_240m (scripts/build_s6_anneal_240m.py: S4 STEM chat / smoltalk_v2 chat / S5 textbooks 3:3:2)
# Secondary: processed_v5 Dravidian/English replay (exactly 8/16 micro-steps)
# LR: hold 3.0e-4 to step 10,300, then linear to 1.0e-5 at step 13,000
# NOT processed_anneal_240m_apache: its "bitext" pillar duplicates Sanskrit as fake <kan> text and it reads
# in-copyright Kannada_Literature (Claude Code audit 2026-11-04).
#   -SmokeSteps N : run only N steps into checkpoints\_s6_smoke (for preflight), then exit

param([int]$SmokeSteps = 0)

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONWARNINGS = "ignore"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:128"
$env:CUDA_MODULE_LOADING = "LAZY"

$RepoRoot = "E:\Parentheses\parentheses-0.9"
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
Set-Location $RepoRoot

# --- Preflight: refuse rather than guess ---
$Anchor = Join-Path $RepoRoot "checkpoints\parentheses-1.0-240M-s5\step_10000_final.pt"
$AnchorSize = 3042870189
if (-not (Test-Path $Anchor)) { throw "[FAIL] S5 anchor missing: $Anchor" }
if ((Get-Item $Anchor).Length -ne $AnchorSize) { throw "[FAIL] S5 anchor size $((Get-Item $Anchor).Length) != $AnchorSize" }

# Resume anchor selection: auto-detect latest S6 checkpoint if resuming
$S6Latest = Join-Path $RepoRoot "checkpoints\parentheses-1.0-240M-s6\latest.pt"
if ($SmokeSteps -eq 0 -and (Test-Path $S6Latest) -and (Get-Item $S6Latest).Length -gt 3000000000) {
    $ResumeFrom = $S6Latest
    Write-Host "[info] Auto-resuming from existing S6 milestone: $ResumeFrom"
} else {
    $ResumeFrom = $Anchor
}
foreach ($f in "data\processed_s6_anneal_240m\train.bin", "data\processed_s6_anneal_240m\train_boundaries.npy", "data\processed_s6_anneal_240m\manifest.json", "data\processed_v5\train.bin") {
    if (-not (Test-Path $f)) { throw "[FAIL] missing data file: $f" }
}
$Lock = Join-Path $RepoRoot "out\locks\GPU.lock"
if (Test-Path $Lock) { throw "[FAIL] GPU.lock held: $(Get-Content $Lock)" }
$eFree = [System.IO.DriveInfo]::new("E").AvailableFreeSpace / 1GB
$cFree = [System.IO.DriveInfo]::new("C").AvailableFreeSpace / 1GB
if ($eFree -lt 15) { throw "[FAIL] E: free $([math]::Round($eFree,1)) GB < 15 GB" }
if ($cFree -lt 70) { throw "[FAIL] C: free $([math]::Round($cFree,1)) GB < 70 GB" }

if ($SmokeSteps -gt 0) {
    $OutDir = "checkpoints/_s6_smoke"
    $MaxSteps = 10000 + $SmokeSteps + 1
    $outLog = Join-Path $RepoRoot "out\session6_smoke.log"
    $CkptEvery = 100000
} else {
    $OutDir = "checkpoints/parentheses-1.0-240M-s6"
    $MaxSteps = 13001
    $outLog = Join-Path $RepoRoot "out\session6_240m_live.log"
    $CkptEvery = 100
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $Lock) | Out-Null

"session6-240m $(Get-Date -Format s) pid=$PID" | Out-File -Encoding utf8 $Lock
Write-Host "[*] Launching Parentheses 1.0-240M Session 6 (max-steps $MaxSteps) from $ResumeFrom ..."
$code = 1
try {
    & $PythonExe train.py `
        --preset parentheses-1.0-240M `
        --resume-from $ResumeFrom `
        --data "data/processed_s6_anneal_240m/train.bin" `
        --boundaries "data/processed_s6_anneal_240m/train_boundaries.npy" `
        --secondary-data-dir "data/processed_v5" `
        --secondary-per-16 8 `
        --tokenizer "data/tokenizer/v3_32k" `
        --out-dir $OutDir `
        --batch-size 1 `
        --grad-accum 16 `
        --ce-rows 2048 `
        --lr 3.0e-4 `
        --decay-start 10300 `
        --min-lr 1.0e-5 `
        --warmup-steps 0 `
        --differential-adamw `
        --recurrent-lr-ratio 0.1 `
        --grad-checkpointing `
        --max-steps $MaxSteps `
        --log-every 1 `
        --ckpt-every $CkptEvery 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $outLog -Append
    $code = $LASTEXITCODE
} finally {
    Remove-Item $Lock -Force -ErrorAction SilentlyContinue
}

if ($code -ne 0) {
    Write-Host "[!] train.py exited with error code $code" -ForegroundColor Red
    exit $code
}
exit 0
