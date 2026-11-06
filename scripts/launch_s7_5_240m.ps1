# Engineered by uncoalesced
# Parentheses 1.0-240M (slab_v4) - Session 7.5 Precision Calibrated SFT Launch Script (Steps 12,001 -> 13,000)
# Anchor: checkpoints/parentheses-1.0-240M-s6/step_12000.pt (never touches s7/latest.pt)
# Data: data/processed_sft_240m_v4_s7_5 (16,000 bins = 65.54M tokens across 5 calibrated buckets)
# LR: 1.0e-4 constant to step 12,300, then linear decay to 5.0e-6 at step 13,000
# Enforces: --container-sft, --sft-spans, --differential-adamw, --recurrent-lr-ratio 0.1

param(
    [int]$SmokeSteps = 0,
    [int]$TargetSteps = 13001,
    [double]$BaseLR = 1.0e-4
)

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONWARNINGS = "ignore"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:128"
$env:CUDA_MODULE_LOADING = "LAZY"

$RepoRoot = "E:\Parentheses\parentheses-0.9"
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
Set-Location $RepoRoot

# --- Preflight Checks ---
$Anchor = Join-Path $RepoRoot "checkpoints\parentheses-1.0-240M-s6\step_12000.pt"
$AnchorSize = 3042863665
if (-not (Test-Path $Anchor)) { throw "[FAIL] S6 anchor missing: $Anchor" }
if ((Get-Item $Anchor).Length -ne $AnchorSize) { throw "[FAIL] S6 anchor size $((Get-Item $Anchor).Length) != $AnchorSize" }

# Resume anchor selection: only auto-detect S7.5 checkpoint if resuming an interrupted S7.5 run
$S75Latest = Join-Path $RepoRoot "checkpoints\parentheses-1.0-240M-s7_5\latest.pt"
if ($SmokeSteps -eq 0 -and (Test-Path $S75Latest) -and (Get-Item $S75Latest).Length -gt 3000000000) {
    $ResumeFrom = $S75Latest
    Write-Host "[info] Auto-resuming from existing S7.5 milestone: $ResumeFrom" -ForegroundColor Green
} else {
    $ResumeFrom = $Anchor
    Write-Host "[info] Launching S7.5 from certified S6 step_12000 anchor: $ResumeFrom" -ForegroundColor Cyan
}

foreach ($f in "data\processed_sft_240m_v4_s7_5\train.bin", "data\processed_sft_240m_v4_s7_5\train.mask.bin", "data\processed_sft_240m_v4_s7_5\train_boundaries_flat.npy", "data\processed_sft_240m_v4_s7_5\manifest.json") {
    if (-not (Test-Path $f)) { throw "[FAIL] missing required S7.5 data file: $f" }
}

$Lock = Join-Path $RepoRoot "out\locks\GPU.lock"
if (Test-Path $Lock) { throw "[FAIL] GPU.lock held: $(Get-Content $Lock)" }

$eFree = [System.IO.DriveInfo]::new("E").AvailableFreeSpace / 1GB
$cFree = [System.IO.DriveInfo]::new("C").AvailableFreeSpace / 1GB
if ($eFree -lt 15) { throw "[FAIL] E: free $([math]::Round($eFree,1)) GB < 15 GB" }
if ($cFree -lt 70) { throw "[FAIL] C: free $([math]::Round($cFree,1)) GB < 70 GB" }

if ($SmokeSteps -gt 0) {
    $OutDir = "checkpoints\_s7_5_smoke"
    $MaxSteps = 12000 + $SmokeSteps + 1
    $outLog = Join-Path $RepoRoot "out\session7_5_smoke.log"
    $CkptEvery = 100000
} else {
    $OutDir = "checkpoints\parentheses-1.0-240M-s7_5"
    $MaxSteps = $TargetSteps
    $outLog = Join-Path $RepoRoot "out\session7_5_240m_live.log"
    $CkptEvery = 50
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $Lock) | Out-Null

"session7_5-240m $(Get-Date -Format s) pid=$PID" | Out-File -Encoding utf8 $Lock
Write-Host "[*] Launching Parentheses 1.0-240M Session 7.5 SFT (max-steps $MaxSteps) from $ResumeFrom ..." -ForegroundColor Cyan
$code = 1
try {
    & $PythonExe train.py `
        --preset parentheses-1.0-240M `
        --resume-from $ResumeFrom `
        --data "data/processed_sft_240m_v4_s7_5/train.bin" `
        --boundaries "data/processed_sft_240m_v4_s7_5/train_boundaries_flat.npy" `
        --sft-spans "data/processed_sft_240m_v4_s7_5/train.mask.bin" `
        --container-sft `
        --tokenizer "data/tokenizer/v3_32k" `
        --out-dir $OutDir `
        --batch-size 1 `
        --grad-accum 16 `
        --ce-rows 2048 `
        --lr $BaseLR `
        --decay-start 12300 `
        --min-lr 5.0e-6 `
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
