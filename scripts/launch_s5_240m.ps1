# Engineered by uncoalesced
# Parentheses 1.0-240M (slab_v4) - Session 5 Launch Script (Steps 8,001 -> 10,000)
# Primary: processed_s5_textbooks_240m (Cosmopedia-v2 + FineWeb-Edu, English-only, 8/16 micro-steps)
# Secondary: processed_v5 Dravidian/English replay (exactly 8/16 micro-steps; raised from S4's 6/16
#   because S4 worsened held-out BPB in all 9 languages, Kuvi +7.2%, Kolami +5.9% relative)
# LR: constant 3e-4 (no decay; the single anneal happens in Session 6)
#   -SmokeSteps N : run only N steps into checkpoints\_s5_smoke (for preflight), then exit

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
$Anchor = Join-Path $RepoRoot "checkpoints\parentheses-1.0-240M-s4\step_8000_final.pt"
$AnchorSize = 3042869091
if (-not (Test-Path $Anchor)) { throw "[FAIL] S4 anchor missing: $Anchor" }
if ((Get-Item $Anchor).Length -ne $AnchorSize) { throw "[FAIL] S4 anchor size $((Get-Item $Anchor).Length) != $AnchorSize" }
foreach ($f in "data\processed_s5_textbooks_240m\train.bin", "data\processed_s5_textbooks_240m\train_boundaries.npy", "data\processed_s5_textbooks_240m\manifest.json", "data\processed_v5\train.bin") {
    if (-not (Test-Path $f)) { throw "[FAIL] missing data file: $f" }
}
$Lock = Join-Path $RepoRoot "out\locks\GPU.lock"
if (Test-Path $Lock) { throw "[FAIL] GPU.lock held: $(Get-Content $Lock)" }
$eFree = (Get-PSDrive E).Free / 1GB
$cFree = (Get-PSDrive C).Free / 1GB
if ($eFree -lt 15) { throw "[FAIL] E: free $([math]::Round($eFree,1)) GB < 15 GB" }
if ($cFree -lt 70) { throw "[FAIL] C: free $([math]::Round($cFree,1)) GB < 70 GB" }

if ($SmokeSteps -gt 0) {
    $OutDir = "checkpoints/_s5_smoke"
    $MaxSteps = 8001 + $SmokeSteps - 1
    $outLog = Join-Path $RepoRoot "out\session5_smoke.log"
    $CkptEvery = 100000
} else {
    $OutDir = "checkpoints/parentheses-1.0-240M-s5"
    $MaxSteps = 10001
    $outLog = Join-Path $RepoRoot "out\session5_240m_live.log"
    $CkptEvery = 100
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $Lock) | Out-Null

"session5-240m $(Get-Date -Format s) pid=$PID" | Out-File -Encoding utf8 $Lock
Write-Host "[*] Launching Parentheses 1.0-240M Session 5 (max-steps $MaxSteps) from $Anchor ..."
$code = 1
try {
    & $PythonExe train.py `
        --preset parentheses-1.0-240M `
        --resume-from $Anchor `
        --data "data/processed_s5_textbooks_240m/train.bin" `
        --boundaries "data/processed_s5_textbooks_240m/train_boundaries.npy" `
        --secondary-data-dir "data/processed_v5" `
        --secondary-per-16 8 `
        --tokenizer "data/tokenizer/v3_32k" `
        --out-dir $OutDir `
        --batch-size 1 `
        --grad-accum 16 `
        --ce-rows 2048 `
        --lr 3.0e-4 `
        --min-lr 3.0e-4 `
        --warmup-steps 0 `
        --differential-adamw `
        --recurrent-lr-ratio 0.1 `
        --grad-checkpointing `
        --max-steps $MaxSteps `
        --log-every 1 `
        --ckpt-every $CkptEvery 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $outLog
    $code = $LASTEXITCODE
} finally {
    Remove-Item $Lock -Force -ErrorAction SilentlyContinue
}

if ($code -ne 0) {
    Write-Host "[!] train.py exited with error code $code" -ForegroundColor Red
    exit $code
}
exit 0
