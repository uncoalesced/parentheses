# Engineered by uncoalesced
# Parentheses 1.0-240M (slab_v4) - Session 3 Launch Script (Steps 4,001 -> 6,001)
# Dual-Memmap Micro-Batch Interleaving: processed_240m_apache (Anchor) + processed_v5 (Multilingual/Low-Res)

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONWARNINGS = "ignore"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:128"
$env:CUDA_MODULE_LOADING = "LAZY"

$RepoRoot = "E:\Parentheses\parentheses-0.9"
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python.exe"
}

$S2FinalCkpt = "I:\parentheses\checkpoints\parentheses-1.0-240M-s2\step_4000_final.pt"
$S2LocalFallback = Join-Path $RepoRoot "checkpoints\parentheses-1.0-240M-s2\latest.pt"

$ResumeCkpt = if (Test-Path $S2LocalFallback) { $S2LocalFallback } else { $S2FinalCkpt }
if (-not (Test-Path $ResumeCkpt)) {
    throw "[FAIL] Session 2 terminal checkpoint not found at $S2FinalCkpt or $S2LocalFallback"
}

$OutDir = "checkpoints/parentheses-1.0-240M-s3"
$ColdCkptDir = "I:\parentheses\checkpoints\parentheses-1.0-240M-s3"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path $ColdCkptDir | Out-Null

$outLog = "E:\Parentheses\parentheses-0.9\out\session3_240m_live.log"
New-Item -ItemType Directory -Force -Path (Split-Path $outLog) | Out-Null

Set-Location $RepoRoot
Write-Host "[*] Launching Parentheses 1.0-240M (slab_v4) Session 3 (Dual-Memmap Interleaving) from $ResumeCkpt..."

& $PythonExe train.py `
    --preset parentheses-1.0-240M `
    --resume-from $ResumeCkpt `
    --data "data/processed_240m_apache/train.bin" `
    --secondary-data-dir "data/processed_v5" `
    --boundaries "data/processed_240m_apache/train_boundaries.npy" `
    --tokenizer "data/tokenizer/v3_32k" `
    --out-dir $OutDir `
    --batch-size 1 `
    --grad-accum 16 `
    --ce-rows 2048 `
    --lr 4.0e-4 `
    --min-lr 3.0e-4 `
    --decay-start 5001 `
    --decay-end 6001 `
    --warmup-steps 0 `
    --differential-adamw `
    --recurrent-lr-ratio 0.1 `
    --grad-checkpointing `
    --max-steps 6001 `
    --log-every 1 `
    --ckpt-every 100 2>&1 | Tee-Object -FilePath $outLog

if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] train.py exited with error code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
