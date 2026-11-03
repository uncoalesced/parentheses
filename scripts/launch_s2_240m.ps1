# Engineered by uncoalesced
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONWARNINGS = "ignore"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:128"
$env:CUDA_MODULE_LOADING = "LAZY"

$RepoRoot = "E:\Parentheses\parentheses-0.9"
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$S2Latest = Join-Path $RepoRoot "checkpoints\parentheses-1.0-240M-s2\latest.pt"
$S2Cold = "I:\parentheses\checkpoints\parentheses-1.0-240M-s2\latest.pt"
$S1FinalCkpt = "I:\parentheses\checkpoints\parentheses-1.0-240M-s1\step_2000.pt"
$S1LocalFallback = Join-Path $RepoRoot "checkpoints\parentheses-1.0-240M-s1\latest.pt"

$ResumeCkpt = if (Test-Path $S2Latest) { $S2Latest }
              elseif (Test-Path $S2Cold) { $S2Cold }
              elseif (Test-Path $S1FinalCkpt) { $S1FinalCkpt }
              else { $S1LocalFallback }
if (-not (Test-Path $ResumeCkpt)) {
    throw "[FAIL] Resume checkpoint not found at $ResumeCkpt"
}

$OutDir = "checkpoints/parentheses-1.0-240M-s2"
$ColdCkptDir = "I:\parentheses\checkpoints\parentheses-1.0-240M-s2"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path $ColdCkptDir | Out-Null

$outLog = "E:\Parentheses\parentheses-0.9\out\session2_240m_live.log"
New-Item -ItemType Directory -Force -Path (Split-Path $outLog) | Out-Null

Set-Location $RepoRoot
Write-Host "[*] Launching Parentheses 1.0-240M (slab_v4) Session 2 from $ResumeCkpt..."

& $PythonExe train.py `
    --preset parentheses-1.0-240M `
    --resume-from $ResumeCkpt `
    --data "data/processed_240m_apache/train.bin" `
    --boundaries "data/processed_240m_apache/train_boundaries.npy" `
    --tokenizer "data/tokenizer/v3_32k" `
    --out-dir $OutDir `
    --batch-size 1 `
    --grad-accum 16 `
    --ce-rows 2048 `
    --lr 4.0e-4 `
    --warmup-steps 0 `
    --differential-adamw `
    --recurrent-lr-ratio 0.1 `
    --grad-checkpointing `
    --max-steps 4001 `
    --log-every 1 `
    --ckpt-every 100 2>&1 | Tee-Object -FilePath $outLog
