# Engineered by uncoalesced
# Parentheses 1.0-240M (slab_v4) - Session 1 Deterministic Launch Script
# Architecture: 253.54M params, 16 layers, d_model=1024, 16 heads, seq_len=4096
# Initial State: Net2WiderNet warm-start from golden 130M-v3 (init_grown.pt)

param(
    [string]$ResumeFrom = "checkpoints/parentheses-1.0-240M/init_grown.pt",
    [string]$DataPath = "data/processed_240m_apache/train.bin",
    [string]$BoundariesPath = "data/processed_240m_apache/train_boundaries.npy",
    [string]$TokenizerPath = "data/tokenizer/v3_32k",
    [string]$OutDir = "checkpoints/parentheses-1.0-240M-s1",
    [int]$BatchSize = 1,
    [int]$GradAccum = 16,
    [int]$CeRows = 2048,
    [double]$Lr = 3e-4,
    [int]$WarmupSteps = 150,
    [int]$MaxSteps = 2001,
    [int]$LogEvery = 1,
    [int]$CkptEvery = 250
)

Set-Location "E:\Parentheses\parentheses-0.9"

# 1. Environment & CUDA Guardrails
$env:PYTHONPATH = "E:\Parentheses\parentheses-0.9"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:CUDA_MODULE_LOADING = "LAZY"
# Prevent Windows WDDM shared-memory spillover past 7.6 GB
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:128"

$py = "E:\Parentheses\parentheses-0.9\.venv\Scripts\python.exe"
$outLog = "E:\Parentheses\parentheses-0.9\out\session1_240m_live.log"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path "E:\Parentheses\parentheses-0.9\out" | Out-Null

Write-Host "================================================================="
Write-Host " LAUNCHING PARENTHESES 1.0-240M (SLAB_V4) - SESSION 1 "
Write-Host " Target Checkpoint Dir : $OutDir"
Write-Host " Resume From           : $ResumeFrom"
Write-Host " Micro-batch / Accum   : $BatchSize / $GradAccum (65,536 tok/step)"
Write-Host " Context / CE Rows     : 4096 / $CeRows"
Write-Host " Log File              : $outLog"
Write-Host "================================================================="

# 2. Check for intermediate resume checkpoint
$existingCkpts = Get-ChildItem "$OutDir\step_*.pt" -ErrorAction SilentlyContinue | Where-Object { $_.Name -notmatch "final|swa" }
if ($existingCkpts) {
    $latest = $existingCkpts | Sort-Object {
        if ($_.BaseName -match "step_(\d+)") { [int]$matches[1] } else { 0 }
    } -Descending | Select-Object -First 1
    if ($latest) {
        $ResumeFrom = $latest.FullName
        Write-Host "[Auto-Resume] Found existing Session 1 checkpoint: $ResumeFrom"
    }
}

# 3. Launch Process
& $py train.py `
  --preset parentheses-1.0-240M `
  --resume-from $ResumeFrom `
  --data $DataPath `
  --boundaries $BoundariesPath `
  --tokenizer $TokenizerPath `
  --out-dir $OutDir `
  --batch-size $BatchSize `
  --grad-accum $GradAccum `
  --ce-rows $CeRows `
  --lr $Lr `
  --warmup-steps $WarmupSteps `
  --differential-adamw `
  --recurrent-lr-ratio 0.1 `
  --grad-checkpointing `
  --max-steps $MaxSteps `
  --log-every $LogEvery `
  --ckpt-every $CkptEvery 2>&1 | Tee-Object -FilePath $outLog

