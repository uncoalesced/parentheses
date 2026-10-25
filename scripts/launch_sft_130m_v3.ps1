# Parentheses 0.9-130M-v3 (Project Tanzinite) Post-Pretraining SFT Launch Script
# Architecture: 148.38M params, 16 layers, d_model=768, 12 heads, block_size=4096
# Dataset: data/processed_sft_stem_open_v2 (105.88k docs / 64.63M tokens: 45k STEM, 25k OpenData, 20.88k Distilled Dialogues with <think> traces, 15k Dravidian Replay)
# Mask: data/processed_sft_stem_open_v2/train.mask.bin (87.2% supervised tokens, assistant + think tags only)

param(
    [string]$ResumeFrom = "checkpoints/parentheses-0.9-130M-v3/step_22888_swa3_base.pt",
    [string]$DataDir = "data/processed_sft_stem_open_v2",
    [string]$OutDir = "checkpoints/parentheses-0.9-130M-v3-sft",
    [string]$ReplayData = "data/processed_130m_v1/train.bin",
    [double]$ReplayFrac = 0.2,
    [int]$Steps = 2000,
    [double]$Lr = 3e-5,
    [double]$MinLr = 5e-6,
    [int]$BatchSize = 4,
    [int]$GradAccum = 4,
    [int]$CeRows = 4096
)

Set-Location E:\Parentheses\parentheses-0.9
$env:PYTHONIOENCODING = 'utf-8'
$py = '.venv\Scripts\python.exe'
$log = 'out\launch_sft_130m_v3.log'

function Say($m) {
    $l = "$(Get-Date -f 'yyyy-MM-dd HH:mm:ss') $m"
    Write-Host $l
    $l | Out-File $log -Append -Encoding utf8
}

Say "=== Initializing SFT Launch Sequence for parentheses-0.9-130M-v3 ==="

# 1. Resolve Resume Checkpoint (Prioritizing SWA-3 Model Soup)
$candidates = @(
    $ResumeFrom,
    "checkpoints/parentheses-0.9-130M-v3/step_22888_swa3_base.pt",
    "I:/parentheses/checkpoints/parentheses-0.9-130M-v3/step_22888_swa3_base.pt",
    "I:/parentheses/checkpoints/parentheses-0.9-130M-v3/step_22888_swa3.pt",
    "checkpoints/parentheses-0.9-130M-v3/step_22887_final.pt",
    "checkpoints/parentheses-0.9-130M-v3/step_22888_final.pt",
    "checkpoints/parentheses-0.9-130M-v3/step_22888.pt"
)

$resolvedPath = $null
foreach ($cand in $candidates) {
    if (Test-Path $cand) {
        $resolvedPath = (Resolve-Path $cand).Path
        break
    }
}

if (-not $resolvedPath) {
    $latest = Get-ChildItem "checkpoints/parentheses-0.9-130M-v3\step_*.pt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latest) {
        $resolvedPath = $latest.FullName
        Say "Warning: Specified SWA checkpoint not found. Falling back to latest available: $resolvedPath"
    } else {
        Say "[ABORT] No base pretraining checkpoint found at $ResumeFrom"
        exit 1
    }
}
$ResumeFrom = $resolvedPath

# 2. Inspect starting step from checkpoint & Guard Against Step-Counter Trap
$inspectCmd = "& '$py' -c `"import sys; sys.path.insert(0, '.'); import torch; ckpt = torch.load('$ResumeFrom', map_location='cpu', weights_only=False); print(ckpt.get('step', 0))`""
$rawStep = [int](Invoke-Expression $inspectCmd)

# If resuming from pretraining (rawStep >= 20000), auto-package clean SFT starting checkpoint with step=0
if ($rawStep -ge 20000) {
    Say "Notice: Pretraining checkpoint detected at step $rawStep with unreset step index."
    $normalizedCkpt = "checkpoints/parentheses-0.9-130M-v3/step_22888_sft_normalized.pt"
    Say "Packaging normalized SFT starting checkpoint (step=0, fresh AdamW moments) at: $normalizedCkpt..."
    $normCmd = "& '$py' -c `"import sys; sys.path.insert(0, '.'); import torch; c = torch.load('$ResumeFrom', map_location='cpu', weights_only=False); out = {'model': c['model'], 'cfg': c.get('cfg', c.get('config')), 'step': 0, 'tokenizer': c.get('tokenizer'), 'normalized_from_step': $rawStep}; torch.save(out, '$normalizedCkpt')`""
    Invoke-Expression $normCmd
    $ResumeFrom = (Resolve-Path $normalizedCkpt).Path
    $rawStep = 0
    Say "Clean SFT base checkpoint armed: $ResumeFrom"
}

if ($rawStep -le 0) {
    Say "Initializing fresh SFT run from step 0."
    $startStep = 0
    $targetStep = $Steps
    $decayStart = [int]($Steps * 0.75)
} else {
    Say "Resuming in-progress SFT run from step $rawStep."
    $startStep = $rawStep
    $targetStep = $startStep + $Steps
    $decayStart = $startStep + [int]($Steps * 0.75)
}

Say "Base Checkpoint: $ResumeFrom (Base Step: $rawStep -> SFT Start Step: $startStep)"
Say "SFT Target Step: $targetStep (Total SFT Steps: $Steps | Cooldown starts at: $decayStart | Peak LR: $Lr | Min LR: $MinLr)"

# 3. Verify SFT data assets
$trainBin = "$DataDir/train.bin"
$maskBin = "$DataDir/train.mask.bin"
$boundariesNpy = "$DataDir/train_boundaries.npy"

if (-not (Test-Path $trainBin) -or -not (Test-Path $maskBin) -or -not (Test-Path $boundariesNpy)) {
    Say "[ABORT] Tokenized SFT dataset files missing in $DataDir"
    exit 1
}

# 4. Acquire GPU lock
$lockFile = 'out\locks\GPU.lock'
if (Test-Path $lockFile) {
    $holder = (Get-Content $lockFile -Raw).Trim()
    Say "Notice: GPU.lock currently held by: $holder. Waiting for lock release..."
    while (Test-Path $lockFile) {
        Start-Sleep 15
    }
    Say "GPU.lock freed."
}

New-Item -ItemType Directory -Path (Split-Path $lockFile) -Force | Out-Null
"parentheses-130m-v3-sft (PID $PID, started $(Get-Date -f 'yyyy-MM-dd HH:mm:ss'))" | Out-File $lockFile -Encoding ascii
Say "Acquired GPU.lock."

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

$trainArgs = @(
    'train.py',
    '--preset', 'parentheses-0.9-130M-v3',
    '--data', $trainBin,
    '--sft-spans', $maskBin,
    '--boundaries', $boundariesNpy,
    '--resume-from', $ResumeFrom,
    '--out-dir', $OutDir,
    '--batch-size', $BatchSize,
    '--grad-accum', $GradAccum,
    '--ce-rows', $CeRows,
    '--lr', $Lr,
    '--min-lr', $MinLr,
    '--decay-start', $decayStart,
    '--max-steps', $targetStep,
    '--differential-adamw',
    '--recurrent-lr-ratio', '0.1',
    '--agc',
    '--clip-factor', '0.05',
    '--decay-clip-factor', '0.01',
    '--weight-decay', '0.05',
    '--grad-checkpointing',
    '--ckpt-every', '500',
    '--log-every', '10'
)

if ((Test-Path $ReplayData) -and ($ReplayFrac -gt 0.0)) {
    $trainArgs += @('--replay-data', $ReplayData, '--replay-frac', $ReplayFrac)
    Say "Pretrain Replay Buffer Active: $ReplayData ($($ReplayFrac * 100)% per micro-batch)"
}

Say "Executing SFT training run on RTX 5050..."
try {
    & $py $trainArgs
    Say "SFT Sprint Completed Successfully! Checkpoint saved in $OutDir"
} finally {
    if (Test-Path $lockFile) {
        Remove-Item $lockFile -Force
        Say "Released GPU.lock."
    }
}
