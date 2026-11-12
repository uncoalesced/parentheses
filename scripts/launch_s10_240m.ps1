# Engineered by uncoalesced
# Parentheses 1.0-240M (Tanzanite) - Session 10 behaviour SFT launcher.
# Modes:  -Mode pilot  150 steps on the 4% smoke corpus (checks stop rate / Final: rate before the full run)
#         -Mode full   1,000 steps on data/processed_sft_240m_s10 (about 6.6 h on the RTX 5050)
# Bases:  -Base blend  checkpoints\parentheses-1.0-240M-s8-golden\parentheses-1.0-240M-s6-s8-blend-best.pt  (Joel, 2026-11-09)
#         -Base s6     checkpoints\parentheses-1.0-240M-s6\step_12000.pt                                     (fallback)
# The blend has no optimizer state, so a 30-step LR warmup is used. Its saved step is 13080, so training resumes at 13081.
# Checkpoints go to I: because E: has under 20 GB free. This script does NOT start by itself: run it on purpose.
#
#   powershell -File scripts\launch_s10_240m.ps1 -Mode pilot
#   powershell -File scripts\launch_s10_240m.ps1 -Mode full
#   powershell -File scripts\launch_s10_240m.ps1 -Mode full -Base s6

param(
    [ValidateSet("pilot", "full")][string]$Mode = "pilot",
    [ValidateSet("blend", "s6")][string]$Base = "blend",
    [double]$BaseLR = 1.0e-4,
    [double]$EosWeight = 3.0
)

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONWARNINGS = "ignore"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:128"
$env:CUDA_MODULE_LOADING = "LAZY"

$RepoRoot = "E:\Parentheses\parentheses-0.9"
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
Set-Location $RepoRoot

if ($Base -eq "blend") {
    $Anchor = Join-Path $RepoRoot "checkpoints\parentheses-1.0-240M-s8-golden\parentheses-1.0-240M-s6-s8-blend-best.pt"
    $StartStep = 13081
} else {
    $Anchor = Join-Path $RepoRoot "checkpoints\parentheses-1.0-240M-s6\step_12000.pt"
    $StartStep = 12001
}
if (-not (Test-Path $Anchor)) { throw "[FAIL] base checkpoint missing: $Anchor" }

if ($Mode -eq "pilot") {
    $DataDir = "data\processed_sft_240m_s10_smoke"
    $Steps = 150
    $OutDir = "I:\parentheses\checkpoints\parentheses-1.0-240M-s10-pilot-$Base"
    $DecayArgs = @()
    $CkptEvery = 150
    $outLog = Join-Path $RepoRoot "out\s10\pilot_$Base.log"
} else {
    $DataDir = "data\processed_sft_240m_s10"
    $Steps = 1000
    $OutDir = "I:\parentheses\checkpoints\parentheses-1.0-240M-s10-$Base"
    $DecayArgs = @("--decay-start", ($StartStep + 300), "--min-lr", "5.0e-6")
    $CkptEvery = 250
    $outLog = Join-Path $RepoRoot "out\s10\full_$Base.log"
}
$MaxSteps = $StartStep + $Steps

foreach ($f in "$DataDir\train.bin", "$DataDir\train.mask.bin", "$DataDir\train_boundaries_flat.npy", "$DataDir\manifest.json") {
    if (-not (Test-Path $f)) { throw "[FAIL] missing required S10 data file: $f" }
}

$Lock = Join-Path $RepoRoot "out\locks\GPU.lock"
if (Test-Path $Lock) { throw "[FAIL] GPU.lock held: $(Get-Content $Lock)" }
$eFree = [System.IO.DriveInfo]::new("E").AvailableFreeSpace / 1GB
$iFree = [System.IO.DriveInfo]::new("I").AvailableFreeSpace / 1GB
if ($eFree -lt 5) { throw "[FAIL] E: free $([math]::Round($eFree,1)) GB < 5 GB" }
if ($iFree -lt 40) { throw "[FAIL] I: free $([math]::Round($iFree,1)) GB < 40 GB" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $Lock) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $outLog) | Out-Null

"session10-$Mode-$Base $(Get-Date -Format s) pid=$PID" | Out-File -Encoding utf8 $Lock
Write-Host "[*] S10 $Mode from $Anchor (resume step $StartStep, max-steps $MaxSteps, eos-weight $EosWeight, out $OutDir)" -ForegroundColor Cyan
$code = 1
try {
    & $PythonExe train.py `
        --preset parentheses-1.0-240M `
        --resume-from $Anchor `
        --data "$DataDir/train.bin" `
        --boundaries "$DataDir/train_boundaries_flat.npy" `
        --sft-spans "$DataDir/train.mask.bin" `
        --container-sft `
        --tokenizer "data/tokenizer/v3_32k" `
        --out-dir $OutDir `
        --batch-size 1 `
        --grad-accum 16 `
        --ce-rows 512 `
        --lr $BaseLR `
        --warmup-steps 30 `
        @DecayArgs `
        --eos-weight $EosWeight `
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
