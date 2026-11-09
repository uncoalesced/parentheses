# Engineered by uncoalesced
# Session 9 Certification Runner Wrapper for Parentheses 1.0-240M

param (
    [string]$GoldenCkpt = "checkpoints/parentheses-1.0-240M-s8-golden/parentheses-1.0-240M-dpo-swa3-golden.pt",
    [string]$Device = "cuda",
    [switch]$DryRun,
    [switch]$SelfTest,
    [switch]$ReuseC1,
    [switch]$SkipC1,
    [switch]$SkipC2,
    [switch]$SkipC3,
    [switch]$SkipC4,
    [switch]$SkipC5,
    [switch]$SkipC6,
    [switch]$SkipC7,
    [switch]$SkipC8
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# Enforce strict C: drive write ban - redirect all temp directories to E:\scratch\tmp
$ScratchTmp = "E:\scratch\tmp"
if (-not (Test-Path $ScratchTmp)) {
    New-Item -ItemType Directory -Force -Path $ScratchTmp | Out-Null
}
$env:TEMP = $ScratchTmp
$env:TMP = $ScratchTmp

$PyExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PyExe)) {
    Write-Error "Python executable not found at $PyExe"
    exit 1
}

$LogDir = Join-Path $RepoRoot "out\eval_s9"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
}
$LogFile = Join-Path $LogDir "session9_live.log"

$ArgsList = @("scripts/run_session9_certification.py")

if ($SelfTest) {
    $ArgsList += "--self-test"
    Write-Host "[*] Executing Session 9 Self-Tests..." -ForegroundColor Cyan
    & $PyExe @ArgsList
    exit $LASTEXITCODE
}

if ($DryRun) {
    $ArgsList += "--dry-run"
    $ArgsList += "--golden-ckpt"
    $ArgsList += $GoldenCkpt
    Write-Host "[*] Executing Session 9 Dry-Run..." -ForegroundColor Cyan
    & $PyExe @ArgsList
    exit $LASTEXITCODE
}

# Forward any skip/reuse flags
if ($ReuseC1) { $ArgsList += "--reuse-c1" }
if ($SkipC1)  { $ArgsList += "--skip-c1" }
if ($SkipC2)  { $ArgsList += "--skip-c2" }
if ($SkipC3)  { $ArgsList += "--skip-c3" }
if ($SkipC4)  { $ArgsList += "--skip-c4" }
if ($SkipC5)  { $ArgsList += "--skip-c5" }
if ($SkipC6)  { $ArgsList += "--skip-c6" }
if ($SkipC7)  { $ArgsList += "--skip-c7" }
if ($SkipC8)  { $ArgsList += "--skip-c8" }

$ArgsList += "--golden-ckpt"
$ArgsList += $GoldenCkpt
$ArgsList += "--device"
$ArgsList += $Device

# GPU.lock Management
$LockDir = Join-Path $RepoRoot "out\locks"
if (-not (Test-Path $LockDir)) {
    New-Item -ItemType Directory -Force -Path $LockDir | Out-Null
}
$GpuLock = Join-Path $LockDir "GPU.lock"

if (Test-Path $GpuLock) {
    $ExistingLock = Get-Content $GpuLock -Raw
    Write-Host "[!] Existing GPU lock detected:`n$ExistingLock" -ForegroundColor Yellow
    Write-Error "GPU.lock is present at $GpuLock. Another training or evaluation process may be running."
    exit 1
}

$LockPayload = @"
PID: $PID
Task: Session 9 Master Certification
Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss IST')
User: $env:USERNAME
Golden: $GoldenCkpt
"@

Write-Host "==========================================================" -ForegroundColor Green
Write-Host " Launching Parentheses 1.0-240M Session 9 Certification" -ForegroundColor Green
Write-Host " Target Checkpoint: $GoldenCkpt" -ForegroundColor Yellow
Write-Host " Device:            $Device" -ForegroundColor Yellow
Write-Host " Log File:          $LogFile" -ForegroundColor Cyan
Write-Host " Temp Directory:    $env:TEMP" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Green

try {
    Set-Content -Path $GpuLock -Value $LockPayload -Encoding UTF8
    # Execute and tee to log (temporarily allow stderr streaming without terminating script)
    $origEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $PyExe @ArgsList 2>&1 | Tee-Object -FilePath $LogFile
    $CertExitCode = $LASTEXITCODE
    $ErrorActionPreference = $origEAP
}
finally {
    if (Test-Path $GpuLock) {
        Remove-Item -Force $GpuLock
        Write-Host "[*] Released GPU lock at $GpuLock" -ForegroundColor DarkGray
    }
}

exit $CertExitCode
