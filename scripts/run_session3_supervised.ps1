<#
.SYNOPSIS
    Master Supervised Orchestrator for Parentheses 1.0-240M Session 3 Pre-Training.
.DESCRIPTION
    Launches:
    1. Parallel Real-Time Telemetry & Watchdog Engine (telemetry_watchdog_240m.py)
    2. Parallel Checkpoint Archival & Headroom Governance Engine (checkpoint_manager_s3.py)
    3. Core Training Pipeline (launch_s3_240m.ps1)
    
    Upon completion of Step 6,000, automatically triggers:
    4. Post-Training Telemetry, Validation & Adversarial Evaluation (post_training_eval_s3.py)
#>

$ErrorActionPreference = "Stop"

$RepoRoot = "E:\Parentheses\parentheses-0.9"
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$LogPath = Join-Path $RepoRoot "out\session3_240m_live.log"

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host " PARENTHESES 1.0-240M -- SESSION 3 SUPERVISED EXECUTION PIPELINE" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan

# Pre-flight Check: GPU & NVMe Headroom
$cFree = [math]::Round((Get-PSDrive C).Free / 1GB, 2)
$eFree = [math]::Round((Get-PSDrive E).Free / 1GB, 2)
Write-Host "[pre-flight] Disk Space Check: C: ${cFree} GB free | E: ${eFree} GB free" -ForegroundColor Gray

if ($cFree -lt 70.0) {
    Write-Warning "C: drive free space is below safety threshold (<70 GB). Aborting."
    exit 1
}
if ($eFree -lt 15.0) {
    Write-Warning "E: drive free space is below safety threshold (<15 GB). Aborting."
    exit 1
}

# 1. Start Background Telemetry & Watchdog Engine
Write-Host "[1/4] Spawning Parallel Telemetry & Watchdog Engine..." -ForegroundColor Yellow
$telemetryJob = Start-Job -ScriptBlock {
    param($root, $py, $log)
    Set-Location $root
    & $py "scripts\telemetry_watchdog_240m.py" --log-path $log --max-steps 6001 --interval 15 *> "out\telemetry_watchdog.log"
} -ArgumentList $RepoRoot, $Python, $LogPath

# 2. Start Background Checkpoint Archiver & Headroom Manager
Write-Host "[2/4] Spawning Parallel Checkpoint Archiver & Headroom Manager..." -ForegroundColor Yellow
$archiverJob = Start-Job -ScriptBlock {
    param($root, $py)
    Set-Location $root
    & $py "scripts\checkpoint_manager_s3.py" --interval 20 *> "out\checkpoint_manager_s3.log"
} -ArgumentList $RepoRoot, $Python

Write-Host "[3/4] Launching Session 3 Pre-Training Pipeline (Steps 4,001 -> 6,000)..." -ForegroundColor Green
Write-Host "      Live Log: $LogPath" -ForegroundColor Gray
Write-Host "      Live Dashboard: out\telemetry_dashboard.md" -ForegroundColor Gray

$trainExitCode = 0
try {
    # Execute launch script directly
    powershell -ExecutionPolicy Bypass -File "scripts\launch_s3_240m.ps1"
    $trainExitCode = $LASTEXITCODE
} catch {
    Write-Host "[!] Exception occurred during training: $_" -ForegroundColor Red
    $trainExitCode = 1
}

Write-Host "`n[training] Training process exited with code $trainExitCode." -ForegroundColor Cyan

# 3. Stop Background Telemetry Jobs
Write-Host "[telemetry] Stopping parallel telemetry background jobs..." -ForegroundColor Yellow
Stop-Job $telemetryJob -ErrorAction SilentlyContinue | Out-Null
Remove-Job $telemetryJob -ErrorAction SilentlyContinue | Out-Null
Stop-Job $archiverJob -ErrorAction SilentlyContinue | Out-Null
Remove-Job $archiverJob -ErrorAction SilentlyContinue | Out-Null

# 4. Trigger Automated Post-Training Telemetry & Evaluation Suite
if ($trainExitCode -eq 0) {
    Write-Host "`n[4/4] Executing Automated Post-Training Telemetry & Evaluation Suite..." -ForegroundColor Green
    & $Python "scripts\post_training_eval_s3.py"
    Write-Host "`n[+] Session 3 Supervised Pipeline Completed Successfully!" -ForegroundColor Green
} else {
    Write-Host "`n[!] Training did not exit cleanly (code $trainExitCode). Post-training evaluation skipped." -ForegroundColor Red
}

Write-Host "========================================================================" -ForegroundColor Cyan
