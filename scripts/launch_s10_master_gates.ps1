# Engineered by uncoalesced
# Session 10 Master Gates (G1-G9) Launcher

$ErrorActionPreference = "Stop"
$RepoRoot = "E:\Parentheses\parentheses-0.9"
Set-Location $RepoRoot

$LockFile = "$RepoRoot\out\locks\GPU.lock"
$PythonExe = "$RepoRoot\.venv\Scripts\python.exe"
$Script = "$RepoRoot\scripts\run_s10_master_gates.py"
$Checkpoint = "$RepoRoot\checkpoints\parentheses-1.0-240M-s10-patch\step_14200_golden.pt"

# Check GPU lock
if (Test-Path $LockFile) {
    Write-Error "GPU lock already exists: $(Get-Content $LockFile)"
}

try {
    # Acquire lock
    "antigravity session10-master-gates $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss IST')" | Out-File -FilePath $LockFile -Encoding utf8
    Write-Host "[*] Acquired GPU.lock for Session 10 Master Gates"

    # Execute master gates
    & $PythonExe $Script --checkpoint $Checkpoint --device cuda
    if ($LASTEXITCODE -ne 0) {
        throw "Master gates evaluation failed with exit code $LASTEXITCODE"
    }
    Write-Host "[+] Session 10 Master Gates evaluation completed successfully."
}
finally {
    if (Test-Path $LockFile) {
        Remove-Item $LockFile -Force -ErrorAction SilentlyContinue
        Write-Host "[*] Released GPU.lock"
    }
}
