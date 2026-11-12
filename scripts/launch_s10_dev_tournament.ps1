# Engineered by uncoalesced
# Session 10 Dev Tournament Launcher

$ErrorActionPreference = "Stop"
$RepoRoot = "E:\Parentheses\parentheses-0.9"
Set-Location $RepoRoot

$LockFile = "$RepoRoot\out\locks\GPU.lock"
$PythonExe = "$RepoRoot\.venv\Scripts\python.exe"
$Script = "$RepoRoot\scripts\run_s10_dev_tournament.py"

# Check GPU lock
if (Test-Path $LockFile) {
    Write-Error "GPU lock already exists: $(Get-Content $LockFile)"
}

try {
    # Acquire lock
    "antigravity session10-dev-tournament $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss IST')" | Out-File -FilePath $LockFile -Encoding utf8
    Write-Host "[*] Acquired GPU.lock for Session 10 Dev Tournament"

    # Execute tournament
    & $PythonExe $Script --device cuda
    if ($LASTEXITCODE -ne 0) {
        throw "Dev tournament failed with exit code $LASTEXITCODE"
    }
    Write-Host "[+] Session 10 Dev Tournament completed successfully."
}
finally {
    if (Test-Path $LockFile) {
        Remove-Item $LockFile -Force -ErrorAction SilentlyContinue
        Write-Host "[*] Released GPU.lock"
    }
}
