# Engineered by uncoalesced
# Session 10 Patch Dev Selection Tournament Launcher
# Evaluates all candidate checkpoints from patch v2 on suite_v2_dev.jsonl (300 prompts)

$ErrorActionPreference = "Stop"
$RepoRoot = "E:\Parentheses\parentheses-0.9"
Set-Location $RepoRoot

$Lock = "$RepoRoot\out\locks\GPU.lock"
$PythonExe = "$RepoRoot\.venv\Scripts\python.exe"
$CkptDir = "I:\parentheses\checkpoints\parentheses-1.0-240M-s10-patch-v2"
$OutDir = "out/s10/dev_tournament_patch"

if (Test-Path $Lock) {
    Write-Error "GPU lock already exists: $(Get-Content $Lock)"
}

$LogDir = "$RepoRoot\out\s10"
if (!(Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

$LogFile = "$LogDir\patch_dev_tournament.log"
"antigravity session10-dev-tournament-patch $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss IST')" | Out-File -FilePath $Lock -Encoding utf8
Write-Host "[*] Acquired GPU.lock for S10 Patch Dev Tournament"

try {
    & $PythonExe scripts/run_s10_dev_tournament.py `
        --checkpoints-dir $CkptDir `
        --suite "data/eval_suites/suite_v2_dev.jsonl" `
        --device cuda `
        --out-dir $OutDir 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $LogFile
    
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        throw "run_s10_dev_tournament.py exited with code $code"
    }
    Write-Host "[+] S10 Patch Dev Tournament completed successfully."
} finally {
    if (Test-Path $Lock) {
        Remove-Item $Lock -Force -ErrorAction SilentlyContinue
        Write-Host "[*] Released GPU.lock"
    }
}
