# Engineered by uncoalesced
# Session 10 Master Release Gates (G1 - G9) on Fresh Held-Out Suite (300 prompts)

$ErrorActionPreference = "Stop"
$RepoRoot = "E:\Parentheses\parentheses-0.9"
Set-Location $RepoRoot

$Lock = "$RepoRoot\out\locks\GPU.lock"
$PythonExe = "$RepoRoot\.venv\Scripts\python.exe"
$ChampionCkpt = "checkpoints/parentheses-1.0-240M-s10-blend/sft-nudge-winner.pt"
$HeldoutSuite = "data/eval_suites/suite_v3_heldout.jsonl"
$OutDir = "out/s10"

if (Test-Path $Lock) {
    Write-Error "GPU lock already exists: $(Get-Content $Lock)"
}

$LogDir = "$RepoRoot\out\s10"
if (!(Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

$LogFile = "$LogDir\master_heldout_gates.log"
"antigravity session10-heldout-gates $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss IST')" | Out-File -FilePath $Lock -Encoding utf8
Write-Host "[*] Acquired GPU.lock for S10 Master Held-Out Gates"

try {
    & $PythonExe scripts/run_s10_master_gates.py `
        --checkpoint $ChampionCkpt `
        --test-suite $HeldoutSuite `
        --device cuda `
        --out-dir $OutDir 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $LogFile
    
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        throw "run_s10_master_gates.py exited with code $code"
    }
    Write-Host "[+] S10 Master Held-Out Gates completed successfully."
} finally {
    if (Test-Path $Lock) {
        Remove-Item $Lock -Force -ErrorAction SilentlyContinue
        Write-Host "[*] Released GPU.lock"
    }
}
