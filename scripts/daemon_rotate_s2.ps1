# Engineered by uncoalesced
$ErrorActionPreference = "Stop"

$CkptDir = "E:\Parentheses\parentheses-0.9\checkpoints\parentheses-1.0-240M-s2"
$ColdDir = "I:\parentheses\checkpoints\parentheses-1.0-240M-s2"
$ManifestPath = Join-Path $ColdDir "MANIFEST_s2_sha256.tsv"
$LogFile = "E:\Parentheses\parentheses-0.9\out\daemon_rotate_s2.log"
$PythonExe = "E:\Parentheses\parentheses-0.9\.venv\Scripts\python.exe"
$RepoRoot = "E:\Parentheses\parentheses-0.9"
$SyncTarget = "C:\Users\Joel Anthony\Documents\Phile\Heafod\Heafod\Parentheses\parentheses-1.0-240M"

New-Item -ItemType Directory -Force -Path $ColdDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $LogFile) | Out-Null

function LogMsg($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line
    $line | Out-File -FilePath $LogFile -Append -Encoding utf8
}

LogMsg "=== Starting Session 2 Checkpoint Rotator & Watchdog Daemon ==="

while ($true) {
    try {
        if (Test-Path $CkptDir) {
            $files = Get-ChildItem -Path $CkptDir -Filter "step_*.pt" | Sort-Object CreationTime
            foreach ($f in $files) {
                $stepName = $f.Name
                $coldDest = Join-Path $ColdDir $stepName
                $latestE = Join-Path $CkptDir "latest.pt"

                # 1. Copy to cold storage if not already there
                if (-not (Test-Path $coldDest)) {
                    Copy-Item -Path $f.FullName -Destination $coldDest -Force
                    $sha = (Get-FileHash -Path $coldDest -Algorithm SHA256).Hash.ToLower()
                    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
                    $manifestLine = "$sha`t$stepName`t$ts"
                    Add-Content -Path $ManifestPath -Value $manifestLine -Encoding utf8
                    LogMsg "[ARCHIVED] Copied $stepName to I: and logged SHA256 $sha"
                }

                # 2. Copy to latest.pt on E:
                Copy-Item -Path $f.FullName -Destination $latestE -Force

                # 3. If step is 3000 or 4000, run validation and evaluation
                if ($stepName -match "step_(3000|4000)\.pt") {
                    LogMsg "[VALIDATION] Running validate_s1_anchor.py and eval_s1_generations.py for $stepName..."
                    Set-Location $RepoRoot
                    & $PythonExe scripts/validate_s1_anchor.py --ckpt $coldDest | Out-File -FilePath (Join-Path $RepoRoot "out/validate_$stepName.log") -Encoding utf8
                    & $PythonExe scripts/eval_s1_generations.py --ckpt $coldDest | Out-File -FilePath (Join-Path $RepoRoot "out/eval_$stepName.log") -Encoding utf8
                    LogMsg "[VALIDATION] Completed validation for $stepName"
                }

                # 4. Remove raw step file on E: to keep NVMe space clean
                Remove-Item -Path $f.FullName -Force
                LogMsg "[PRUNED] Removed raw $stepName from E: (kept as latest.pt)"
            }
        }

        # 5. Run digest plotter & sync to Heafod vault
        Set-Location $RepoRoot
        $jsonl = Join-Path $RepoRoot "out\session2_240m_metrics.jsonl"
        $mdOut = Join-Path $RepoRoot "out\session2_digest.md"
        $pngOut = Join-Path $RepoRoot "out\session2_digest.png"
        if (Test-Path $jsonl) {
            & $PythonExe scripts/plot_s1_morning_digest.py --jsonl $jsonl --out-md $mdOut --out-png $pngOut
            if (Test-Path $SyncTarget) {
                Copy-Item -Path $mdOut -Destination $SyncTarget -Force -ErrorAction SilentlyContinue
                Copy-Item -Path $pngOut -Destination $SyncTarget -Force -ErrorAction SilentlyContinue
            }
        }
    } catch {
        LogMsg "[ERROR] Exception in rotator daemon: $_"
    }
    Start-Sleep -Seconds 60
}
