# Engineered by uncoalesced
# Live Checkpoint Archiver & Rotator for Parentheses 0.9-130M-v3 (Project Tanzinite)
# Keeps latest 3 checkpoints on E:, archives 2,500-step milestones and final checkpoints to I:

param(
    [string]$CkptDir = "E:\Parentheses\parentheses-0.9\checkpoints\parentheses-0.9-130M-v3",
    [string]$ArchiveDir = "I:\parentheses\checkpoints\parentheses-0.9-130M-v3",
    [int]$KeepCount = 3,
    [int]$IntervalSec = 300,
    [string]$LogFile = "E:\Parentheses\parentheses-0.9\out\checkpoint_rotator.log"
)

function LogMsg($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line
    $line | Out-File -FilePath $LogFile -Append -Encoding utf8
}

New-Item -ItemType Directory -Force -Path $ArchiveDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $LogFile) | Out-Null

LogMsg "=== Starting 130M Checkpoint Archiver & Rotator Daemon ==="
LogMsg "Source: $CkptDir | Archive: $ArchiveDir | Retaining latest $KeepCount checkpoints on E:"

while ($true) {
    try {
        if (Test-Path $CkptDir) {
            $files = Get-ChildItem -Path $CkptDir -Filter "step_*.pt" | Sort-Object LastWriteTime
            if ($files.Count -gt $KeepCount) {
                $toPrune = $files | Select-Object -First ($files.Count - $KeepCount)
                foreach ($f in $toPrune) {
                    # Archive every 2500th step milestone or final checkpoint to I:\, prune intermediate 500-step snapshots from E:\
                    if ($f.Name -match "step_0*(2500|5000|7500|10000|12500|15000|17500|20000|22500|.*final.*)\.pt") {
                        $dest = Join-Path $ArchiveDir $f.Name
                        Move-Item -Path $f.FullName -Destination $dest -Force
                        LogMsg "[ARCHIVED] Moved milestone checkpoint $($f.Name) -> $ArchiveDir"
                    } else {
                        Remove-Item -Path $f.FullName -Force
                        LogMsg "[PRUNED] Removed intermediate checkpoint $($f.Name) from E: (retaining latest $KeepCount)"
                    }
                }
            }
        }
    } catch {
        LogMsg "[ERROR] Exception during rotation: $_"
    }
    Start-Sleep -Seconds $IntervalSec
}
