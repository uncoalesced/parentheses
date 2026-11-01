<#
.SYNOPSIS
    Watchdog circuit-breaker for Parentheses 1.0-240M Session 3 pre-training.
.DESCRIPTION
    Polls session3_240m_live.log every 20 seconds. Triggers abort/alarm if:
    - NaN or Inf in loss
    - Loss < 1.45 (anomaly collapse)
    - Grad norm > 10.0 (explosion)
    - GPU temperature > 82°C
#>

$LogPath = "E:\Parentheses\parentheses-0.9\out\session3_240m_live.log"
Write-Host "[watchdog] Starting Session 3 Poison & Thermal Watchdog..." -ForegroundColor Cyan

while ($true) {
    # Check GPU temperature via nvidia-smi
    try {
        $tempOutput = nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits
        $temp = [int]$tempOutput.Trim()
        if ($temp -gt 82) {
            Write-Warning "[watchdog] CRITICAL THERMAL ALERT: GPU temperature is ${temp}C (> 82C)! Aborting training..."
            Get-Process -Name python -ErrorAction SilentlyContinue | Stop-Process -Force
            break
        }
    } catch {
        # ignore if nvidia-smi fails momentarily
    }

    if (Test-Path $LogPath) {
        $tail = Get-Content -Path $LogPath -Tail 20 -ErrorAction SilentlyContinue
        foreach ($line in $tail) {
            if ($line -match "nan" -or $line -match "inf") {
                Write-Warning "[watchdog] CRITICAL: NaN or Inf detected in log! Aborting training..."
                Get-Process -Name python -ErrorAction SilentlyContinue | Stop-Process -Force
                break
            }
            if ($line -match "loss:\s*([0-9\.]+)") {
                $loss = [float]$Matches[1]
                if ($loss -lt 1.45) {
                    Write-Warning "[watchdog] CRITICAL: Loss anomaly detected ($loss < 1.45)! Aborting training..."
                    Get-Process -Name python -ErrorAction SilentlyContinue | Stop-Process -Force
                    break
                }
            }
            if ($line -match "grad_norm:\s*([0-9\.]+)") {
                $gn = [float]$Matches[1]
                if ($gn -gt 10.0) {
                    Write-Warning "[watchdog] CRITICAL: Gradient explosion detected (grad_norm = $gn > 10.0)! Aborting training..."
                    Get-Process -Name python -ErrorAction SilentlyContinue | Stop-Process -Force
                    break
                }
            }
        }
    }
    Start-Sleep -Seconds 20
}
