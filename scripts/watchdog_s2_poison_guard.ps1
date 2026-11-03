param(
    [string]$LogPath = "E:\Parentheses\parentheses-0.9\out\session2_240m_live.log",
    [string]$PatrolPath1 = "E:\Parentheses\parentheses-0.9\out\agron_overnight_patrol.md",
    [string]$PatrolPath2 = "C:\Users\Joel Anthony\Documents\Phile\Heafod\Heafod\Parentheses\parentheses-1.0-240M\agron_overnight_patrol.md",
    [string]$HaltPath1 = "E:\Parentheses\parentheses-0.9\out\EMERGENCY_HALT.txt",
    [string]$HaltPath2 = "C:\Users\Joel Anthony\Documents\Phile\Heafod\Heafod\Parentheses\parentheses-1.0-240M\EMERGENCY_HALT.txt"
)

$consecutiveGradExplosion = 0
$lastProcessedStep = -1

while ($true) {
    Start-Sleep -Seconds 20

    if (-not (Test-Path $LogPath)) {
        continue
    }

    $lines = Get-Content $LogPath -Tail 5
    if (-not $lines) { continue }

    $latestStepLine = $lines | Where-Object { $_ -match "step\s+(\d+)" } | Select-Object -Last 1
    if (-not $latestStepLine) { continue }

    if ($latestStepLine -match "step\s+(\d+)\s+\|\s+loss\s+([0-9\.e+-]+)\s+\|\s+lr\s+([0-9\.e+-]+)\s+\|\s+grad_norm\s+([0-9\.e+-]+)\s+\|\s+tok/s\s+(\d+)\s+\|\s+v0_mix\s+([0-9\.e+-]+)\s+\|\s+beta_write\s+([0-9\.e+-]+)") {
        $step = [int]$matches[1]
        $lossStr = $matches[2]
        $lr = $matches[3]
        $gradNormStr = $matches[4]
        $toks = [int]$matches[5]
        $v0Mix = $matches[6]
        $betaWrite = $matches[7]

        $loss = 0.0
        $gradNorm = 0.0
        $isLossNan = ($lossStr -eq "NaN" -or $lossStr -eq "Inf" -or -not [double]::TryParse($lossStr, [ref]$loss))
        $isGradNan = ($gradNormStr -eq "NaN" -or $gradNormStr -eq "Inf" -or -not [double]::TryParse($gradNormStr, [ref]$gradNorm))

        $gpuInfo = nvidia-smi --query-gpu=temperature.gpu,memory.used,power.draw --format=csv,noheader,nounits 2>&1
        $temp = 0
        $vram = 0
        $watt = 0
        if ($gpuInfo -match "^\s*(\d+),\s*(\d+),\s*([0-9.]+)") {
            $temp = [int]$matches[1]
            $vram = [int]$matches[2]
            $watt = [double]$matches[3]
        }

        $haltReason = $null

        if (-not $isLossNan -and $loss -lt 1.45) {
            $haltReason = "Poisoned run / zero-collapse detected: loss $loss is below 1.45 threshold."
        }

        if ($isLossNan -or $isGradNan) {
            $haltReason = "Numerical divergence detected: loss or grad_norm is NaN/Inf (loss: $lossStr, grad_norm: $gradNormStr)."
        }

        if (-not $isGradNan -and $gradNorm -gt 10.0) {
            $consecutiveGradExplosion++
            if ($consecutiveGradExplosion -ge 3) {
                $haltReason = "Gradient explosion detected: grad_norm > 10.0 for 3 consecutive updates ($gradNorm)."
            }
        } else {
            $consecutiveGradExplosion = 0
        }

        if ($toks -gt 0 -and $toks -lt 1200) {
            $haltReason = "WDDM spillover / low throughput detected: tok/s drops to $toks (< 1200)."
        }

        if ($temp -gt 82) {
            $haltReason = "Thermal envelope breached: GPU temperature is ${temp}C (> 82C)."
        }

        if ($haltReason) {
            $dateStr = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
            $msg = "[EMERGENCY HALT at $dateStr] Step $step`: $haltReason"
            Set-Content -Path $HaltPath1 -Value $msg
            Set-Content -Path $HaltPath2 -Value $msg

            Stop-Process -Name "python" -Force -ErrorAction SilentlyContinue
            break
        }

        if ($step -ne $lastProcessedStep -and ($step % 100 -eq 0)) {
            $lastProcessedStep = $step
            $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            $freeE = (Get-PSDrive E).Free / 1GB
            $row = "| $timestamp IST | $step | $loss | $lr | $gradNorm | $betaWrite | $v0Mix | $toks | $vram | $temp | $watt | $([math]::Round($freeE, 1)) |"
            
            if (Test-Path $PatrolPath1) { Add-Content -Path $PatrolPath1 -Value $row }
            if (Test-Path $PatrolPath2) { Add-Content -Path $PatrolPath2 -Value $row }
        }
    }
}
