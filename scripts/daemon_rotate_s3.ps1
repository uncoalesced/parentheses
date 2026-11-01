<#
.SYNOPSIS
    Daemon for rotating and archiving Parentheses 1.0-240M Session 3 checkpoints.
.DESCRIPTION
    Monitors E:\Parentheses\parentheses-0.9\checkpoints\parentheses-1.0-240M-s3\
    Copies step_*.pt checkpoints to I:\parentheses\checkpoints\parentheses-1.0-240M-s3\
    Computes SHA256 hashes and updates MANIFEST_s3_sha256.tsv.
#>

$SourceDir = "E:\Parentheses\parentheses-0.9\checkpoints\parentheses-1.0-240M-s3"
$ArchiveDir = "I:\parentheses\checkpoints\parentheses-1.0-240M-s3"
$ManifestPath = Join-Path $ArchiveDir "MANIFEST_s3_sha256.tsv"

if (!(Test-Path $ArchiveDir)) {
    New-Item -ItemType Directory -Force -Path $ArchiveDir | Out-Null
}

if (!(Test-Path $ManifestPath)) {
    Set-Content -Path $ManifestPath -Value "sha256\tfilename\tstep\ttimestamp" -Encoding utf8
}

Write-Host "[daemon] Starting Session 3 Checkpoint Rotation Daemon..." -ForegroundColor Cyan

while ($true) {
    if (Test-Path $SourceDir) {
        $checkpoints = Get-ChildItem -Path $SourceDir -Filter "step_*.pt"
        foreach ($cp in $checkpoints) {
            $destPath = Join-Path $ArchiveDir $cp.Name
            if (!(Test-Path $destPath)) {
                Write-Host "[daemon] Copying $($cp.Name) to cold storage I:\..." -ForegroundColor Yellow
                Copy-Item -Path $cp.FullName -Destination $destPath -Force
                
                # Compute SHA256
                $hash = (Get-FileHash -Path $destPath -Algorithm SHA256).Hash.ToLower()
                $step = [regex]::Match($cp.Name, 'step_(\d+)').Groups[1].Value
                if ([string]::IsNullOrEmpty($step)) { $step = "unknown" }
                $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
                
                Add-Content -Path $ManifestPath -Value "$hash`t$($cp.Name)`t$step`t$timestamp" -Encoding utf8
                Write-Host "[daemon] Archived and recorded SHA256 for $($cp.Name)." -ForegroundColor Green
            }
        }
    }
    Start-Sleep -Seconds 30
}
