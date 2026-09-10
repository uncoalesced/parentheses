# Unattended overnight run: download 22 Tier A/B parallel corpora -> report ->
# tokenize the combined corpus -> launch training detached.
#
# Nobody is watching this live, so every phase is bounded: each language gets a
# wall-clock timeout and one retry (zips are cached, so a retry resumes rather
# than restarting), the download phase as a whole has a deadline after which it
# stops and moves on with whatever it has, and the training step count is
# clamped at both ends so an unexpected corpus size can't produce a run that
# either finishes in minutes or is still going at noon.
#
# Run detached:
#   Start-Process powershell -ArgumentList '-ExecutionPolicy','Bypass','-File','scripts\nightly.ps1' -WorkingDirectory 'E:\Parentheses\parentheses-0.9' -NoNewWindow
#
# -StageSteps N: run training in sequential stages of N steps each instead of
# one long unattended launch, each stage resuming from the previous one's
# checkpoint (train.py --resume-from). The whole call blocks until every
# stage finishes -- still start it detached per the line above if nobody
# should have to keep a window open. Per-stage output goes to
# train_log_upto_<N>.txt / train_err_upto_<N>.txt so earlier stages' logs
# aren't overwritten by later ones. Example, continuing the locked
# 22-language config in 50k-step stages:
#   ... -TrainOnly -DataDir data\processed_multilingual_22 -CkptName multilingual-22 -StageSteps 50000
#
# PowerShell 5.1: no &&, no ternary, no ??.

# -TrainOnly skips the download/report/tokenize phases and runs only the
# sizing + launch step against -DataDir. Added 2026-08-31: the 22-language
# retrain needed the same epoch-target-and-clamp arithmetic over an
# already-tokenized corpus, and copying Get-MaxSteps into a second script
# would have meant two definitions of the number that decides how long the
# machine runs overnight.
param([switch]$SelfTest,
      [switch]$TrainOnly,
      [string]$DataDir  = 'data\processed_multilingual',
      [string]$CkptName = 'multilingual',
      # 0 = old behaviour: one detached, unattended launch (default,
      # unchanged). >0 = run in sequential stages of this many steps each,
      # synchronously, resuming from the previous stage's checkpoint via
      # train.py --resume-from -- so a crash or a deliberate stop loses at
      # most one stage, not the whole run, and the process driving the loop
      # can be watched or killed between stages.
      [int]$StageSteps = 0)

$ErrorActionPreference = 'Continue'

$Root      = Split-Path -Parent $PSScriptRoot
$Python    = Join-Path $Root 'venv\Scripts\python.exe'
$Log       = Join-Path $Root 'nightly_log.txt'
$LogErr    = Join-Path $Root 'nightly_log_errors.txt'
$Status    = Join-Path $Root 'nightly_status.txt'
$Scratch   = Join-Path $env:TEMP 'parentheses_nightly'
$TrainBin  = Join-Path $Root (Join-Path $DataDir 'train.bin')
$ValBin    = Join-Path $Root (Join-Path $DataDir 'val.bin')
$Utf8      = [System.Text.UTF8Encoding]::new($false)   # no BOM; grep/tail friendly

# --- knobs ------------------------------------------------------------------
$Langs = @('kn','hi','zh','fr','es','ar','ru','pt','uk','de','pl','ja','ko',
           'vi','tr','fa','bn','nl','id','ms','ta','ml')
$PerLangTimeoutSec  = 45 * 60      # one language, one attempt
$Attempts           = 3            # cached zips make a retry cheap
$RetryBackoffSec    = @(15, 60)    # waits before attempts 2 and 3
$DownloadDeadline   = (Get-Date).AddHours(6)

# Sizing the training run. 16384 = batch 64 x block_size 256 x grad-accum 1,
# train.py's defaults, which is also what the 162,000-step English run used.
$TokensPerStep = 16384
$EpochTarget   = 3.0               # 3 passes so every language is seen 3x
$StepFloor     = 162000            # the English run's exact step count = 2.65B
                                   # tokens of gradient signal, already shown
                                   # enough to hit this preset's loss ceiling
$StepCeiling   = 700000            # ~10.5h at the measured 18.5 steps/sec
$CkptEvery     = 2000              # ~108s between saves; a crash loses <2 min
$Preset        = 'parentheses-0.9-300k'
# Separate out-dir on purpose: train.py writes step_<N>.pt, and checkpoints/
# already holds the English run's step_1000..step_161000. Sharing the directory
# would overwrite step_161000.pt, the usable English result both handoffs and
# the README point at.
$CkptDir       = Join-Path $Root (Join-Path 'checkpoints' $CkptName)

New-Item -ItemType Directory -Force -Path $Scratch | Out-Null

function Invoke-SelfTest {
    $t = Join-Path ([System.IO.Path]::GetTempPath()) ("nightly_selftest_" + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Force -Path $t | Out-Null
    $f = Join-Path $t 'log.txt'
    $fail = 0
    try {
        [System.IO.File]::WriteAllText($f, "start`r`n", $Utf8)

        # Hold the file the way a `tail -f` does: open for read, sharing
        # read+write. This is the exact condition that silently killed last
        # night's log.
        $reader = [System.IO.FileStream]::new($f, [System.IO.FileMode]::Open,
                    [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
        try {
            # the new primitive must keep landing writes
            for ($i = 1; $i -le 5; $i++) {
                if (-not (Append-Line $f "held $i")) { Write-Output "FAIL: Append-Line blocked by a reader"; $fail++ }
            }
            $held = (Get-Content $f | Where-Object { $_ -like 'held *' }).Count
            if ($held -ne 5) { Write-Output "FAIL: expected 5 appended lines under a reader, got $held"; $fail++ }

            # and the old one must still demonstrate the bug, so this test
            # keeps meaning something if PowerShell's behaviour ever changes
            $broke = $false
            try { Add-Content -Path $f -Value 'via Add-Content' -Encoding utf8 -ErrorAction Stop } catch { $broke = $true }
            if (-not $broke) { Write-Output "NOTE: Add-Content no longer fails under a reader on this host" }
        } finally { $reader.Dispose() }

        # unheld appends obviously still work
        if (-not (Append-Line $f 'free')) { Write-Output 'FAIL: Append-Line failed with no reader'; $fail++ }

        # status file is rewritten whole and stays one line
        $Script:Status = Join-Path $t 'status.txt'
        Set-Status 'phase one'
        Set-Status 'phase two'
        $s = @(Get-Content $Script:Status)
        if ($s.Count -ne 1 -or $s[0] -notlike '*phase two') { Write-Output "FAIL: status not a single current line: $($s -join '|')"; $fail++ }

        # step sizing: normal, floored and clamped
        $n = Get-MaxSteps 1523000000 16384 3.0 162000 700000
        if ($n[0] -ne 279000) { Write-Output "FAIL: expected 279000 steps for last night's corpus, got $($n[0])"; $fail++ }
        $lo = Get-MaxSteps 10000000 16384 3.0 162000 700000
        if ($lo[0] -ne 162000 -or $lo[1] -notlike '*floor*') { Write-Output "FAIL: floor not applied: $($lo -join ' ')"; $fail++ }
        $hi = Get-MaxSteps 50000000000 16384 3.0 162000 700000
        if ($hi[0] -ne 700000 -or $hi[1] -notlike '*ceiling*') { Write-Output "FAIL: ceiling not applied: $($hi -join ' ')"; $fail++ }
        # the 22-language corpus, the run this switch was added for
        $n22 = Get-MaxSteps 1633099830 16384 3.0 162000 700000
        if ($n22[0] -ne 299000) { Write-Output "FAIL: expected 299000 steps for the 22-language corpus, got $($n22[0])"; $fail++ }

        # every launch argument is ONE element -- this is what the 13:55
        # failure looked like from the outside, and it is cheap to catch
        $ta = Get-TrainArgs 'parentheses-0.9-300k' 'data\processed_multilingual_22' 'multilingual-22' 299000 2000
        if ($ta.Count -ne 12) { Write-Output "FAIL: train args should be 12 elements, got $($ta.Count): $($ta -join '|')"; $fail++ }
        $di = [array]::IndexOf($ta, '--data')
        if ($ta[$di + 1] -ne 'data/processed_multilingual_22/train.bin') { Write-Output "FAIL: --data is '$($ta[$di + 1])'"; $fail++ }
        $oi = [array]::IndexOf($ta, '--out-dir')
        if ($ta[$oi + 1] -ne 'checkpoints/multilingual-22') { Write-Output "FAIL: --out-dir is '$($ta[$oi + 1])'"; $fail++ }
        if ($ta | Where-Object { $_ -match '\s' }) { Write-Output 'FAIL: a train arg contains whitespace'; $fail++ }

        # --resume-from: only appended when non-empty, forward slashes like
        # every other path train.py sees
        $taNoResume = Get-TrainArgs 'parentheses-0.9-300k' 'data\processed_multilingual_22' 'multilingual-22' 100000 2000 ''
        if ($taNoResume.Count -ne 12) { Write-Output "FAIL: no-resume train args should stay 12 elements, got $($taNoResume.Count)"; $fail++ }
        $taResume = Get-TrainArgs 'parentheses-0.9-300k' 'data\processed_multilingual_22' 'multilingual-22' 100000 2000 'checkpoints\multilingual-22\step_49999_final.pt'
        if ($taResume.Count -ne 14) { Write-Output "FAIL: resume train args should be 14 elements, got $($taResume.Count)"; $fail++ }
        $ri = [array]::IndexOf($taResume, '--resume-from')
        if ($ri -lt 0 -or $taResume[$ri + 1] -ne 'checkpoints/multilingual-22/step_49999_final.pt') { Write-Output "FAIL: --resume-from is '$($taResume[$ri + 1])'"; $fail++ }

        # Get-Stages: cumulative absolute targets, last one lands exactly on
        # the total rather than overshooting it
        $g1 = Get-Stages 299000 50000
        if (($g1 -join ',') -ne '50000,100000,150000,200000,250000,299000') { Write-Output "FAIL: Get-Stages 299000 50000 = $($g1 -join ',')"; $fail++ }
        $g2 = Get-Stages 100000 100000
        if (($g2 -join ',') -ne '100000') { Write-Output "FAIL: Get-Stages exact multiple should be one stage, got $($g2 -join ',')"; $fail++ }
        $g3 = Get-Stages 50000 0
        if (($g3 -join ',') -ne '50000') { Write-Output "FAIL: Get-Stages with stageSize 0 should disable staging, got $($g3 -join ',')"; $fail++ }
        $g4 = Get-Stages 90000 40000
        if (($g4 -join ',') -ne '40000,80000,90000') { Write-Output "FAIL: Get-Stages 90000 40000 = $($g4 -join ',')"; $fail++ }

        # -DataDir/-CkptName must still default to exactly where the
        # 20-language run wrote, or this refactor silently moved things
        if ($TrainBin -notlike '*data\processed_multilingual\train.bin') { Write-Output "FAIL: default TrainBin moved: $TrainBin"; $fail++ }
        if ($CkptDir -notlike '*checkpoints\multilingual')   { Write-Output "FAIL: default CkptDir moved: $CkptDir"; $fail++ }
    } catch {
        Write-Output "FAIL: unexpected exception: $_"
        $fail++
    } finally {
        Remove-Item -Recurse -Force $t -ErrorAction SilentlyContinue
    }
    if ($fail) { Write-Output "[self-test] nightly FAILED ($fail)"; exit 1 }
    Write-Output '[self-test] nightly ok'
}

# Append one line, tolerating a concurrent reader.
#
# This is the fix for the 2026-08-30 run, where nightly_log.txt stopped after
# its 4th line while the pipeline ran on correctly for another four hours.
# Add-Content was the cause: PowerShell's file provider opens the target for
# read+write (it sniffs the existing bytes to preserve encoding) and asks for
# an exclusive share mode, so *any* other process holding the file -- a
# `tail -f`, an editor -- makes the open fail with
#   IOException: The process cannot access the file ... used by another process
# Measured on this machine with a git-bash `tail -f` attached: Add-Content lost
# 22 of 40 appends; [IO.File]::AppendAllText, which opens write-only/append-only
# with FileShare.Read, lost 0 of 40. Set-Content is no more robust -- it only
# looked that way last night because nothing was tailing nightly_status.txt.
#
# The second half of the bug was silence: $ErrorActionPreference='Continue' in
# a -WindowStyle Hidden process with no stderr redirection meant those 22
# IOExceptions went nowhere at all. Failures now land in nightly_log_errors.txt.
function Append-Line($path, $line) {
    try {
        [System.IO.File]::AppendAllText($path, $line + [Environment]::NewLine, $Utf8)
        return $true
    } catch {
        return $false
    }
}

function Set-Status($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    try { [System.IO.File]::WriteAllText($Status, $line + [Environment]::NewLine, $Utf8) } catch { }
}

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Output $line
    if (-not (Append-Line $Log $line)) {
        # A lost log line must never be invisible again. The error file is a
        # different path, so whatever is holding the log is not holding this.
        Append-Line $LogErr ("[{0}] LOG WRITE FAILED, line was: {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg) | Out-Null
    }
    # nightly_status.txt is rewritten whole each time, so it is always exactly
    # one line and always current -- the cheapest "is this thing alive" check
    # even if the append log is being held by something.
    Set-Status $msg
}

# Extracted so the self-test can exercise the clamp arithmetic without running
# the pipeline. Returns the step count and the reason string.
function Get-MaxSteps($trainTokens, $tokensPerStep, $epochTarget, $floor, $ceiling) {
    $steps = [int]([math]::Round(($epochTarget * $trainTokens / $tokensPerStep) / 1000.0) * 1000)
    $why   = "$epochTarget epochs over $($trainTokens.ToString('N0')) tokens"
    if ($steps -lt $floor)   { $steps = $floor;   $why += " (raised to the English run's $floor-step floor)" }
    if ($steps -gt $ceiling) { $steps = $ceiling; $why += " (clamped to the $ceiling-step overnight ceiling)" }
    return @($steps, $why)
}

# Build train.py's argument list. Extracted so the self-test can assert on it:
# the 2026-08-31 13:55 launch died in under a second because
#   '--data', ($DataDir -replace '\\', '/') + '/train.bin'
# inside the @() literal parsed as two array elements rather than one
# concatenation, so train.py saw a stray '/train.bin' and exited on
# "unrecognized arguments". Anything computed now gets its own line and its
# own variable before it goes near the array.
function Get-TrainArgs($preset, $dataDir, $ckptName, $steps, $ckptEvery, $resumeFrom = '') {
    $dataPath = ($dataDir -replace '\\', '/') + '/train.bin'
    $outDir   = 'checkpoints/' + $ckptName
    $trainArgs = @('-u', 'train.py', '--preset', $preset,
             '--data', $dataPath,
             '--max-steps', "$steps", '--ckpt-every', "$ckptEvery",
             '--out-dir', $outDir)
    if ($resumeFrom) { $trainArgs += @('--resume-from', ($resumeFrom -replace '\\', '/')) }
    return $trainArgs
}

# Split a total step count into cumulative stage targets, each <= $stageSize
# apart, the last one landing exactly on $total rather than overshooting it.
# Each --resume-from run's --max-steps is one of these values (absolute, not
# an additional-steps count) -- see train.py's --resume-from.
function Get-Stages($total, $stageSize) {
    if ($stageSize -le 0 -or $stageSize -ge $total) { return @($total) }
    $stages = @()
    $s = $stageSize
    while ($s -lt $total) { $stages += $s; $s += $stageSize }
    $stages += $total
    return $stages
}

# Run python with a hard wall-clock cap. prepare_parallel.py's urlopen timeout
# is per-read, so a download trickling at 1 KB/s never trips it -- this does.
# Returns $true on a clean exit 0.
function Invoke-Bounded($argLine, $tag, $timeoutSec) {
    $out = Join-Path $Scratch "$tag.out"
    $err = Join-Path $Scratch "$tag.err"
    # -u: output is redirected to a file, so Python would block-buffer it --
    # which loses everything written so far if the timeout kills the process,
    # and makes train_log.txt lag minutes behind the run.
    $p = Start-Process -FilePath $Python -ArgumentList (@('-u') + $argLine) -WorkingDirectory $Root `
                       -NoNewWindow -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
    $ok = $p.WaitForExit($timeoutSec * 1000)
    if (-not $ok) {
        Say "  TIMEOUT after ${timeoutSec}s -- killing pid $($p.Id)"
        try { Stop-Process -Id $p.Id -Force -ErrorAction Stop } catch { Say "  kill failed: $_" }
        Start-Sleep -Seconds 2
    }
    foreach ($f in @($out, $err)) {
        if ((Test-Path $f) -and (Get-Item $f).Length -gt 0) {
            Get-Content $f | ForEach-Object { Append-Line $Log "    $_" | Out-Null }
        }
    }
    if (-not $ok) { return $false }
    # Start-Process -PassThru can hand back a null ExitCode; a process that
    # exited on its own is treated as a clean run rather than a spurious fail.
    $code = 0
    try { if ($null -ne $p.ExitCode) { $code = $p.ExitCode } } catch { }
    return ($code -eq 0)
}

# One staged training run: like Invoke-Bounded (timeout, kill on hang), but
# for a long training subprocess rather than a short bounded one -- output
# streams straight to its own log files instead of being buffered and
# replayed into nightly_log.txt afterward (a stage can log thousands of
# lines), and success additionally requires the checkpoint the stage was
# supposed to produce to actually be on disk, not just a zero exit code.
function Invoke-TrainStage($trainArgs, $stageEnd, $ckptDir, $timeoutSec) {
    $outLog = Join-Path $Root "train_log_upto_$stageEnd.txt"
    $errLog = Join-Path $Root "train_err_upto_$stageEnd.txt"
    $p = Start-Process -FilePath $Python -ArgumentList $trainArgs -WorkingDirectory $Root `
                       -NoNewWindow -PassThru `
                       -RedirectStandardOutput $outLog -RedirectStandardError $errLog
    Set-Content -Path (Join-Path $Root 'train_pid.txt') -Value $p.Id -Encoding utf8
    Say "  stage pid $($p.Id) -> $(Split-Path -Leaf $outLog) / $(Split-Path -Leaf $errLog)"
    $ok = $p.WaitForExit($timeoutSec * 1000)
    if (-not $ok) {
        Say "  TIMEOUT after ${timeoutSec}s -- killing pid $($p.Id)"
        try { Stop-Process -Id $p.Id -Force -ErrorAction Stop } catch { Say "  kill failed: $_" }
        return $false
    }
    $code = 0
    try { if ($null -ne $p.ExitCode) { $code = $p.ExitCode } } catch { }
    if ($code -ne 0) { Say "  stage exited $code -- see $(Split-Path -Leaf $errLog)"; return $false }
    $ckptPath = Join-Path $ckptDir "step_$($stageEnd - 1)_final.pt"
    if (-not (Test-Path $ckptPath)) {
        Say "  stage exited 0 but expected checkpoint is missing: $ckptPath"
        return $false
    }
    return $true
}

# Dispatched here, not next to the function: PowerShell executes top to bottom,
# so Append-Line/Set-Status/Get-MaxSteps have to be defined before the
# self-test calls them.
if ($SelfTest) { Invoke-SelfTest; return }

# One generation of history: last night's log was the only record that the run
# had gone quiet, so don't clobber it on the next start.
if (Test-Path $Log) { Move-Item -Force $Log (Join-Path $Root 'nightly_log.prev.txt') -ErrorAction SilentlyContinue }
Remove-Item $LogErr -ErrorAction SilentlyContinue

Say "=== nightly run starting ==="
Say "python: $Python"
Say "data: $DataDir -> checkpoints\$CkptName"

if ($TrainOnly) {
    Say "-TrainOnly: skipping download, report and tokenize"
    if (-not (Test-Path $TrainBin)) {
        Say "FATAL: -TrainOnly but no train.bin at $TrainBin"
        Set-Status "FAILED: -TrainOnly with no train.bin"
        exit 1
    }
} else {
Say "download deadline: $($DownloadDeadline.ToString('yyyy-MM-dd HH:mm:ss'))"

# --- phase 1: download ------------------------------------------------------
$done = @(); $failed = @(); $skipped = @()
$i = 0
foreach ($L in $Langs) {
    $i++
    if ((Get-Date) -gt $DownloadDeadline) {
        Say "download deadline passed -- skipping remaining languages"
        $skipped += $Langs[($i - 1)..($Langs.Count - 1)]
        break
    }
    Set-Status "downloading en-$L ($i/$($Langs.Count))"
    # Ground truth for "this language worked" is a manifest written by this
    # attempt -- prepare_parallel.py exits 0 even when an individual corpus
    # download fails (it records the failure in stats and carries on) and on
    # the "no openly-licensed corpus within the size cap" warning path too.
    $manifest = Join-Path $Root "data\raw\parallel\en-$L\manifest.json"
    $ok = $false
    for ($a = 1; $a -le $Attempts -and -not $ok; $a++) {
        if ($a -gt 1) {
            # Backoff, added after the 2026-08-30 run. pt and uk were the only
            # 2 of 22 to fail, and both died in opus_index() -- the OPUS *API*
            # call, not a download -- on a transient SSLEOFError/HTTP 500. The
            # retry fired 1-2 seconds later, straight back into the same blip.
            # pl and id hit the identical error and only survived because their
            # retry happened to land after it passed.
            $wait = $RetryBackoffSec[[math]::Min($a - 2, $RetryBackoffSec.Count - 1)]
            Say "en-${L}: waiting ${wait}s before retry"
            Start-Sleep -Seconds $wait
        }
        Say "en-${L}: attempt $a/$Attempts"
        $started = Get-Date
        # Defaults only: no --max-download-mb, no --force-corpus, no --allow.
        $ran = Invoke-Bounded @('data\prepare_parallel.py', '--lang', $L) "dl_${L}_$a" $PerLangTimeoutSec
        $ok = $ran -and (Test-Path $manifest) -and ((Get-Item $manifest).LastWriteTime -ge $started)
        if (-not $ok) { Say "en-${L}: attempt $a failed" }
    }
    if ($ok) { Say "en-${L}: OK"; $done += $L } else { Say "en-${L}: GIVING UP"; $failed += $L }
}
Say "downloads finished: $($done.Count) ok, $($failed.Count) failed, $($skipped.Count) skipped"
if ($failed.Count)  { Say "  failed:  $($failed  -join ', ')" }
if ($skipped.Count) { Say "  skipped: $($skipped -join ', ')" }

# --- phase 2: report --------------------------------------------------------
Set-Status "writing corpus report"
Say "--- prepare_parallel --report ---"
Invoke-Bounded @('data\prepare_parallel.py', '--report') 'report' 600 | Out-Null

# --- phase 3: tokenize ------------------------------------------------------
Set-Status "tokenizing combined corpus"
Say "--- tokenize_corpus (wikipedia + books + parallel) ---"
$tok = Invoke-Bounded @('data\tokenize_corpus.py',
                        '--raw-dirs', 'data\raw\wikipedia', 'data\raw\books', 'data\raw\parallel',
                        '--out-dir', $DataDir) 'tokenize' (4 * 3600)
if (-not $tok -or -not (Test-Path $TrainBin)) {
    Say "FATAL: tokenize failed, no train.bin -- not launching training"
    Set-Status "FAILED: tokenize did not produce train.bin"
    exit 1
}
}   # end of the three phases -TrainOnly skips

# --- phase 4: size and launch training --------------------------------------
$trainTokens = [int64]((Get-Item $TrainBin).Length / 2)
$valTokens   = [int64]((Get-Item $ValBin).Length / 2)
Say "corpus: $($trainTokens.ToString('N0')) train / $($valTokens.ToString('N0')) val byte tokens"

$r     = Get-MaxSteps $trainTokens $TokensPerStep $EpochTarget $StepFloor $StepCeiling
$steps = $r[0]
$why   = $r[1]
$hours = [math]::Round($steps * 0.043 / 3600.0, 1)   # 0.043 s/step measured on the 279k run
Say "max-steps = $steps -- $why; ~${hours}h at the measured 0.043 s/step"

New-Item -ItemType Directory -Force -Path $CkptDir | Out-Null

if ($StageSteps -le 0) {
    # Old behaviour, unchanged: one detached, unattended launch. Whoever
    # started this script (or the process that started it) does not block on
    # training -- it runs on after "driver done" and is watched via
    # nightly_status.txt / train_log.txt.
    $trainArgs = Get-TrainArgs $Preset $DataDir $CkptName $steps $CkptEvery
    Say "launching: $Python $($trainArgs -join ' ')"

    $tp = Start-Process -FilePath $Python -ArgumentList $trainArgs -WorkingDirectory $Root `
                        -NoNewWindow -PassThru `
                        -RedirectStandardOutput (Join-Path $Root 'train_log.txt') `
                        -RedirectStandardError  (Join-Path $Root 'train_err.txt')
    Set-Content -Path (Join-Path $Root 'train_pid.txt') -Value $tp.Id -Encoding utf8
    Say "training pid $($tp.Id) -> train_log.txt / train_err.txt"
    Set-Status "training pid $($tp.Id): $Preset, $steps steps, ckpt every $CkptEvery -> checkpoints/$CkptName"
    Say "=== driver done; training continues detached ==="
} else {
    # Staged: run $StageSteps at a time, synchronously, each stage resuming
    # from the previous stage's checkpoint via train.py --resume-from. This
    # call blocks until every stage finishes (or one fails) -- start the
    # whole nightly.ps1 invocation detached (per the header comment) if
    # nobody should have to keep a window open for however many stages this
    # is. Each stage gets its own train_log_upto_<N>.txt / train_err_upto_<N>.txt
    # so an earlier stage's output survives the next stage's launch, unlike
    # the single-launch path's one shared train_log.txt.
    $stages = Get-Stages $steps $StageSteps
    Say "staged run: $($stages.Count) stage(s), target steps $($stages -join ', ')"
    $resumeFrom = ''
    $prevEnd = 0
    foreach ($stageEnd in $stages) {
        $stageStepsThis = $stageEnd - $prevEnd
        # 0.043 s/step measured on the 279k run, x1.5 for headroom, floored
        # so a small stage isn't timed out by model/CUDA init overhead alone.
        $timeout = [math]::Max(1800, [int]($stageStepsThis * 0.043 * 1.5))
        Set-Status "training stage $prevEnd -> $stageEnd of $steps"
        Say "--- stage $prevEnd -> $stageEnd ($stageStepsThis steps, timeout ${timeout}s, resume='$resumeFrom') ---"
        $trainArgs = Get-TrainArgs $Preset $DataDir $CkptName $stageEnd $CkptEvery $resumeFrom
        $ok = Invoke-TrainStage $trainArgs $stageEnd $CkptDir $timeout
        if (-not $ok) {
            Say "FATAL: stage ending at $stageEnd failed -- stopping, later stages not attempted"
            Set-Status "FAILED: training stage $stageEnd"
            exit 1
        }
        $resumeFrom = Join-Path $CkptDir "step_$($stageEnd - 1)_final.pt"
        Say "stage $stageEnd done -- $resumeFrom"
        $prevEnd = $stageEnd
    }
    Say "=== driver done; all $($stages.Count) stage(s) complete; final checkpoint: $resumeFrom ==="
    Set-Status "training complete: $resumeFrom"
}
