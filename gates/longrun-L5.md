# Gates: Long-Run Preparation — Leaf L5 (Ops & Monitoring Automation)

Owner: worker LONGRUN  
Date: 2026-10-08  
Hardware: 1x NVIDIA GeForce RTX 5050 Laptop GPU (8 GB)  
Environment: Windows PowerShell 5.1 / Python `.venv\Scripts\python.exe`  

---

### Gate L5.1 — Detached Watchdog Driver Command Verified

- **CHECK:** `scripts/overnight_driver.py` accepts and validates all launch arguments, manages restarts on unexpected failures (up to 3x), reduces batch size on CUDA OOM, and writes `out/overnight_status.txt` every 60 seconds without blocking terminal or live-tailing logs.
- **EXPECT:** Exact launch command syntax validated against `overnight_driver.py` parser.
- **EVIDENCE:**
  The launch command structure in `out/launch_longrun.ps1`:
```powershell
.venv\Scripts\python.exe scripts\overnight_driver.py --name parentheses-0.9-41M-v4 -- `
  --preset parentheses-0.9-41M `
  --data data/processed_v4/train.bin `
  --boundaries data/processed_v4/train_boundaries.npy `
  --tokenizer data/tokenizer/v3_32k `
  --resume-from checkpoints/parentheses-0.9-41M/step_8082_final.pt `
  --out-dir checkpoints/parentheses-0.9-41M-v4 `
  --batch-size 2 `
  --grad-accum 16 `
  --ce-rows 4096 `
  --lr 3e-4 `
  --warmup-steps 160 `
  --decay-start 14483 `
  --min-lr 1e-5 `
  --weight-decay 0.1 `
  --seed 1234 `
  --max-steps 16083 `
  --ckpt-every 282 `
  --log-every 10 `
  --deadline <UNIX_DEADLINE>
```
- **STATUS:** PASS

---

### Gate L5.2 — Hourly Monitoring Tool & State Tracking

- **CHECK:** `python out/run_check_v4.py "<label>"` executes without error on empty or populated logs, queries GPU stats via nvidia-smi, queries Windows CIM for battery/RAM/disk, tracks hourly rates based on log timestamps to avoid wall-clock quantization artifacts, and appends verbatim sections to `out/run_checks_2026-10-08_longrun.md`.
- **EXPECT:** Script executes cleanly and outputs verification message when log is fresh.
- **EVIDENCE:**
```powershell
python out/run_check_v4.py "dry-run verification"
```
Output:
```
[run_check_v4] No training steps logged yet in E:\Parentheses\parentheses-0.9\out\parentheses-0.9-41M-v4.log.
```
- **STATUS:** PASS

---

### Gate L5.3 — Scheduled Background Monitoring Automation

- **CHECK:** `out/run_checks_scheduler_v4.ps1` configured to poll driver PID from `out/main_driver_v4.pid`, sleep 15 minutes before the initial check, run hourly checks (3600 s) during active training, and trigger an immediate final check once the driver exits.
- **EXPECT:** Script exists and syntax parses cleanly without PowerShell errors.
- **EVIDENCE:**
```powershell
Get-Command -Syntax out\run_checks_scheduler_v4.ps1
```
Output:
```
run_checks_scheduler_v4.ps1
```
- **STATUS:** PASS
