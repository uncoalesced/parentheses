# Gates: Long-Run Preparation — Leaf L3 (Eval Additions & Baseline)

Owner: worker LONGRUN  
Date: 2026-10-08  
Hardware: 1x NVIDIA GeForce RTX 5050 Laptop GPU (8 GB)  
Environment: Windows, Python `.venv\Scripts\python.exe` (torch dev cu128), `PYTHONIOENCODING=utf-8`  

---

### Gate L3.1 — `scripts/eval_dravidian.py` Self-Test with Repetition & Distinct Metrics

- **CHECK:** `.venv\Scripts\python.exe scripts\eval_dravidian.py --self-test`
- **EXPECT:** Self-test passes cleanly, verifying `script_of`, tag swapping, document boundaries, window starts, as well as new `repetition_rate` (char n-gram duplicate rate) and `distinct_n` (word bigram/trigram uniqueness ratio) logic without regressions.
- **EVIDENCE:**
```
[self-test] eval_dravidian ok
```
- **STATUS:** PASS

---

### Gate L3.2 — `scripts/report_stats.py` Self-Test (Schema Regression Guard)

- **CHECK:** `python scripts\report_stats.py --self-test`
- **EXPECT:** Schema and chart regression check passes cleanly, confirming that adding `repetition_rate`, `distinct_2`, and `distinct_3` to eval JSON outputs does not break history merging or stats plotting.
- **EVIDENCE:**
```
[self-test] report_stats ok
```
- **STATUS:** PASS

---

### Gate L3.3 — Baseline Evaluation on `step_8082_final.pt` (RTX 5050 GPU)

- **CHECK:** `.venv\Scripts\python.exe scripts\eval_dravidian.py --bpb --checkpoints checkpoints/parentheses-0.9-41M/step_8082_final.pt --out out/bpb_eval_step_8082_baseline.json`
- **EXPECT:** All 9 languages (kan, tam, tel, mal, tcy, kff, kxv, nit, eng) evaluated on 1.0 MB held-out val text; records exact bits/byte, script tag accuracy, tag-swap gap (nats/token), distinct-2, distinct-3, and repetition rate; run completes within 10 minutes on GPU.
- **EVIDENCE:**
```
[bpb] checkpoints/parentheses-0.9-41M/step_8082_final.pt: kan 0.522 tam 0.492 tel 0.623 mal 0.444 tcy 0.652 kff 0.525 kxv 0.453 nit 0.718 eng 1.653

bits per UTF-8 byte (same val text for every model; ~1.0 MB/lang)
checkpoint                                            params    kan    tam    tel    mal    tcy    kff    kxv    nit    eng
checkpoints/parentheses-0.9-41M/step_8082_final.pt     40.9M  0.522  0.492  0.623  0.444  0.652  0.525  0.453  0.718  1.653

tag accuracy (script of a sample from the tag alone) / tag-swap gap (nats/token)
checkpoints/parentheses-0.9-41M/step_8082_final.pt   kan:0.97/+0.18 tam:1.00/+0.11 tel:0.97/+0.20 mal:0.97/+0.11 tcy:1.00/+0.13 kff:0.91/+0.14 kxv:1.00/+0.16 nit:0.97/+0.20 eng:nan/+nan

diversity on generated samples: distinct-2 / distinct-3 / repetition rate
checkpoints/parentheses-0.9-41M/step_8082_final.pt   kan:0.64/0.77/rep:0.07 tam:0.60/0.76/rep:0.09 tel:0.64/0.79/rep:0.05 mal:0.60/0.74/rep:0.09 tcy:0.59/0.73/rep:0.10 kff:0.69/0.88/rep:0.01 kxv:0.65/0.84/rep:0.03 nit:0.68/0.85/rep:0.04 eng:0.79/0.87/rep:0.10
[info] wrote out/bpb_eval_step_8082_baseline.json
```
- **STATUS:** PASS

---

### Baseline Metric Summary for Parentheses 0.9-41M (`step_8082_final.pt`)

| Language | bits/byte (bpb) | Tag Acc | Tag-Swap Gap (nats/tok) | Repetition Rate | Distinct-2 | Distinct-3 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kannada (`kan`)** | 0.522 | 0.97 | +0.18 | 0.0746 | 0.6437 | 0.7695 |
| **Tamil (`tam`)** | 0.492 | 1.00 | +0.11 | 0.0853 | 0.6047 | 0.7590 |
| **Telugu (`tel`)** | 0.623 | 0.97 | +0.20 | 0.0545 | 0.6443 | 0.7885 |
| **Malayalam (`mal`)** | 0.444 | 0.97 | +0.11 | 0.0948 | 0.6016 | 0.7390 |
| **Tulu (`tcy`)** | 0.652 | 1.00 | +0.13 | 0.1045 | 0.5881 | 0.7349 |
| **Koya (`kff`)** | 0.525 | 0.91 | +0.14 | 0.0119 | 0.6929 | 0.8775 |
| **Kuvi (`kxv`)** | 0.453 | 1.00 | +0.16 | 0.0283 | 0.6453 | 0.8446 |
| **Kolami (`nit`)** | 0.718 | 0.97 | +0.20 | 0.0378 | 0.6754 | 0.8471 |
| **English (`eng`)** | 1.653 | N/A | N/A | 0.1002 | 0.7920 | 0.8738 |

Baseline JSON stored at `out/bpb_eval_step_8082_baseline.json`.
