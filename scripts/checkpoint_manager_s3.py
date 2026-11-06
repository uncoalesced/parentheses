"""
Parallel Checkpoint Archival & Headroom Governance Engine for Parentheses 1.0-240M
Monitors working checkpoints on NVMe E:, computes SHA256 hashes, archives to cold storage I:,
verifies bitwise parity, updates MANIFEST_s3_sha256.tsv, and prunes older intermediate
checkpoints from E: to safeguard disk headroom.
"""

import argparse
import hashlib
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def compute_sha256(filepath):
    """Computes SHA256 hash in 8MB chunks to minimize memory footprint."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest().lower()


def is_file_stable(filepath, min_size_mb=2000, wait_sec=3):
    """Ensures file is completely written by checking minimum size and stability."""
    if not os.path.exists(filepath):
        return False
    size1 = os.path.getsize(filepath)
    if size1 < min_size_mb * 1024 * 1024:
        return False
    time.sleep(wait_sec)
    size2 = os.path.getsize(filepath)
    return size1 == size2


def main():
    parser = argparse.ArgumentParser(description="Parallel Checkpoint Archiver & Headroom Manager")
    parser.add_argument("--source-dir", default=str(REPO_ROOT / "checkpoints" / "parentheses-1.0-240M-s3"))
    parser.add_argument("--archive-dir", default=r"I:\parentheses\checkpoints\parentheses-1.0-240M-s3")
    parser.add_argument("--keep-latest-n", type=int, default=2)
    parser.add_argument("--interval", type=int, default=20)
    parser.add_argument("--manifest-name", default="MANIFEST_s3_sha256.tsv",
                        help="manifest file inside --archive-dir (S4: MANIFEST_s4_sha256.tsv)")
    parser.add_argument("--milestones", default="5000,6000",
                        help="comma-separated steps whose checkpoints are never pruned from E: (S4: 7000,8000)")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    archive_dir = Path(args.archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = archive_dir / args.manifest_name
    milestones = {f"step_{m.strip()}" for m in args.milestones.split(",") if m.strip()}

    if not manifest_path.exists():
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write("sha256\tfilename\tstep\tfile_size_bytes\ttimestamp\n")

    print(f"[*] Starting Parallel Checkpoint Archival & Headroom Governance Engine")
    print(f"    - Source (NVMe E:) : {source_dir}")
    print(f"    - Archive (Cold I:): {archive_dir}")
    print(f"    - Retention on E:  : Latest {args.keep_latest_n} working checkpoints + milestones")
    print(f"    - Polling Cadence  : {args.interval}s\n")

    # Load existing manifest records
    archived_files = set()
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    archived_files.add(parts[1])

    while True:
        try:
            if source_dir.exists():
                all_pts = [p for p in source_dir.glob("step_*.pt") if not p.name.endswith(".tmp")]
                
                # Sort by step number
                def step_num(p):
                    m = re.search(r"step_(\d+)", p.name)
                    return int(m.group(1)) if m else -1

                all_pts.sort(key=step_num)

                # Process unarchived checkpoints
                for cp in all_pts:
                    if cp.name not in archived_files:
                        if is_file_stable(cp):
                            dest_cp = archive_dir / cp.name
                            print(f"[archiver] New stable checkpoint detected: {cp.name} ({cp.stat().st_size / (1024**3):.2f} GB)")
                            print(f"           Copying to {dest_cp}...")
                            t0 = time.time()
                            shutil.copy2(cp, dest_cp)
                            copy_dt = time.time() - t0

                            # Bitwise SHA256 verification
                            src_hash = compute_sha256(cp)
                            dst_hash = compute_sha256(dest_cp)

                            if src_hash == dst_hash:
                                print(f"[archiver] SHA256 verified bitwise match: {src_hash[:16]}... (in {copy_dt:.1f}s)")
                                s_match = re.search(r"step_(\d+)", cp.name)
                                step_val = s_match.group(1) if s_match else "unknown"
                                now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                                with open(manifest_path, "a", encoding="utf-8") as f:
                                    f.write(f"{src_hash}\t{cp.name}\t{step_val}\t{cp.stat().st_size}\t{now_ts}\n")

                                archived_files.add(cp.name)

                                # Update latest.pt on source_dir
                                latest_pt = source_dir / "latest.pt"
                                shutil.copy2(cp, latest_pt)
                            else:
                                print(f"[!] ERROR: Hash mismatch for {cp.name}! Re-trying next cycle.", file=sys.stderr)
                                if dest_cp.exists():
                                    dest_cp.unlink()

                # Headroom Governance: Prune older intermediate checkpoints on E:
                # Keep latest N, the --milestones steps, and any *_final checkpoint
                verified_on_src = [p for p in all_pts if p.name in archived_files]
                if len(verified_on_src) > args.keep_latest_n:
                    to_prune = verified_on_src[:-args.keep_latest_n]
                    for p in to_prune:
                        # Protect milestone anchors
                        if p.stem.removesuffix("_final") in milestones or "final" in p.name:
                            continue
                        print(f"[archiver] Headroom Prune: Removing older verified checkpoint {p.name} from E: (archived on I:)")
                        p.unlink()

        except KeyboardInterrupt:
            print("\n[*] Checkpoint manager stopped by user.")
            break
        except Exception as e:
            print(f"[!] Warning: Checkpoint manager error: {e}", file=sys.stderr)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
