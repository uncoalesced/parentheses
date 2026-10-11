import os, sys, time, urllib.request

REPO = "nicoboss/Qwen3-14B-Uncensored"
BASE_URL = f"https://huggingface.co/{REPO}/resolve/main"
TARGET_DIR = "F:/Models/Qwen3-14B-Uncensored"

SHARDS = [
    "model-00001-of-00006.safetensors",
    "model-00004-of-00006.safetensors",
    "model-00005-of-00006.safetensors",
    "model-00006-of-00006.safetensors",
]

CHUNK_SIZE = 16 * 1024 * 1024  # 16MB sequential write buffer

def download_file(filename):
    final_path = os.path.join(TARGET_DIR, filename)
    part_path = final_path + ".part"
    url = f"{BASE_URL}/{filename}"
    
    # Check if already complete
    req_head = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req_head, timeout=30) as resp:
            total_size = int(resp.headers.get("Content-Length", 0))
    except Exception as e:
        print(f"[{filename}] HEAD request failed: {e}", flush=True)
        return False

    if os.path.exists(final_path):
        cur_size = os.path.getsize(final_path)
        if cur_size == total_size and total_size > 0:
            print(f"[{filename}] Already completely downloaded ({cur_size / (1024**3):.3f} GB). Skipping.", flush=True)
            return True
        else:
            print(f"[{filename}] Existing file size mismatch ({cur_size} vs {total_size}). Redownloading.", flush=True)

    # Resume capability using Range header
    downloaded = 0
    if os.path.exists(part_path):
        downloaded = os.path.getsize(part_path)
        if downloaded > total_size:
            downloaded = 0
            open(part_path, 'wb').close()

    headers = {"User-Agent": "Mozilla/5.0"}
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"
        print(f"[{filename}] Resuming from {downloaded / (1024**3):.3f} GB / {total_size / (1024**3):.3f} GB...", flush=True)
    else:
        print(f"[{filename}] Starting download ({total_size / (1024**3):.3f} GB)...", flush=True)

    req = urllib.request.Request(url, headers=headers)
    start_time = time.time()
    last_log = start_time
    bytes_since_log = 0

    mode = "ab" if downloaded > 0 else "wb"
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(part_path, mode) as out_f:
            while True:
                chunk = resp.read(CHUNK_SIZE)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)
                bytes_since_log += len(chunk)

                now = time.time()
                if now - last_log >= 10.0:  # Log every 10 seconds
                    speed_mb = (bytes_since_log / (now - last_log)) / (1024 * 1024)
                    pct = (downloaded / total_size * 100) if total_size > 0 else 0
                    rem_bytes = max(0, total_size - downloaded)
                    eta_sec = (rem_bytes / (speed_mb * 1024 * 1024)) if speed_mb > 0 else 0
                    print(f"[{filename}] {pct:5.1f}% | {downloaded / (1024**3):.2f}/{total_size / (1024**3):.2f} GB | {speed_mb:5.1f} MB/s | ETA: {eta_sec/60:.1f} min", flush=True)
                    last_log = now
                    bytes_since_log = 0

        # Verify size
        if os.path.getsize(part_path) == total_size:
            if os.path.exists(final_path):
                os.remove(final_path)
            os.rename(part_path, final_path)
            total_time = time.time() - start_time
            print(f"[{filename}] Finished successfully! ({total_size / (1024**3):.3f} GB in {total_time/60:.1f} min)", flush=True)
            return True
        else:
            print(f"[{filename}] Error: Downloaded size {os.path.getsize(part_path)} does not match {total_size}", flush=True)
            return False

    except Exception as e:
        print(f"[{filename}] Download error: {e}", flush=True)
        return False

def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"=== Starting Qwen3-14B-Uncensored Shard Downloader ===")
    print(f"Target: {TARGET_DIR}")
    print(f"Shards to download: {len(SHARDS)}")
    
    for shard in SHARDS:
        success = False
        retries = 3
        while not success and retries > 0:
            success = download_file(shard)
            if not success:
                retries -= 1
                if retries > 0:
                    print(f"[{shard}] Retrying in 5 seconds... ({retries} retries left)", flush=True)
                    time.sleep(5)
        if not success:
            print(f"[{shard}] FAILED after all retries. Stopping.", flush=True)
            sys.exit(1)

    print("=== All Qwen3-14B-Uncensored Shards Downloaded Successfully! ===", flush=True)

if __name__ == "__main__":
    main()
