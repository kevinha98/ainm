"""Monitor v3f training on VM. When mAP50 >= target, export ONNX and download."""
import subprocess, time, sys, re, os

GCLOUD = r"C:\Users\AD10209\AppData\Local\Google\Cloud SDK\google-cloud-sdk\lib\gcloud.py"
PYTHON = r"C:\Users\AD10209\AppData\Local\Python\pythoncore-3.14-64\python.exe"
VM = "obj-detect-train"
ZONE = "europe-west4-a"
PROJECT = "ai-nm26osl-1724"
TARGET_MAP50 = 0.82
POLL_INTERVAL = 45  # seconds
BEST_PT = "/home/AD10209/object-detection/runs/detect/runs/detect/train_v3f/weights/best.pt"
ONNX_OUT = "/home/AD10209/object-detection/runs/detect/runs/detect/train_v3f/weights/best.onnx"

os.environ["CLOUDSDK_PYTHON"] = PYTHON

def ssh_cmd(cmd):
    r = subprocess.run(
        [PYTHON, GCLOUD, "compute", "ssh", VM, f"--zone={ZONE}", f"--project={PROJECT}", f"--command={cmd}"],
        capture_output=True, text=True, timeout=30
    )
    return r.stdout + r.stderr

def get_status():
    out = ssh_cmd("grep ' all ' /home/AD10209/object-detection/train_v3f.log | tail -5")
    lines = [l.strip() for l in out.split('\n') if 'all' in l and '248' in l]
    if not lines:
        return 0, 0.0, 0.0
    # Parse last line: all  248  22731  P  R  mAP50  mAP50-95
    parts = lines[-1].split()
    # Find numeric values
    nums = [float(x) for x in parts if re.match(r'^[\d.]+$', x)]
    # nums: [248, 22731, P, R, mAP50, mAP50-95]
    epochs = len(lines)
    # Get total epoch count
    out2 = ssh_cmd("grep ' all ' /home/AD10209/object-detection/train_v3f.log | wc -l")
    for l in out2.split('\n'):
        l = l.strip()
        if l.isdigit():
            epochs = int(l)
            break
    if len(nums) >= 6:
        return epochs, nums[4], nums[5]  # mAP50, mAP50-95
    return epochs, 0.0, 0.0

def export_onnx():
    print("\n>>> EXPORTING ONNX on VM...")
    out = ssh_cmd(f"cd /home/AD10209 && python3 export_v3f.py")
    print(out)
    return "DONE" in out or "best.onnx" in out.lower()

def download_onnx():
    print("\n>>> DOWNLOADING ONNX from VM...")
    local_path = r"c:\ainm\object-detection\best_v3f.onnx"
    r = subprocess.run(
        [PYTHON, GCLOUD, "compute", "scp",
         f"{VM}:{ONNX_OUT}", local_path,
         f"--zone={ZONE}", f"--project={PROJECT}"],
        capture_output=True, text=True, timeout=120
    )
    print(r.stdout + r.stderr)
    if os.path.exists(local_path):
        sz = os.path.getsize(local_path) / 1024 / 1024
        print(f"  Downloaded: {sz:.1f} MB")
        return True
    return False

def package_submission():
    print("\n>>> PACKAGING SUBMISSION...")
    import zipfile
    v3f_path = r"c:\ainm\object-detection\best_v3f.onnx"
    v10b_run = r"c:\ainm\object-detection\run_v10b.py"
    v4_path = r"c:\ainm\object-detection\best_v4.onnx"
    v6_path = r"c:\ainm\object-detection\best_v6.onnx"
    out_path = r"c:\ainm\object-detection\submission_v14_v3f.zip"
    
    zf = zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=1)
    # v3f replaces v3 as best.onnx
    zf.write(v3f_path, "best.onnx")
    print(f"  best.onnx (v3f): {os.path.getsize(v3f_path)/1024/1024:.1f} MB")
    zf.write(v10b_run, "run.py")
    print(f"  run.py (v10b baseline)")
    zf.write(v4_path, "best_v4.onnx")
    print(f"  best_v4.onnx: {os.path.getsize(v4_path)/1024/1024:.1f} MB")
    zf.write(v6_path, "best_v6.onnx")
    print(f"  best_v6.onnx: {os.path.getsize(v6_path)/1024/1024:.1f} MB")
    zf.close()
    
    total = os.path.getsize(out_path) / 1024 / 1024
    print(f"\n  SUBMISSION READY: {out_path}")
    print(f"  Total size: {total:.1f} MB")
    
    # Copy to submissions folder
    import shutil
    shutil.copy2(out_path, r"c:\ainm\object-detection\submissions\submission_v14_v3f.zip")
    print(f"  Copied to submissions/")
    return True

if __name__ == "__main__":
    # Log to both console and file
    LOG = open(r"c:\ainm\object-detection\monitor.log", "w")
    def log(msg):
        print(msg)
        LOG.write(msg + "\n")
        LOG.flush()
        sys.stdout.flush()
    
    log(f"Monitoring v3f training. Target: mAP50 >= {TARGET_MAP50}")
    log(f"Polling every {POLL_INTERVAL}s...")
    log("=" * 60)
    
    best_map = 0.0
    while True:
        try:
            epochs, map50, map5095 = get_status()
            if map50 > best_map:
                best_map = map50
            ts = time.strftime("%H:%M:%S")
            log(f"  [{ts}] Epoch {epochs:3d} | mAP50={map50:.3f} | best={best_map:.3f} | target={TARGET_MAP50}")
            
            if best_map >= TARGET_MAP50:
                log(f"\n{'='*60}")
                log(f"  TARGET REACHED! best mAP50 = {best_map:.3f} >= {TARGET_MAP50}")
                log(f"{'='*60}")
                
                # Export
                if export_onnx():
                    log("  ONNX export SUCCESS")
                else:
                    log("  ONNX export may have issues, trying download anyway...")
                
                # Download
                if download_onnx():
                    log("  Download SUCCESS")
                else:
                    log("  Download FAILED!")
                    sys.exit(1)
                
                # Package
                package_submission()
                log("\n" + "=" * 60)
                log("  ALL DONE! Submit submission_v14_v3f.zip")
                log("=" * 60)
                LOG.close()
                break
            
        except Exception as e:
            log(f"  [{time.strftime('%H:%M:%S')}] Error: {e}")
        
        time.sleep(POLL_INTERVAL)
