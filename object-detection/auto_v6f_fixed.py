"""Auto-monitor v6f using results.csv (correct path). Export + download when ready."""
import subprocess, os, time, csv, io

GCLOUD_PY = r"C:\Users\AD10209\AppData\Local\Google\Cloud SDK\google-cloud-sdk\lib\gcloud.py"
PYTHON = r"C:\Users\AD10209\AppData\Local\Python\pythoncore-3.14-64\python.exe"
os.environ["CLOUDSDK_PYTHON"] = PYTHON
BASE = r"c:\ainm\object-detection"
TARGET_MAP50 = 0.75
POLL_INTERVAL = 180  # 3 minutes
CSV_PATH = "~/object-detection/runs/detect/runs/detect/train_v6f/results.csv"
BEST_PT = "~/object-detection/runs/detect/runs/detect/train_v6f/weights/best.pt"

def ssh(cmd):
    try:
        r = subprocess.run(
            [PYTHON, GCLOUD_PY, "compute", "ssh", "obj-detect-train",
             "--zone=europe-west4-a", "--project=ai-nm26osl-1724",
             f"--command={cmd}"],
            capture_output=True, timeout=90
        )
        return r.stdout.decode("utf-8", errors="replace").strip()
    except Exception as e:
        return f"ERROR: {e}"

def get_progress():
    out = ssh(f"tail -1 {CSV_PATH} 2>/dev/null")
    if not out or "ERROR" in out:
        return None, None
    parts = out.strip().split(",")
    if len(parts) >= 8:
        try:
            epoch = int(parts[0].strip())
            map50 = float(parts[7].strip())  # metrics/mAP50(B) is column index 7
            return epoch, map50
        except:
            pass
    return None, None

def export_onnx():
    print("EXPORTING v6f ONNX...")
    script = f"""cd ~/object-detection && python3 -c "
from ultralytics import YOLO
import os
m = YOLO('runs/detect/runs/detect/train_v6f/weights/best.pt')
m.export(format='onnx', imgsz=1280, dynamic=True, opset=17, simplify=True)
p = 'runs/detect/runs/detect/train_v6f/weights/best.onnx'
if os.path.exists(p):
    print(f'ONNX_OK size={{os.path.getsize(p)}}')
else:
    print('ONNX_FAIL')
"
"""
    out = ssh(script)
    print(f"Export: {out}")
    return "ONNX_OK" in out

def download_onnx():
    local = os.path.join(BASE, "best_v6f.onnx")
    r = subprocess.run(
        [PYTHON, GCLOUD_PY, "compute", "scp",
         "obj-detect-train:/home/AD10209/object-detection/runs/detect/runs/detect/train_v6f/weights/best.onnx",
         local,
         "--zone=europe-west4-a", "--project=ai-nm26osl-1724"],
        capture_output=True, timeout=300
    )
    if os.path.exists(local):
        return os.path.getsize(local)
    return 0

def package():
    import zipfile
    zippath = os.path.join(BASE, "submission_v15_v6f.zip")
    with zipfile.ZipFile(zippath, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(os.path.join(BASE, "best_v3f.onnx"), "best.onnx")     # v3f as primary
        zf.write(os.path.join(BASE, "best_v4.onnx"), "best_v4.onnx")   # v4 original
        zf.write(os.path.join(BASE, "best_v6f.onnx"), "best_v6.onnx")  # v6f NEW
        zf.write(os.path.join(BASE, "run_v10b.py"), "run.py")          # golden config
    return os.path.getsize(zippath)

print(f"V6f monitor. Target: mAP50 >= {TARGET_MAP50}. Poll every {POLL_INTERVAL}s")
print("=" * 60)

while True:
    ts = time.strftime("%H:%M:%S")
    epoch, map50 = get_progress()
    
    if epoch is not None:
        status = f"[{ts}] Epoch {epoch}: mAP50 = {map50:.4f}"
        print(status)
        
        # Write status file for external reading
        with open(os.path.join(BASE, "v6f_live.txt"), "w") as f:
            f.write(f"{status}\n")
        
        if map50 >= TARGET_MAP50:
            print(f"\n{'='*60}")
            print(f"TARGET REACHED! mAP50 = {map50:.4f} >= {TARGET_MAP50}")
            
            if export_onnx():
                print("ONNX exported successfully!")
                sz = download_onnx()
                if sz > 0:
                    print(f"Downloaded: {sz/1e6:.1f} MB")
                    zipsz = package()
                    print(f"PACKAGED: submission_v15_v6f.zip ({zipsz/1e6:.1f} MB)")
                    print("\n*** READY TO SUBMIT! ***")
                else:
                    print("DOWNLOAD FAILED!")
            else:
                print("EXPORT FAILED!")
            break
    else:
        print(f"[{ts}] No data yet...")
    
    # Check GPU
    gpu = ssh("nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null")
    if not gpu.strip() and epoch and epoch > 10:
        print(f"[{ts}] GPU FREE! Training may have finished/crashed at epoch {epoch}")
        if epoch and map50 and map50 > 0.65:
            print(f"Attempting export with mAP50={map50:.4f}...")
            if export_onnx():
                sz = download_onnx()
                if sz > 0:
                    zipsz = package()
                    print(f"PACKAGED (early): submission_v15_v6f.zip ({zipsz/1e6:.1f} MB)")
        break
    
    time.sleep(POLL_INTERVAL)
