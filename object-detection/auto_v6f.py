"""Auto-monitor v6f training. Poll every 5 min, export ONNX when mAP50 >= 0.80."""
import subprocess, os, time, sys

GCLOUD_PY = r"C:\Users\AD10209\AppData\Local\Google\Cloud SDK\google-cloud-sdk\lib\gcloud.py"
PYTHON = r"C:\Users\AD10209\AppData\Local\Python\pythoncore-3.14-64\python.exe"
os.environ["CLOUDSDK_PYTHON"] = PYTHON
TARGET_MAP50 = 0.80
POLL_INTERVAL = 300  # 5 minutes

def ssh(cmd):
    try:
        r = subprocess.run(
            [PYTHON, GCLOUD_PY, "compute", "ssh", "obj-detect-train",
             "--zone=europe-west4-a", "--project=ai-nm26osl-1724",
             f"--command={cmd}"],
            capture_output=True, timeout=60
        )
        return r.stdout.decode("utf-8", errors="replace").strip()
    except Exception as e:
        return f"ERROR: {e}"

def get_latest_map50():
    out = ssh("grep 'all ' ~/train_v6f_v4f.log 2>/dev/null | tail -1")
    # Format: all 248 22731 P R mAP50 mAP50-95
    parts = out.strip().split()
    if len(parts) >= 6:
        try:
            return float(parts[5])  # mAP50 column
        except:
            pass
    return None

def get_epoch():
    out = ssh("grep -c 'all ' ~/train_v6f_v4f.log 2>/dev/null")
    try:
        return int(out.strip())
    except:
        return 0

def export_onnx():
    print("EXPORTING v6f ONNX...")
    script = """
cd ~/object-detection
python3 -c "
from ultralytics import YOLO
m = YOLO('runs/detect/train_v6f/weights/best.pt')
m.export(format='onnx', imgsz=1280, dynamic=True, opset=17, simplify=True)
import os
p = 'runs/detect/train_v6f/weights/best.onnx'
print(f'ONNX_SIZE={os.path.getsize(p)}')
"
"""
    out = ssh(script)
    print(f"Export output: {out}")
    return "ONNX_SIZE=" in out

print(f"Monitoring v6f. Target mAP50 >= {TARGET_MAP50}. Polling every {POLL_INTERVAL}s.")
print("=" * 60)

while True:
    ts = time.strftime("%H:%M:%S")
    epoch = get_epoch()
    map50 = get_latest_map50()
    
    if map50 is not None:
        print(f"[{ts}] Epoch ~{epoch}: mAP50 = {map50:.4f}")
        
        if map50 >= TARGET_MAP50:
            print(f"[{ts}] TARGET REACHED! mAP50={map50:.4f} >= {TARGET_MAP50}")
            if export_onnx():
                print(f"[{ts}] ONNX EXPORT SUCCESS!")
                # Download
                print(f"[{ts}] Downloading v6f ONNX...")
                dl = subprocess.run(
                    [PYTHON, GCLOUD_PY, "compute", "scp",
                     "obj-detect-train:/home/AD10209/object-detection/runs/detect/train_v6f/weights/best.onnx",
                     r"c:\ainm\object-detection\best_v6f.onnx",
                     "--zone=europe-west4-a", "--project=ai-nm26osl-1724"],
                    capture_output=True, timeout=180
                )
                if os.path.exists(r"c:\ainm\object-detection\best_v6f.onnx"):
                    sz = os.path.getsize(r"c:\ainm\object-detection\best_v6f.onnx")
                    print(f"[{ts}] DOWNLOAD COMPLETE: {sz/1e6:.1f} MB")
                    
                    # Package submission
                    import zipfile
                    base = r"c:\ainm\object-detection"
                    zippath = os.path.join(base, "submission_v15_v6f.zip")
                    with zipfile.ZipFile(zippath, 'w', zipfile.ZIP_DEFLATED) as zf:
                        zf.write(os.path.join(base, "best_v3f.onnx"), "best.onnx")  # v3f
                        zf.write(os.path.join(base, "best_v4.onnx"), "best_v4.onnx")  # v4 original
                        zf.write(os.path.join(base, "best_v6f.onnx"), "best_v6.onnx")  # v6f (NEW!)
                        zf.write(os.path.join(base, "run_v10b.py"), "run.py")  # exact v10b config
                    
                    zipsz = os.path.getsize(zippath)
                    print(f"[{ts}] PACKAGED: {zippath} ({zipsz/1e6:.1f} MB)")
                    print(f"[{ts}] ALL DONE! Ready to submit.")
                else:
                    print(f"[{ts}] DOWNLOAD FAILED - file not found locally")
            else:
                print(f"[{ts}] EXPORT FAILED")
            break
    else:
        print(f"[{ts}] Epoch ~{epoch}: waiting for val results...")
    
    # Check if GPU is free (training crashed/finished)
    gpu = ssh("nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv 2>/dev/null")
    if "python3" not in gpu and epoch > 5:
        print(f"[{ts}] GPU FREE - training may have stopped! Last mAP50={map50}")
        break
    
    time.sleep(POLL_INTERVAL)
