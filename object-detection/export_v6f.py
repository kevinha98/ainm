"""Export v6f ONNX, download, and package submission."""
import subprocess, os, zipfile

GCLOUD_PY = r"C:\Users\AD10209\AppData\Local\Google\Cloud SDK\google-cloud-sdk\lib\gcloud.py"
PYTHON = r"C:\Users\AD10209\AppData\Local\Python\pythoncore-3.14-64\python.exe"
os.environ["CLOUDSDK_PYTHON"] = PYTHON
BASE = r"c:\ainm\object-detection"

def ssh(cmd, timeout=120):
    try:
        r = subprocess.run(
            [PYTHON, GCLOUD_PY, "compute", "ssh", "obj-detect-train",
             "--zone=europe-west4-a", "--project=ai-nm26osl-1724",
             f"--command={cmd}"],
            capture_output=True, timeout=timeout
        )
        return r.stdout.decode("utf-8", errors="replace") + "\nSTDERR:" + r.stderr.decode("utf-8", errors="replace")[:300]
    except Exception as e:
        return f"ERROR: {e}"

# Step 1: Export ONNX
print("Step 1: Exporting ONNX...")
export_cmd = """cd ~/object-detection && python3 -c "
from ultralytics import YOLO
import os
m = YOLO('runs/detect/runs/detect/train_v6f/weights/best.pt')
m.export(format='onnx', imgsz=1280, dynamic=True, opset=17, simplify=True)
p = 'runs/detect/runs/detect/train_v6f/weights/best.onnx'
if os.path.exists(p):
    print('ONNX_OK size=' + str(os.path.getsize(p)))
else:
    print('ONNX_FAIL')
"
"""
out = ssh(export_cmd, timeout=300)
safe = out[:500].encode("ascii", errors="replace").decode("ascii")
print(safe)

if "ONNX_OK" in out:
    print("\nStep 2: Downloading ONNX...")
    local = os.path.join(BASE, "best_v6f.onnx")
    r = subprocess.run(
        [PYTHON, GCLOUD_PY, "compute", "scp",
         "obj-detect-train:/home/AD10209/object-detection/runs/detect/runs/detect/train_v6f/weights/best.onnx",
         local,
         "--zone=europe-west4-a", "--project=ai-nm26osl-1724"],
        capture_output=True, timeout=300
    )
    if os.path.exists(local):
        sz = os.path.getsize(local)
        print(f"Downloaded: {sz/1e6:.1f} MB")
        
        print("\nStep 3: Packaging submission...")
        zippath = os.path.join(BASE, "submission_v15_v6f.zip")
        with zipfile.ZipFile(zippath, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(os.path.join(BASE, "best_v3f.onnx"), "best.onnx")
            zf.write(os.path.join(BASE, "best_v4.onnx"), "best_v4.onnx")
            zf.write(os.path.join(BASE, "best_v6f.onnx"), "best_v6.onnx")
            zf.write(os.path.join(BASE, "run_v10b.py"), "run.py")
        zipsz = os.path.getsize(zippath)
        print(f"PACKAGED: {zippath} ({zipsz/1e6:.1f} MB)")
        print("\n*** READY TO SUBMIT ***")
    else:
        print("DOWNLOAD FAILED!")
        print(r.stderr.decode("utf-8", errors="replace")[:300])
elif "ONNX_FAIL" in out:
    print("EXPORT FAILED - no ONNX file created")
else:
    # Maybe ONNX already exists
    print("Checking if ONNX already exists...")
    check = ssh("ls -la ~/object-detection/runs/detect/runs/detect/train_v6f/weights/best.onnx 2>/dev/null")
    print(check[:300])
