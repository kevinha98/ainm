"""Re-export v6f best.pt (epoch 46, mAP50=0.815) and package v16 submission."""
import subprocess, os, zipfile

GCLOUD_PY = r"C:\Users\AD10209\AppData\Local\Google\Cloud SDK\google-cloud-sdk\lib\gcloud.py"
PYTHON = r"C:\Users\AD10209\AppData\Local\Python\pythoncore-3.14-64\python.exe"
os.environ["CLOUDSDK_PYTHON"] = PYTHON
BASE = r"c:\ainm\object-detection"

def ssh(cmd, timeout=300):
    r = subprocess.run(
        [PYTHON, GCLOUD_PY, "compute", "ssh", "obj-detect-train",
         "--zone=europe-west4-a", "--project=ai-nm26osl-1724",
         f"--command={cmd}"],
        capture_output=True, timeout=timeout
    )
    return r.stdout.decode("utf-8", errors="replace")

# Write a remote export script (avoids quoting hell)
print("Writing remote export script...")
write_script = """cat > /tmp/do_export.py << 'PYEOF'
from ultralytics import YOLO
import os
print("Loading best.pt...")
m = YOLO("runs/detect/runs/detect/train_v6f/weights/best.pt")
print("Exporting ONNX...")
m.export(format="onnx", imgsz=1280, dynamic=True, opset=17, simplify=True)
p = "runs/detect/runs/detect/train_v6f/weights/best.onnx"
if os.path.exists(p):
    print(f"ONNX_OK size={os.path.getsize(p)}")
else:
    print("ONNX_FAIL")
PYEOF
echo SCRIPT_WRITTEN"""
out = ssh(f"cd ~/object-detection && {write_script}")
print(out[:200].encode("ascii", errors="replace").decode())

print("Running export (takes ~60s)...")
out = ssh("cd ~/object-detection && python3 /tmp/do_export.py")
safe = out.encode("ascii", errors="replace").decode()
print(safe[:800])

if "ONNX_OK" in out:
    print("\nDownloading...")
    local = os.path.join(BASE, "best_v6f_e46.onnx")
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
        
        zippath = os.path.join(BASE, "submission_v16_v6f_e46.zip")
        with zipfile.ZipFile(zippath, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(os.path.join(BASE, "best_v3f.onnx"), "best.onnx")       # v3f as v3
            zf.write(os.path.join(BASE, "best_v4.onnx"), "best_v4.onnx")     # v4 original
            zf.write(local, "best_v6.onnx")                                   # v6f epoch46!
            zf.write(os.path.join(BASE, "run_v10b.py"), "run.py")
        zipsz = os.path.getsize(zippath)
        print(f"PACKAGED: {zippath} ({zipsz/1e6:.1f} MB)")
        print("*** READY TO SUBMIT v16 ***")
    else:
        print("DOWNLOAD FAILED")
        print(r.stderr.decode("utf-8", errors="replace")[:300])
else:
    print("Export output didn't contain ONNX_OK")
