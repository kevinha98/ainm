import subprocess, os, time, sys

GCLOUD_PY = r"C:\Users\AD10209\AppData\Local\Google\Cloud SDK\google-cloud-sdk\lib\gcloud.py"
PYTHON = r"C:\Users\AD10209\AppData\Local\Python\pythoncore-3.14-64\python.exe"
os.environ["CLOUDSDK_PYTHON"] = PYTHON
OUT = r"c:\ainm\object-detection\monitor_out.txt"

def ssh(cmd):
    try:
        r = subprocess.run(
            [PYTHON, GCLOUD_PY, "compute", "ssh", "obj-detect-train",
             "--zone=europe-west4-a", "--project=ai-nm26osl-1724",
             f"--command={cmd}"],
            capture_output=True, timeout=45
        )
        return r.stdout.decode("utf-8", errors="replace").strip()
    except Exception as e:
        return f"SSH_ERROR: {e}"

poll = 0
while True:
    poll += 1
    ts = time.strftime("%H:%M:%S")
    
    # Single SSH with semicolons (not &&) to avoid chain breaks 
    cmd = "tail -3 ~/object-detection/runs/detect/train_v6f/results.csv 2>/dev/null; echo SPLIT; nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv 2>/dev/null; echo SPLIT2; tail -5 ~/train_v6f_v4f.log 2>/dev/null"
    out = ssh(cmd)
    
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(f"\n[{ts}] Poll {poll}\n{out}\n{'='*60}\n")
    
    # Parse mAP50 from results.csv if available
    lines = out.split("\n")
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) >= 8:
            try:
                epoch = int(parts[0].strip())
                map50 = float(parts[7].strip())
                print(f"[{ts}] Epoch {epoch}: mAP50={map50:.4f}")
            except:
                pass
    
    # Check if GPU is still being used
    if "python3" not in out and poll > 5:
        print(f"[{ts}] GPU free — training may have finished!")
        # Do final check
        final = ssh("ls -la ~/object-detection/runs/detect/train_v6f/weights/best.onnx 2>/dev/null; echo SPLIT; ls -la ~/object-detection/runs/detect/train_v4f/weights/best.onnx 2>/dev/null")
        with open(OUT, "a") as f:
            f.write(f"\n[{ts}] FINAL CHECK:\n{final}\n")
        print(f"FINAL: {final}")
        break
    
    time.sleep(60)
