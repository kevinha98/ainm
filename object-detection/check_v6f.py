import subprocess, os, time

GCLOUD_PY = r"C:\Users\AD10209\AppData\Local\Google\Cloud SDK\google-cloud-sdk\lib\gcloud.py"
PYTHON = r"C:\Users\AD10209\AppData\Local\Python\pythoncore-3.14-64\python.exe"
os.environ["CLOUDSDK_PYTHON"] = PYTHON
OUT = r"c:\ainm\object-detection\v6f_status.txt"

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
        return f"ERROR: {e}"

out = ssh("echo LOGSIZE; wc -l ~/train_v6f_v4f.log 2>/dev/null; echo EPOCHS; grep -c 'Epoch ' ~/train_v6f_v4f.log 2>/dev/null; echo VALRESULTS; grep 'all ' ~/train_v6f_v4f.log 2>/dev/null | tail -5; echo GPUSPLIT; nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv 2>/dev/null; echo LASTLINES; tail -5 ~/train_v6f_v4f.log 2>/dev/null")

with open(OUT, "w", encoding="utf-8") as f:
    f.write(out + "\n")
print("WROTE " + OUT)
