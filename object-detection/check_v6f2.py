"""Check v6f progress via checkpoint files and results.csv"""
import subprocess, os

GCLOUD_PY = r"C:\Users\AD10209\AppData\Local\Google\Cloud SDK\google-cloud-sdk\lib\gcloud.py"
PYTHON = r"C:\Users\AD10209\AppData\Local\Python\pythoncore-3.14-64\python.exe"
os.environ["CLOUDSDK_PYTHON"] = PYTHON
OUT = r"c:\ainm\object-detection\v6f_status2.txt"

cmd = """
echo '===WEIGHTS==='
ls -la ~/object-detection/runs/detect/train_v6f/weights/ 2>/dev/null || echo 'NO WEIGHTS DIR'
echo '===CSV==='
wc -l ~/object-detection/runs/detect/train_v6f/results.csv 2>/dev/null || echo 'NO CSV'
echo '===CSVTAIL==='
tail -5 ~/object-detection/runs/detect/train_v6f/results.csv 2>/dev/null || echo 'NO CSV'
echo '===LOGTAIL==='  
tail -c 2000 ~/train_v6f_v4f.log 2>/dev/null || echo 'NO LOG'
echo '===GPU==='
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null
echo '===PROCS==='
ps aux | grep train | grep -v grep 2>/dev/null
echo '===LOGSIZE==='
ls -la ~/train_v6f_v4f.log 2>/dev/null
"""

r = subprocess.run(
    [PYTHON, GCLOUD_PY, "compute", "ssh", "obj-detect-train",
     "--zone=europe-west4-a", "--project=ai-nm26osl-1724",
     f"--command={cmd}"],
    capture_output=True, timeout=60
)
out = r.stdout.decode("utf-8", errors="replace")
err = r.stderr.decode("utf-8", errors="replace")
with open(OUT, "w", encoding="utf-8") as f:
    f.write("STDOUT:\n" + out + "\nSTDERR:\n" + err[:500] + "\n")
print("WROTE " + OUT)
