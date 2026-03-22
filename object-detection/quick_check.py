import subprocess, sys, os, time

GCLOUD_PY = r"C:\Users\AD10209\AppData\Local\Google\Cloud SDK\google-cloud-sdk\lib\gcloud.py"
PYTHON = r"C:\Users\AD10209\AppData\Local\Python\pythoncore-3.14-64\python.exe"
os.environ["CLOUDSDK_PYTHON"] = PYTHON
OUT = r"c:\ainm\object-detection\vm_check.txt"

# Delete old file
if os.path.exists(OUT):
    os.remove(OUT)

cmd = "nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv && echo DELIM1 && ls -la ~/object-detection/runs/detect/train_v6f/weights/ 2>/dev/null && echo DELIM2 && tail -5 ~/object-detection/runs/detect/train_v6f/results.csv 2>/dev/null && echo DELIM3 && tail -15 ~/train_v6f_v4f.log 2>/dev/null && echo DONE_OK"

result = subprocess.run(
    [PYTHON, GCLOUD_PY, "compute", "ssh", "obj-detect-train",
     "--zone=europe-west4-a", "--project=ai-nm26osl-1724",
     f"--command={cmd}"],
    capture_output=True, text=True, timeout=60
)

with open(OUT, "w") as f:
    f.write(f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\nRETCODE:{result.returncode}\n")

print("WROTE " + OUT)
