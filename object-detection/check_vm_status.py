import subprocess, sys, os

GCLOUD_PY = r"C:\Users\AD10209\AppData\Local\Google\Cloud SDK\google-cloud-sdk\lib\gcloud.py"
PYTHON = r"C:\Users\AD10209\AppData\Local\Python\pythoncore-3.14-64\python.exe"
os.environ["CLOUDSDK_PYTHON"] = PYTHON

def ssh_cmd(cmd):
    result = subprocess.run(
        [PYTHON, GCLOUD_PY, "compute", "ssh", "obj-detect-train",
         "--zone=europe-west4-a", "--project=ai-nm26osl-1724",
         f"--command={cmd}"],
        capture_output=True, text=True, timeout=60
    )
    return f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

out1 = ssh_cmd("nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv")
out2 = ssh_cmd("ls -la ~/object-detection/runs/detect/train_v6f/weights/ 2>/dev/null || echo V6F_NOT_STARTED")
out3 = ssh_cmd("tail -10 ~/train_v6f_v4f.log 2>/dev/null || echo LOG_NOT_FOUND")
out4 = ssh_cmd("tail -3 ~/object-detection/runs/detect/train_v6f/results.csv 2>/dev/null || echo NO_RESULTS_YET")

combined = f"=== GPU ===\n{out1}\n=== V6F WEIGHTS ===\n{out2}\n=== LOG TAIL ===\n{out3}\n=== V6F RESULTS ===\n{out4}\n"

with open(r"c:\ainm\object-detection\vm_status.txt", "w", encoding="utf-8") as f:
    f.write(combined)
print("DONE")
