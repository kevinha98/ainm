import subprocess, os, sys
PYTHON = r"C:\Users\AD10209\AppData\Local\Python\pythoncore-3.14-64\python.exe"
GCLOUD = r"C:\Users\AD10209\AppData\Local\Google\Cloud SDK\google-cloud-sdk\lib\gcloud.py"
os.environ["CLOUDSDK_PYTHON"] = PYTHON
r = subprocess.run(
    [PYTHON, GCLOUD, "compute", "ssh", "obj-detect-train",
     "--zone=europe-west4-a", "--project=ai-nm26osl-1724",
     "--command=grep ' all ' /home/AD10209/object-detection/train_v3f.log | tail -5; echo EPOCHS; grep ' all ' /home/AD10209/object-detection/train_v3f.log | wc -l"],
    capture_output=True, text=True, timeout=30
)
with open(r"c:\ainm\object-detection\vm_status.txt", "w") as f:
    f.write(f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}\n")
