"""Check training status and write results to /tmp/status.txt"""
import os
log = "/home/AD10209/object-detection/training.log"
out = "/tmp/status.txt"
lines = []
try:
    with open(log) as f:
        all_lines = f.readlines()
    lines.append(f"LOG LINES: {len(all_lines)}")
    lines.append(f"LOG SIZE: {os.path.getsize(log)} bytes")
    lines.append("=== HEAD ===")
    lines.extend(l.rstrip() for l in all_lines[:15])
    lines.append("=== TAIL ===")
    lines.extend(l.rstrip() for l in all_lines[-20:])
except Exception as e:
    lines.append(f"ERROR reading log: {e}")

import subprocess
r = subprocess.run(["ps", "aux"], capture_output=True, text=True)
py_procs = [l for l in r.stdout.split("\n") if "python" in l.lower() and "grep" not in l and "check_status" not in l]
lines.append("=== PROCESSES ===")
lines.extend(py_procs if py_procs else ["No python training processes found"])

r2 = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
for l in r2.stdout.split("\n"):
    if "MiB" in l:
        lines.append(f"GPU: {l.strip()}")

with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print("Written to", out)
