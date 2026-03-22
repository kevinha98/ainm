import os
p = r"c:\ainm\object-detection\best_v3f.onnx"
out = r"c:\ainm\object-detection\onnx_check.txt"
if os.path.exists(p):
    msg = f"EXISTS: {os.path.getsize(p)/1024/1024:.1f} MB"
else:
    msg = "NOT_FOUND"
with open(out, "w") as f:
    f.write(msg)
