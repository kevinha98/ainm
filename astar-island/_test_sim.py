"""Test a single simulate call."""
import time, sys
sys.path.insert(0, '.')
from src.api import AstarAPI
api = AstarAPI()
t0 = time.time()
print("Calling simulate...")
result = api.simulate("93c39605-???", 0, 0, 0, steps=50)
print(f"Done in {time.time()-t0:.1f}s: {list(result.keys()) if isinstance(result, dict) else result}")
