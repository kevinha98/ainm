"""Test a single simulate call for R23."""
import time, sys
sys.path.insert(0, '.')
from src.api import AstarAPI
api = AstarAPI()

# Get R23 round_id
rounds = api.get_rounds()
r23 = None
for r in rounds:
    if r.get('round_number') == 23 and r['status'] == 'active':
        r23 = r
        break

if not r23:
    print("R23 not active!")
    sys.exit(1)

rid = r23['id']
print(f"R23 id={rid[:8]}")

t0 = time.time()
print(f"Calling simulate at {time.strftime('%H:%M:%S')}...")
result = api.simulate(rid, 0, 0, 0, steps=50)
elapsed = time.time() - t0
if isinstance(result, dict):
    if 'error' in result:
        print(f"Error in {elapsed:.1f}s: {result}")
    else:
        keys = list(result.keys())
        grid = result.get('grid', [])
        gshape = f"{len(grid)}x{len(grid[0]) if grid else 0}" if grid else "no grid"
        print(f"OK in {elapsed:.1f}s: keys={keys}, grid={gshape}")
else:
    print(f"Unexpected in {elapsed:.1f}s: {result}")
