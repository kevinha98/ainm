"""Inspect saved observations."""
import json, numpy as np
d = json.load(open("data/observations_fd3c92ff.json"))
for si in ["0", "1"]:
    obs = d[si]
    print(f"Seed {si}: {len(obs)} observations")
    for o in obs[:2]:
        print(f"  keys: {list(o.keys())}")
        g = np.array(o.get("grid", []))
        row = o.get("row")
        col = o.get("col")
        vp = o.get("viewport")
        print(f"  grid shape: {g.shape}, row={row}, col={col}")
        print(f"  viewport: {vp}")
        if g.size > 0:
            print(f"  unique: {sorted(np.unique(g))}")
