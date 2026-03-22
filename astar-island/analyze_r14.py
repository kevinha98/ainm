"""Fetch R14 GT + leaderboard + analyze the 47.6 disaster."""
import sys, json, time, numpy as np
sys.path.insert(0, "src")
from api import AstarAPI
from pathlib import Path

api = AstarAPI()
rounds = api.get_rounds()
DATA = Path("data")

# Fetch R14 GT
r14 = [r for r in rounds if r.get("round_number") == 14][0]
r14id = r14["id"]
gt_path = DATA / f"ground_truth_{r14id[:8]}.json"
if not gt_path.exists():
    print("Fetching R14 GT...")
    full = api._get(f"/rounds/{r14id}")
    n_seeds = len(full.get("initial_states", []))
    gt = {}
    for si in range(n_seeds):
        for attempt in range(3):
            try:
                a = api.get_analysis(r14id, si)
                ig = full["initial_states"][si]["grid"]
                g = a.get("ground_truth", a.get("probabilities"))
                if g is not None:
                    gt[str(si)] = {"ground_truth": g, "initial_grid": ig}
                    print(f"  Got GT seed {si}")
                break
            except Exception as e:
                print(f"  Attempt {attempt+1} failed seed {si}: {e}")
                time.sleep(2)
        time.sleep(1)
    if gt:
        with open(gt_path, "w") as f:
            json.dump(gt, f)
        print(f"Saved {len(gt)} seeds to {gt_path}")
else:
    print(f"R14 GT already at {gt_path}")

# Fetch leaderboard
print("\n--- LEADERBOARD (top 20) ---")
try:
    lb = api.get_leaderboard()
    if isinstance(lb, list):
        for i, entry in enumerate(lb[:20]):
            name = entry.get("name", entry.get("email", "?"))
            score = entry.get("score", entry.get("total_score", "?"))
            print(f"  #{i+1}: {name} — {score}")
except Exception as e:
    print(f"Leaderboard error: {e}")

# R15 info
r15 = [r for r in rounds if r.get("round_number") == 15][0]
r15id = r15["id"]
full15 = api._get(f"/rounds/{r15id}")
n_seeds = len(full15.get("initial_states", []))
ig0 = np.array(full15["initial_states"][0]["grid"])
print(f"\nR15: {n_seeds} seeds, grid {ig0.shape}")
print(f"R15 id: {r15id}")

# Count initial classes in R15
from models import build_class_grid
from settings import CLASS_NAMES
cls = build_class_grid(ig0)
for c in range(6):
    n = (cls == c).sum()
    if n > 0:
        print(f"  {CLASS_NAMES[c]}: {n} cells ({100*n/cls.size:.1f}%)")
