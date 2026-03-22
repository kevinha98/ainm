"""Fetch R14 GT seed 4 + check leaderboard + R15 status."""
import sys, json, time
sys.path.insert(0, "src")
from api import AstarAPI
from pathlib import Path

api = AstarAPI()
DATA_DIR = Path("data")

# R14 GT — try to get seed 4
r14_id = "d0a2c894-2162-4d49-86cf-435b9013f3b8"
gt_path = DATA_DIR / f"ground_truth_{r14_id[:8]}.json"

if gt_path.exists():
    with open(gt_path) as f:
        gt_data = json.load(f)
    print(f"R14 GT: {len(gt_data)} seeds saved")
else:
    gt_data = {}
    print("R14 GT: not yet saved")
missing = [str(i) for i in range(5) if str(i) not in gt_data]
print(f"Missing seeds: {missing}")

if missing:
    full = api._get(f"/rounds/{r14_id}")
    for si_str in missing:
        si = int(si_str)
        print(f"Fetching seed {si}...", end=" ", flush=True)
        try:
            analysis = api.get_analysis(r14_id, si)
            ig = full["initial_states"][si]["grid"]
            gt = analysis.get("ground_truth", analysis.get("probabilities"))
            if gt is not None:
                gt_data[si_str] = {"ground_truth": gt, "initial_grid": ig}
                print("OK")
            else:
                print("No GT in response")
        except Exception as e:
            print(f"FAIL: {e}")

    with open(gt_path, "w") as f:
        json.dump(gt_data, f)
    print(f"Saved R14 GT: {len(gt_data)} seeds")
else:
    print("R14 GT complete!")

# Check leaderboard
print("\n--- Leaderboard ---")
try:
    lb = api._get("/leaderboard")
    if isinstance(lb, list):
        for i, entry in enumerate(lb[:20]):
            name = entry.get("name", entry.get("team_name", "?"))
            score = entry.get("score", entry.get("total_score", "?"))
            print(f"  #{i+1}: {name} = {score}")
    else:
        print(f"Response type: {type(lb)}")
        print(str(lb)[:500])
except Exception as e:
    print(f"Leaderboard error: {e}")

# R15 budget
print("\n--- R15 Status ---")
b = api.get_budget()
print(f"Budget: {b}")
preds = api._get(f"/my-predictions/cc5442dd-bc5d-418b-911b-7eb960cb0390")
print(f"Predictions: {len(preds)} seeds submitted")
