import sys, json, os, time
sys.path.insert(0, "src")
from api import AstarAPI

api = AstarAPI()
rounds = api.get_rounds()
print(f"Total rounds: {len(rounds)}")

for r in rounds:
    rn = r.get("round_number", "?")
    st = r.get("status", "?")
    rid = r["id"][:8]
    print(f"  R{rn}: {st} ({rid})")

# Check active round
active = [r for r in rounds if r["status"] == "active"]
if active:
    rd = active[0]
    rid = rd["id"]
    full = api._get(f"/rounds/{rid}")
    b = api.get_budget()
    preds = api._get(f"/my-predictions/{rid}")
    print(f"\nActive: R{rd['round_number']} ({rid[:8]})")
    print(f"  End time: {full.get('end_time', '?')}")
    print(f"  Budget: {json.dumps(b)}")
    print(f"  Predictions: {len(preds)} submitted")
else:
    print("\nNo active round")

# Fetch R13 GT if missing
r13 = [r for r in rounds if r.get("round_number") == 13][0]
r13id = r13["id"]
gt_path = f"data/ground_truth_{r13id[:8]}.json"
if os.path.exists(gt_path):
    print(f"\nR13 GT already saved at {gt_path}")
else:
    print(f"\nFetching R13 GT...")
    full13 = api._get(f"/rounds/{r13id}")
    n_seeds = len(full13.get("initial_states", []))
    gt = {}
    for si in range(n_seeds):
        try:
            a = api.get_analysis(r13id, si)
            ig = full13["initial_states"][si]["grid"]
            g = a.get("ground_truth", a.get("probabilities"))
            if g is not None:
                gt[str(si)] = {"ground_truth": g, "initial_grid": ig}
                print(f"  Got GT seed {si}")
            time.sleep(0.5)
        except Exception as e:
            print(f"  Failed seed {si}: {e}")
    if gt:
        with open(gt_path, "w") as f:
            json.dump(gt, f)
        print(f"Saved {len(gt)} seeds to {gt_path}")
