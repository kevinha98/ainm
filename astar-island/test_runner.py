"""Quick test of auto_runner components."""
from auto_runner import *
api = AstarAPI()
state = load_state()
print("State:", state)
rounds = api.get_rounds()
for r in rounds:
    rn = r["round_number"]
    st = r["status"]
    rid = r["id"][:8]
    print(f"  Round {rn}: {st} id={rid}")

# Fetch GT for completed rounds
for r in rounds:
    if r["status"] == "completed":
        fetch_gt_for_round(api, r["id"])

print()
gt_files = sorted(DATA_DIR.glob("ground_truth_*.json"))
print(f"GT files ({len(gt_files)}):", [f.name for f in gt_files])
