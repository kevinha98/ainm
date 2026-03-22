"""Quick check: R22 GT availability + round status."""
import json, sys
sys.path.insert(0, '.')
from src.api import AstarAPI
from src.settings import DATA_DIR

api = AstarAPI()
rounds = api.get_rounds()

for r in rounds:
    rn = r.get('round_number', '?')
    st = r['status']
    rid = r['id'][:8]
    close = r.get('closes_at', '?')
    if isinstance(rn, int) and rn >= 20:
        print(f"R{rn}: status={st} id={rid} closes={close}")

print()

# Try to fetch R22 GT
for r in rounds:
    if r.get('round_number') == 22:
        if r['status'] == 'completed':
            print("R22 completed! Fetching GT...")
            try:
                full_r = api._get(f"/rounds/{r['id']}")
                n_s = len(full_r.get("initial_states", []))
                gt_data = {}
                for si in range(n_s):
                    analysis = api.get_analysis(r['id'], si)
                    ig = full_r["initial_states"][si]["grid"]
                    gt = analysis.get("ground_truth", analysis.get("probabilities"))
                    if gt is not None:
                        gt_data[str(si)] = {"ground_truth": gt, "initial_grid": ig}
                gt_path = DATA_DIR / f"ground_truth_{r['id'][:8]}.json"
                if gt_data:
                    with open(gt_path, "w") as f:
                        json.dump(gt_data, f)
                    print(f"Saved {gt_path.name} ({len(gt_data)} seeds)")
                else:
                    print("No GT data in analysis response")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print(f"R22 still {r['status']} — no GT available yet")
        break
