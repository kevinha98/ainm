"""Fetch missing GT data for completed rounds."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from src.api import AstarAPI

DATA_DIR = Path("data")
api = AstarAPI()

rounds = api.get_rounds()
print(f"Total rounds: {len(rounds)}")
for r in rounds[-5:]:
    rn = r.get('round_number', '?')
    rid = r['id'][:8]
    status = r.get('status', '?')
    print(f"  R{rn}: {rid}... status={status}")

# Check for active/pending round
active = [r for r in rounds if r.get('status') == 'active']
pending = [r for r in rounds if r.get('status') == 'pending']
print(f"\nActive rounds: {len(active)}")
print(f"Pending rounds: {len(pending)}")
if active:
    for r in active:
        print(f"  ACTIVE R{r.get('round_number')}: {r['id'][:8]}")

# Fetch missing GT
fetched = 0
for r in rounds:
    if r.get('status') == 'completed':
        rid = r['id']
        gt_path = DATA_DIR / f"ground_truth_{rid[:8]}.json"
        if not gt_path.exists():
            rn = r.get('round_number', '?')
            print(f"\nMissing GT for R{rn}: {rid[:8]}")
            try:
                n_seeds = len(r.get('initial_states', []))
                gt_data = {}
                for si in range(n_seeds):
                    analysis = api.get_analysis(rid, si)
                    ig = r['initial_states'][si]['grid']
                    gt = analysis.get('ground_truth', analysis.get('probabilities'))
                    if gt and ig:
                        gt_data[str(si)] = {'ground_truth': gt, 'initial_grid': ig}
                        print(f"  Got seed {si}")
                    time.sleep(0.3)
                if gt_data:
                    with open(gt_path, 'w') as f:
                        json.dump(gt_data, f)
                    print(f"  Saved {gt_path.name} ({len(gt_data)} seeds)")
                    fetched += 1
            except Exception as e:
                print(f"  Error: {e}")

print(f"\nFetched {fetched} new GT files")

# Also check our R15 score
r15_id = 'cc5442dd-bc5d-418b-911b-7eb960cb0390'
try:
    my_rounds = api.get_my_rounds()
    for mr in my_rounds:
        if mr.get('id') == r15_id:
            print(f"\nR15 score: {mr.get('score', 'pending')}")
            print(f"R15 status: {mr.get('status', '?')}")
            break
except Exception as e:
    print(f"Could not check R15: {e}")

# Check leaderboard
try:
    lb = api._get("/leaderboard")
    if isinstance(lb, list):
        for entry in lb[:5]:
            print(f"  #{entry.get('rank', '?')} {entry.get('team_name', '?')}: {entry.get('score', '?')}")
        # Find us
        for entry in lb:
            if 'kevin' in str(entry.get('team_name', '')).lower() or 'ha' in str(entry.get('team_name', '')).lower():
                print(f"\n  US: #{entry.get('rank', '?')} {entry.get('team_name', '?')}: {entry.get('score', '?')}")
except Exception as e:
    print(f"Leaderboard: {e}")

print("\nDone")
