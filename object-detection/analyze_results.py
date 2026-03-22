"""Analyze all 100 sim results + elite optimizer results, find absolute best config."""
import json, glob, os
from datetime import datetime, timedelta

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Read all sim results
scores = []
for r in sorted(glob.glob('sims/sim_*_result.json')):
    try:
        d = json.load(open(r))
        b = d['best']
        c = b['combined']
        scores.append((c, d['sim_id'], d.get('name',''), b))
    except Exception as e:
        pass

scores.sort(reverse=True)
print(f"Sims with results: {len(scores)}")
print(f"\nTOP 20 SIM CONFIGS:")
for i, (c, sid, name, b) in enumerate(scores[:20]):
    kw = b.get('kwargs', {})
    w = kw.get('weights', '?')
    wbf = kw.get('wbf_iou', '?')
    sig = kw.get('snms_sigma', '?')
    print(f"  #{i+1}: {c:.5f} sim_{sid:03d} wbf={wbf} sig={sig} w={w}")

# Elite results
for fname in ['sims/ELITE_BEST.json', 'sims/ELITE_FINAL.json']:
    if os.path.exists(fname):
        d = json.load(open(fname))
        if 'combined' in d:
            kw = d['kwargs']
            print(f"\n{fname}:")
            print(f"  Score: {d['combined']:.5f} (det={d['det']:.5f} cls={d['cls']:.5f})")
            print(f"  Weights: {kw['weights']}")
            print(f"  wbf={kw['wbf_iou']}, sig={kw['snms_sigma']}, siou={kw['snms_iou']}")
            print(f"  sscore={kw['snms_score']}, maxd={kw['max_dets']}, vote={kw['vote_mode']}")
            print(f"  Evals: {d.get('eval_count', '?')}")
        elif 'global_best' in d:
            gb = d['global_best']
            kw = gb['kwargs']
            print(f"\n{fname}: Score={gb['combined']:.5f}, evals={d.get('total_evals','?')}")

# Overnight log
lines = open('overnight_log.md', errors='replace').readlines()
print(f"\nOvernight log: {len(lines)} lines")

now = datetime.now()
midnight = now.replace(hour=0, minute=0, second=0) + timedelta(days=1)
remaining = midnight - now
print(f"\nTime: {now.strftime('%H:%M:%S')}, Until midnight: {remaining}")
