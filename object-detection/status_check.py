import glob, json

# Count sims
sims = glob.glob('sims/sim_*_result.json')
print(f'Factory sims: {len(sims)}')

# Check elite best
try:
    with open('sims/ELITE_BEST.json') as f:
        eb = json.load(f)
    score = eb.get('combined', 0)
    evals = eb.get('eval_count', '?')
    wbf = eb.get('kwargs', {}).get('wbf_iou', '?')
    sigma = eb.get('kwargs', {}).get('snms_sigma', '?')
    weights = eb.get('kwargs', {}).get('weights', '?')
    print(f'Elite best: {score:.5f} (evals={evals})')
    print(f'  wbf={wbf}, sigma={sigma}, weights={weights}')
except Exception as e:
    print(f'No elite best yet: {e}')

# Log lines
lines = open('overnight_log.md', errors='replace').readlines()
print(f'Log lines: {len(lines)}')

# Last NEW BEST
for l in reversed(lines):
    if 'NEW BEST' in l or 'NEW SA BEST' in l:
        print(f'Last best: {l.strip()[:150]}')
        break

# Last 5 lines
for l in lines[-5:]:
    print(l.rstrip()[:150])
