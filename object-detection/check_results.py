import json, glob

results = []
for f in sorted(glob.glob('sims/sim_*_result.json')):
    with open(f) as fh:
        d = json.load(fh)
        results.append((d['best']['combined'], d['sim_id'], d['name'], d['best'].get('kwargs',{})))

results.sort(reverse=True)
sim_scripts = len(glob.glob('sims/sim_*.py'))
print(f'Completed sims: {len(results)} | Sim scripts: {sim_scripts}')
print()
print('TOP 20 RESULTS:')
for i, (score, sid, name, kw) in enumerate(results[:20]):
    w = kw.get('weights','?')
    wbf = kw.get('wbf_iou','?')
    sig = kw.get('snms_sigma','?')
    print(f'  #{i+1}: sim_{sid:03d} | {score:.5f} | wbf={wbf} sig={sig} w={w} | {name}')
print()
print('WORST 5:')
for score, sid, name, kw in results[-5:]:
    print(f'  sim_{sid:03d} | {score:.5f} | {name}')
