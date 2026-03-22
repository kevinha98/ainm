import json, numpy as np

for rid in ['71451d74', '76909e29', 'f1dac9a9', '8e839974', 'fd3c92ff', 'ae78003a', '36e581f1', 'c5cdf100', '2a341ace']:
    path = f'data/ground_truth_{rid}.json'
    with open(path) as f:
        data = json.load(f)
    keys = sorted(data.keys())
    print(f'{rid}: {len(keys)} keys: {keys[:5]}')
    for k in keys[:1]:
        v = data[k]
        if isinstance(v, dict):
            subkeys = list(v.keys())[:3]
            first_val = list(v.values())[0]
            if isinstance(first_val, list):
                arr = np.array(first_val)
                print(f'  key={k}: dict with {len(v)} subkeys, first val shape={arr.shape}')
            else:
                print(f'  key={k}: dict with {len(v)} subkeys, first val type={type(first_val).__name__}')
        elif isinstance(v, list):
            arr = np.array(v)
            print(f'  key={k}: list, array shape={arr.shape}')
    print()
