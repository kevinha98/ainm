import re
import os
home = os.path.expanduser('~')
lines = open(os.path.join(home, 'object-detection/train_v5x.log')).readlines()
maps = []
for l in lines:
    if '  all  ' in l:
        parts = l.split()
        maps.append(float(parts[-2]))  # mAP50
print(f'Epochs with val: {len(maps)}')
print(f'Best mAP50: {max(maps):.4f} at epoch ~{maps.index(max(maps))+1}')
print(f'Last 10 mAP50: {[round(m,4) for m in maps[-10:]]}')
print(f'Plateau? Last 10 range: {max(maps[-10:])-min(maps[-10:]):.4f}')
print(f'First 10 mAP50: {[round(m,4) for m in maps[:10]]}')
