import zipfile

z = zipfile.ZipFile(r'c:\ainm\object-detection\submission_v14_v3f.zip', 'r')
code = z.read('run.py').decode('utf-8')

checks = {
    'sigma=0.85': 'sigma=0.85' in code or 'sigma = 0.85' in code,
    'skip_box=0.004705': '0.004705' in code,
    'WBF iou=0.505': '0.505' in code,
    'snms_iou=0.303': '0.303' in code,
    'MAX_DETS=450': 'MAX_DETS = 450' in code or 'MAX_DETS=450' in code,
    'conf=0.01': 'conf=0.01' in code,
    'iou=0.7': 'iou=0.7' in code,
    'max_det=600': 'max_det=600' in code,
    'NO conf_type avg': ("conf_type" not in code) or ("'avg'" not in code),
    'TOTAL_BUDGET=270': '270' in code,
    'time_per_image < 3.0': '3.0' in code,
}

all_ok = True
for name, ok in checks.items():
    status = 'OK' if ok else 'FAIL'
    if not ok:
        all_ok = False
    print(f'  {status}: {name}')

# Check no v3 flip TTA
lines = code.split('\n')
v3_flip_found = False
for i, line in enumerate(lines):
    if 'best.onnx' in line and 'flip' in line.lower():
        v3_flip_found = True
        print(f'  FAIL: v3 flip found at line {i+1}: {line.strip()}')
if not v3_flip_found:
    print('  OK: No v3 flip TTA')

print(f'\nFiles in zip: {len(z.namelist())}')
for n in z.namelist():
    info = z.getinfo(n)
    print(f'  {n}: {info.file_size:,} bytes')

print(f'\nALL CHECKS PASSED: {all_ok and not v3_flip_found}')
z.close()
