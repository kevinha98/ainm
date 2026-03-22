import zipfile
zf = zipfile.ZipFile('submission_v13_v3flip.zip', 'r')
code = zf.read('run.py').decode('utf-8')
checks = [
    ('conf=0.01', 'conf=0.01' in code),
    ('iou=0.7', 'iou=0.7' in code),
    ('max_det=600', 'max_det=600' in code),
    ('sigma=0.85', 'sigma=0.85' in code),
    ('skip_box_thresh=0.02', 'skip_box_thresh=0.02' in code),
    ("NO conf_type='avg'", "conf_type='avg'" not in code),
    ('MAX_DETS=500', 'MAX_DETS_PER_IMAGE = 500' in code),
    ('flip_v3 TTA', '_run_flip_tta(model_v3' in code),
    ('12 passes in docstring', '12 passes' in code),
    ('TOTAL_BUDGET=270', 'TOTAL_BUDGET_SEC = 270' in code),
]
all_ok = True
for name, ok in checks:
    status = 'OK' if ok else 'FAIL'
    if not ok:
        all_ok = False
    print(f'  [{status}] {name}')
print(f'\n{"ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"}')
print(f'Files in zip: {zf.namelist()}')
print(f'Zip size: {sum(i.file_size for i in zf.infolist())/1024/1024:.1f} MB uncompressed')
