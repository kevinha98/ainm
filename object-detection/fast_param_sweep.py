"""Fast parameter sweep for conf_type, skip_box, max_dets with cached detections."""
import pickle, json, numpy as np, os, time, sys, signal
from ensemble_boxes import weighted_boxes_fusion

os.chdir(os.path.dirname(os.path.abspath(__file__)))

c3 = pickle.load(open('cache_v3_29.pkl', 'rb'))
c4 = pickle.load(open('cache_v4_29.pkl', 'rb'))
c6 = pickle.load(open('cache_v6_29.pkl', 'rb'))
image_ids = sorted(set(c3.keys()) & set(c4.keys()) & set(c6.keys()))
gt_data = json.load(open('data/coco/train/annotations.json'))
gt_by_image = {}
for ann in gt_data['annotations']:
    gt_by_image.setdefault(ann['image_id'], []).append(ann)
img_info = {img['id']: img for img in gt_data['images']}
print(f"Loaded {len(image_ids)} images")


def _iou_np(boxes_a, box_b):
    """Vectorized IoU between array of boxes and single box."""
    ax, ay, aw, ah = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx, by, bw, bh = box_b
    ix = np.maximum(0, np.minimum(ax + aw, bx + bw) - np.maximum(ax, bx))
    iy = np.maximum(0, np.minimum(ay + ah, by + bh) - np.maximum(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return np.where(union > 0, inter / union, 0)


def soft_nms_fast(dets, iou_thresh=0.35, sigma=0.85, score_thresh=1e-5, max_dets=500):
    if not dets:
        return []
    n = len(dets)
    boxes = np.array([d['bbox'] for d in dets])
    scores = np.array([d['score'] for d in dets])
    cats = np.array([d['category_id'] for d in dets])

    order = np.argsort(-scores)
    boxes = boxes[order]
    scores = scores[order].copy()
    cats = cats[order]

    kept_idx = []
    absorbed_groups = []
    alive = np.ones(n, dtype=bool)

    for i in range(n):
        if not alive[i]:
            continue
        if scores[i] < score_thresh:
            break
        kept_idx.append(i)
        grp_cats = [cats[i]]
        grp_scores = [scores[i]]

        # Compute IoU against all remaining
        remaining = np.where(alive)[0]
        remaining = remaining[remaining > i]
        if len(remaining) == 0:
            absorbed_groups.append((grp_cats, grp_scores))
            continue

        ious = _iou_np(boxes[remaining], boxes[i])
        absorbed_mask = ious >= iou_thresh
        for j_idx in np.where(absorbed_mask)[0]:
            grp_cats.append(cats[remaining[j_idx]])
            grp_scores.append(scores[remaining[j_idx]])

        # Gaussian decay
        scores[remaining] *= np.exp(-(ious ** 2) / sigma)
        alive[remaining] = scores[remaining] >= score_thresh
        absorbed_groups.append((grp_cats, grp_scores))

    # Category voting (quadratic score)
    result = []
    for ki, idx in enumerate(kept_idx):
        grp_c, grp_s = absorbed_groups[ki]
        if len(grp_c) > 1:
            cs = {}
            for c, s in zip(grp_c, grp_s):
                cs[c] = cs.get(c, 0) + s ** 2
            best_cat = max(cs, key=cs.get)
        else:
            best_cat = cats[idx]
        result.append({
            'category_id': int(best_cat),
            'bbox': boxes[idx].tolist(),
            'score': float(scores[idx]),
        })
        if len(result) >= max_dets:
            break
    return result


def wbf(passes, iw, ih, iou_thresh=0.50, skip=0.005, ct='box_and_model_avg', weights=None):
    bl, sl, ll = [], [], []
    for p in passes:
        b, s, l = [], [], []
        for d in p:
            x, y, w, h = d['bbox']
            x1, y1 = max(0, x / iw), max(0, y / ih)
            x2, y2 = min(1, (x + w) / iw), min(1, (y + h) / ih)
            if x2 <= x1 or y2 <= y1:
                continue
            b.append([x1, y1, x2, y2])
            s.append(d['score'])
            l.append(d['category_id'])
        if b:
            bl.append(np.array(b, np.float32))
            sl.append(np.array(s, np.float32))
            ll.append(np.array(l, np.int32))
    if not bl:
        return []
    kw = dict(iou_thr=iou_thresh, skip_box_thr=skip, conf_type=ct)
    if weights:
        kw['weights'] = weights
    fb, fs, fl = weighted_boxes_fusion(bl, sl, ll, **kw)
    return [{'category_id': int(la),
             'bbox': [round(b[0] * iw, 1), round(b[1] * ih, 1),
                      round((b[2] - b[0]) * iw, 1), round((b[3] - b[1]) * ih, 1)],
             'score': round(float(sc), 3)}
            for b, sc, la in zip(fb, fs, fl)]


def compute_ap(preds, gts, check_class=False):
    if not gts:
        return 1.0 if not preds else 0.0
    if not preds:
        return 0.0
    preds = sorted(preds, key=lambda x: x['score'], reverse=True)
    matched = [False] * len(gts)
    tp, fp = [], []
    for p in preds:
        bi, bj = 0, -1
        for j, g in enumerate(gts):
            if check_class and p['category_id'] != g['category_id']:
                continue
            iou_val = 0
            ax, ay, aw, ah = p['bbox']
            bx, by, bw, bh = g['bbox']
            ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
            iy = max(0, min(ay + ah, by + bh) - max(ay, by))
            inter = ix * iy
            union = aw * ah + bw * bh - inter
            iou_val = inter / union if union > 0 else 0
            if iou_val > bi:
                bi, bj = iou_val, j
        if bi >= 0.5 and bj >= 0 and not matched[bj]:
            tp.append(1)
            fp.append(0)
            matched[bj] = True
        else:
            tp.append(0)
            fp.append(1)
    tc = np.cumsum(tp)
    fc = np.cumsum(fp)
    rec = tc / len(gts)
    prec = tc / (tc + fc)
    mr = np.concatenate(([0.], rec, [1.]))
    mp = np.concatenate(([1.], prec, [0.]))
    for i in range(len(mp) - 2, -1, -1):
        mp[i] = max(mp[i], mp[i + 1])
    idx = np.where(mr[1:] != mr[:-1])[0]
    return np.sum((mr[idx + 1] - mr[idx]) * mp[idx + 1])


def evaluate(ct='box_and_model_avg', skip=0.005, max_dets=400, sigma=0.85, snms_iou=0.35):
    import gc
    gc.collect()
    da, ca = [], []
    wt = [1, 2, 1, 2, 3, 4, 1, 2]
    for idx, iid in enumerate(image_ids):
        info = img_info.get(iid)
        if not info:
            continue
        iw, ih = info['width'], info['height']
        ap = [
            c3[iid]['full_1280'], c3[iid]['full_1408'], c3[iid]['full_1536'],
            c4[iid]['full_1280'], c4[iid]['full_1536'],
            c6[iid]['full_1280'], c6[iid]['full_1408'], c6[iid]['full_1536']
        ]
        fused = wbf(ap, iw, ih, skip=skip, ct=ct, weights=wt)
        dets = soft_nms_fast(fused, iou_thresh=snms_iou, sigma=sigma, max_dets=max_dets)
        gts = [{'bbox': g['bbox'], 'category_id': g['category_id']} for g in gt_by_image.get(iid, [])]
        dp = [{'bbox': d['bbox'], 'score': d['score'], 'category_id': 0} for d in dets]
        dg = [{'bbox': g['bbox'], 'category_id': 0} for g in gts]
        da.append(compute_ap(dp, dg))
        ca.append(compute_ap(dets, gts, check_class=True))
    det = np.mean(da)
    cls = np.mean(ca)
    return 0.7 * det + 0.3 * cls, det, cls


if __name__ == '__main__':
    import subprocess, json as _json

    configs = [
        ('PREV_BEST bma/0.005/450/0.949/0.303', dict(ct='box_and_model_avg', skip=0.004705, max_dets=450, sigma=0.949, snms_iou=0.303)),
        ('sigma85   bma/0.02/400/0.85/0.35',    dict(ct='box_and_model_avg', skip=0.02, max_dets=400, sigma=0.85, snms_iou=0.35)),
        ('bma/0.02/500/0.85/0.35',              dict(ct='box_and_model_avg', skip=0.02, max_dets=500, sigma=0.85, snms_iou=0.35)),
        ('bma/0.01/400/0.85/0.35',              dict(ct='box_and_model_avg', skip=0.01, max_dets=400, sigma=0.85, snms_iou=0.35)),
        ('bma/0.01/500/0.85/0.35',              dict(ct='box_and_model_avg', skip=0.01, max_dets=500, sigma=0.85, snms_iou=0.35)),
        ('bma/0.02/400/0.85/0.30',              dict(ct='box_and_model_avg', skip=0.02, max_dets=400, sigma=0.85, snms_iou=0.30)),
        ('bma/0.02/400/0.90/0.35',              dict(ct='box_and_model_avg', skip=0.02, max_dets=400, sigma=0.90, snms_iou=0.35)),
        ('bma/0.02/400/0.80/0.35',              dict(ct='box_and_model_avg', skip=0.02, max_dets=400, sigma=0.80, snms_iou=0.35)),
        ('max/0.02/400/0.85/0.35',              dict(ct='max', skip=0.02, max_dets=400, sigma=0.85, snms_iou=0.35)),
        ('max/0.02/500/0.85/0.35',              dict(ct='max', skip=0.02, max_dets=500, sigma=0.85, snms_iou=0.35)),
        ('bma/0.02/400/0.85/0.40',              dict(ct='box_and_model_avg', skip=0.02, max_dets=400, sigma=0.85, snms_iou=0.40)),
        ('bma/0.015/400/0.85/0.35',             dict(ct='box_and_model_avg', skip=0.015, max_dets=400, sigma=0.85, snms_iou=0.35)),
        ('bma/0.02/600/0.85/0.35',              dict(ct='box_and_model_avg', skip=0.02, max_dets=600, sigma=0.85, snms_iou=0.35)),
    ]

    print(f'Testing {len(configs)} configs (8 passes, no flip) with subprocess isolation...')
    print('=' * 90)
    sys.stdout.flush()
    results = []

    eval_script = '''
import sys, json
sys.path.insert(0, ".")
from fast_param_sweep import evaluate
kw = json.loads(sys.argv[1])
c, d, cl = evaluate(**kw)
print(json.dumps({"c": c, "d": d, "cl": cl}))
'''

    for label, kw in configs:
        t0 = time.time()
        try:
            result = subprocess.run(
                [sys.executable, '-c', eval_script, _json.dumps(kw)],
                capture_output=True, text=True, timeout=60, cwd=os.getcwd()
            )
            dt = time.time() - t0
            if result.returncode == 0:
                r = _json.loads(result.stdout.strip().split('\n')[-1])
                c, d, cl = r['c'], r['d'], r['cl']
                results.append((label, c, d, cl))
                print(f'  {label:42s} | {c:.5f} | Det={d:.5f} | Cls={cl:.5f} | {dt:.1f}s')
            else:
                print(f'  {label:42s} | FAIL: {result.stderr[:100]} ({dt:.1f}s)')
        except subprocess.TimeoutExpired:
            dt = time.time() - t0
            print(f'  {label:42s} | TIMEOUT ({dt:.1f}s)')
        except Exception as e:
            dt = time.time() - t0
            print(f'  {label:42s} | ERROR: {e} ({dt:.1f}s)')
        sys.stdout.flush()

    results.sort(key=lambda x: x[1], reverse=True)
    print()
    print('RANKED:')
    best = results[0][1]
    for i, (l, c, d, cl) in enumerate(results):
        delta = c - results[-1][1]
        print(f'  #{i+1} {l:42s} | {c:.5f} ({c-best:+.5f}) | Det={d:.5f} | Cls={cl:.5f}')
