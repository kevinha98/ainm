"""Test v5x (YOLOv8x) in various ensemble combinations vs current 3-model.

Since we can only use 3 weight files, v5x must replace one model.
Options to test:
  A: v3+v4+v6 (current baseline)
  B: v3+v4+v5x (replace v6 with v5x)
  C: v3+v5x+v6 (replace v4 with v5x)
  D: v5x+v4+v6 (replace v3 with v5x)
  E: v5x+v4 (2-model, no size concerns)
  F: v3+v5x (2-model)
"""
import pickle, json, time
import numpy as np
from ensemble_boxes import weighted_boxes_fusion

# Load all cached detections
print("Loading caches...")
c3 = pickle.load(open('cache_v3_29.pkl', 'rb'))
c4 = pickle.load(open('cache_v4_29.pkl', 'rb'))
c6 = pickle.load(open('cache_v6_29.pkl', 'rb'))
c5x = pickle.load(open('cache_v5x_29.pkl', 'rb'))

image_ids = sorted(set(c3.keys()) & set(c4.keys()) & set(c6.keys()) & set(c5x.keys()))
gt_data = json.load(open('data/coco/train/annotations.json'))
gt_by_image = {}
for ann in gt_data["annotations"]:
    gt_by_image.setdefault(ann["image_id"], []).append(ann)
img_info = {img["id"]: img for img in gt_data["images"]}
print(f"Common images: {len(image_ids)}")


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax+aw, bx+bw) - max(ax, bx))
    iy = max(0, min(ay+ah, by+bh) - max(ay, by))
    inter = ix * iy
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0


def soft_nms(dets, iou_thresh=0.303, sigma=0.949, score_thresh=6e-6):
    if not dets:
        return []
    dets = [d.copy() for d in dets]
    dets.sort(key=lambda x: x["score"], reverse=True)
    kept, groups = [], []
    while dets:
        best = dets.pop(0)
        kept.append(best)
        grp = [best.copy()]
        rem = []
        for d in dets:
            iou_val = _iou(best["bbox"], d["bbox"])
            if iou_val >= iou_thresh:
                grp.append(d.copy())
            d["score"] *= np.exp(-(iou_val ** 2) / sigma)
            if d["score"] >= score_thresh:
                rem.append(d)
        groups.append(grp)
        dets = sorted(rem, key=lambda x: x["score"], reverse=True)
    # Category voting (quadratic)
    for ki, k in enumerate(kept):
        if len(groups[ki]) > 1:
            cs = {}
            for ab in groups[ki]:
                cs[ab["category_id"]] = cs.get(ab["category_id"], 0) + ab["score"] ** 2
            kept[ki]["category_id"] = max(cs, key=cs.get)
    return kept


def wbf_fuse(passes, img_w, img_h, iou_thresh=0.4989, skip=0.004705, weights=None):
    bl, sl, ll = [], [], []
    for p in passes:
        b, s, l = [], [], []
        for d in p:
            x, y, w, h = d["bbox"]
            x1, y1 = max(0, x/img_w), max(0, y/img_h)
            x2, y2 = min(1, (x+w)/img_w), min(1, (y+h)/img_h)
            if x2 <= x1 or y2 <= y1:
                continue
            b.append([x1, y1, x2, y2])
            s.append(d["score"])
            l.append(d["category_id"])
        if b:
            bl.append(np.array(b, np.float32))
            sl.append(np.array(s, np.float32))
            ll.append(np.array(l, np.int32))
    if not bl:
        return []
    kw = dict(iou_thr=iou_thresh, skip_box_thr=skip, conf_type='box_and_model_avg')
    if weights:
        kw['weights'] = weights
    fb, fs, fl = weighted_boxes_fusion(bl, sl, ll, **kw)
    return [{"category_id": int(la), "bbox": [round(b[0]*img_w, 1), round(b[1]*img_h, 1),
             round((b[2]-b[0])*img_w, 1), round((b[3]-b[1])*img_h, 1)], "score": round(float(sc), 3)}
            for b, sc, la in zip(fb, fs, fl)]


def compute_ap(preds, gts, iou_thresh=0.5, check_class=False):
    if not gts:
        return 1.0 if not preds else 0.0
    if not preds:
        return 0.0
    preds = sorted(preds, key=lambda x: x["score"], reverse=True)
    matched = [False] * len(gts)
    tp, fp = [], []
    for p in preds:
        bi, bj = 0, -1
        for j, g in enumerate(gts):
            if check_class and p["category_id"] != g["category_id"]:
                continue
            iou_val = _iou(p["bbox"], g["bbox"])
            if iou_val > bi:
                bi, bj = iou_val, j
        if bi >= iou_thresh and bj >= 0 and not matched[bj]:
            tp.append(1); fp.append(0); matched[bj] = True
        else:
            tp.append(0); fp.append(1)
    tc = np.cumsum(tp); fc = np.cumsum(fp)
    rec = tc / len(gts); prec = tc / (tc + fc)
    mr = np.concatenate(([0.], rec, [1.]))
    mp = np.concatenate(([1.], prec, [0.]))
    for i in range(len(mp)-2, -1, -1):
        mp[i] = max(mp[i], mp[i+1])
    idx = np.where(mr[1:] != mr[:-1])[0]
    return np.sum((mr[idx+1] - mr[idx]) * mp[idx+1])


def evaluate(passes_fn, weights, max_dets=450):
    """Evaluate an ensemble configuration."""
    da, ca = [], []
    for iid in image_ids:
        info = img_info.get(iid)
        if not info:
            continue
        iw, ih = info["width"], info["height"]
        ap = passes_fn(iid)
        if len(ap) > 1:
            fused = wbf_fuse(ap, iw, ih, weights=weights)
        else:
            fused = ap[0] if ap else []
        dets = soft_nms(fused)
        if len(dets) > max_dets:
            dets.sort(key=lambda x: x["score"], reverse=True)
            dets = dets[:max_dets]
        gts = [{"bbox": g["bbox"], "category_id": g["category_id"]} for g in gt_by_image.get(iid, [])]
        dp = [{"bbox": d["bbox"], "score": d["score"], "category_id": 0} for d in dets]
        dg = [{"bbox": g["bbox"], "category_id": 0} for g in gts]
        da.append(compute_ap(dp, dg))
        ca.append(compute_ap(dets, gts, check_class=True))
    det = np.mean(da); cls = np.mean(ca)
    return 0.7 * det + 0.3 * cls, det, cls


# Check what scales v5x has cached
sample_id = image_ids[0]
v5x_keys = [k for k in c5x[sample_id].keys() if k.startswith('full_')]
print(f"v5x scales: {v5x_keys}")
v3_keys = [k for k in c3[sample_id].keys() if k.startswith('full_')]
print(f"v3 scales: {v3_keys}")

print("\n" + "="*60)

# A: Current baseline v3+v4+v6 (8 passes)
def passes_A(iid):
    return [
        c3[iid]['full_1280'], c3[iid]['full_1408'], c3[iid]['full_1536'],
        c4[iid]['full_1280'], c4[iid]['full_1536'],
        c6[iid]['full_1280'], c6[iid]['full_1408'], c6[iid]['full_1536']
    ]
c, d, cl = evaluate(passes_A, [1,2,1,2,3,4,1,2])
print(f"A: v3+v4+v6 (8 passes): {c:.5f} (det={d:.5f} cls={cl:.5f})")

# B: Replace v6 with v5x: v3+v4+v5x (8 passes)
def passes_B(iid):
    return [
        c3[iid]['full_1280'], c3[iid]['full_1408'], c3[iid]['full_1536'],
        c4[iid]['full_1280'], c4[iid]['full_1536'],
        c5x[iid]['full_1280'], c5x[iid]['full_1408'], c5x[iid]['full_1536']
    ]
c, d, cl = evaluate(passes_B, [1,2,1,2,3,4,1,2])
print(f"B: v3+v4+v5x (8 passes): {c:.5f} (det={d:.5f} cls={cl:.5f})")

# C: Replace v4 with v5x: v3+v5x+v6 (9 passes)
def passes_C(iid):
    return [
        c3[iid]['full_1280'], c3[iid]['full_1408'], c3[iid]['full_1536'],
        c5x[iid]['full_1280'], c5x[iid]['full_1408'], c5x[iid]['full_1536'],
        c6[iid]['full_1280'], c6[iid]['full_1408'], c6[iid]['full_1536']
    ]
c, d, cl = evaluate(passes_C, [1,2,1,2,1,1,4,1,2])
print(f"C: v3+v5x+v6 (9 passes): {c:.5f} (det={d:.5f} cls={cl:.5f})")

# D: Replace v3 with v5x: v5x+v4+v6 (8 passes)
def passes_D(iid):
    return [
        c5x[iid]['full_1280'], c5x[iid]['full_1408'], c5x[iid]['full_1536'],
        c4[iid]['full_1280'], c4[iid]['full_1536'],
        c6[iid]['full_1280'], c6[iid]['full_1408'], c6[iid]['full_1536']
    ]
c, d, cl = evaluate(passes_D, [1,2,1,2,3,4,1,2])
print(f"D: v5x+v4+v6 (8 passes): {c:.5f} (det={d:.5f} cls={cl:.5f})")

# E: All 4 models (hypothetical - need FP16 for v5x)
def passes_E(iid):
    return [
        c3[iid]['full_1280'], c3[iid]['full_1408'], c3[iid]['full_1536'],
        c4[iid]['full_1280'], c4[iid]['full_1536'],
        c5x[iid]['full_1280'], c5x[iid]['full_1408'], c5x[iid]['full_1536'],
        c6[iid]['full_1280'], c6[iid]['full_1408'], c6[iid]['full_1536']
    ]
c, d, cl = evaluate(passes_E, [1,2,1,2,3,1,1,1,4,1,2])
print(f"E: v3+v4+v5x+v6 (11 passes): {c:.5f} (det={d:.5f} cls={cl:.5f})")

# Quick weight sweep for best configurations
print("\n--- Weight sweep for top configs ---")
best_overall = 0
best_config = None

configs = {
    'B': (passes_B, 8),
    'D': (passes_D, 8),
}

for name, (fn, n_passes) in configs.items():
    best_local = 0
    best_w = None
    # Try various weight patterns
    for w_pattern in [
        [1]*n_passes,
        [1,2,1,2,3,4,1,2][:n_passes],
        [1,1,1,2,3,1,1,1][:n_passes],
        [2,2,1,2,3,4,2,2][:n_passes],
        [1,2,1,3,4,2,1,1][:n_passes],
        [1,2,1,2,3,3,1,2][:n_passes],
        [1,1,1,1,2,4,2,2][:n_passes],
    ]:
        c, d, cl = evaluate(fn, w_pattern)
        if c > best_local:
            best_local = c
            best_w = w_pattern
    print(f"  {name} best: {best_local:.5f} with weights={best_w}")
    if best_local > best_overall:
        best_overall = best_local
        best_config = (name, best_w)

print(f"\nBest alternative: {best_config[0]} = {best_overall:.5f} (vs A baseline 0.95155)")
