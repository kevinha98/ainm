"""Sweep 3-model ensemble (v3 + v4 + v6) using cached detections.
Tests various combinations and weight ratios to find optimal config.
"""
import pickle, json, time
import numpy as np
from ensemble_boxes import weighted_boxes_fusion

print("Loading caches...")
c3 = pickle.load(open('cache_v3_29.pkl', 'rb'))
c4 = pickle.load(open('cache_v4_29.pkl', 'rb'))
c6 = pickle.load(open('cache_v6_29.pkl', 'rb'))
image_ids = sorted(c3.keys())
gt_data = json.load(open('data/coco/train/annotations.json'))
gt_by_image = {}
for ann in gt_data["annotations"]:
    gt_by_image.setdefault(ann["image_id"], []).append(ann)
img_info = {img["id"]: img for img in gt_data["images"]}
print(f"Images: {len(image_ids)}")

# Check v6 has same images
v6_ids = sorted(c6.keys())
print(f"v6 images: {len(v6_ids)}")
common = sorted(set(image_ids) & set(v6_ids))
print(f"Common: {len(common)}")
image_ids = common  # Use only common images

def _iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax+aw, bx+bw) - max(ax, bx))
    iy = max(0, min(ay+ah, by+bh) - max(ay, by))
    inter = ix * iy; union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0

def soft_nms(dets, iou_thresh=0.45, sigma=5.0, score_thresh=0.001):
    if not dets: return dets
    dets = [d.copy() for d in dets]
    dets.sort(key=lambda x: x["score"], reverse=True)
    kept = []
    while dets:
        best = dets.pop(0)
        kept.append(best)
        remaining = []
        for d in dets:
            iou = _iou(best["bbox"], d["bbox"])
            d["score"] *= np.exp(-(iou**2)/sigma)
            if d["score"] >= score_thresh: remaining.append(d)
        dets = sorted(remaining, key=lambda x: x["score"], reverse=True)
    return kept

def wbf(passes, img_w, img_h, iou_thresh=0.55, skip=0.005, ct='box_and_model_avg', weights=None):
    bl, sl, ll = [], [], []
    for p in passes:
        b, s, l = [], [], []
        for d in p:
            x, y, w, h = d["bbox"]
            x1, y1 = max(0, x/img_w), max(0, y/img_h)
            x2, y2 = min(1, (x+w)/img_w), min(1, (y+h)/img_h)
            if x2 <= x1 or y2 <= y1: continue
            b.append([x1,y1,x2,y2]); s.append(d["score"]); l.append(d["category_id"])
        if b:
            bl.append(np.array(b, np.float32))
            sl.append(np.array(s, np.float32))
            ll.append(np.array(l, np.int32))
    if not bl: return []
    kwargs = dict(iou_thr=iou_thresh, skip_box_thr=skip, conf_type=ct)
    if weights: kwargs['weights'] = weights
    fb, fs, fl = weighted_boxes_fusion(bl, sl, ll, **kwargs)
    return [{"category_id": int(la), "bbox": [round(b[0]*img_w,1), round(b[1]*img_h,1),
             round((b[2]-b[0])*img_w,1), round((b[3]-b[1])*img_h,1)], "score": round(float(sc),3)}
            for b, sc, la in zip(fb, fs, fl)]

def compute_ap(preds, gts, iou_thresh=0.5, check_class=False):
    if not gts: return 1.0 if not preds else 0.0
    if not preds: return 0.0
    preds = sorted(preds, key=lambda x: x["score"], reverse=True)
    matched = [False]*len(gts)
    tp, fp = [], []
    for p in preds:
        best_iou, best_j = 0, -1
        for j, g in enumerate(gts):
            if check_class and p["category_id"] != g["category_id"]: continue
            iou = _iou(p["bbox"], g["bbox"])
            if iou > best_iou: best_iou, best_j = iou, j
        if best_iou >= iou_thresh and best_j >= 0 and not matched[best_j]:
            tp.append(1); fp.append(0); matched[best_j] = True
        else:
            tp.append(0); fp.append(1)
    tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
    rec = tp_c / len(gts); prec = tp_c / (tp_c + fp_c)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    for i in range(len(mpre)-2, -1, -1): mpre[i] = max(mpre[i], mpre[i+1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1])

def evaluate(passes_fn, weights=None, wbf_iou=0.55, max_dets=300):
    det_aps, cls_aps = [], []
    for iid in image_ids:
        info = img_info.get(iid)
        if not info: continue
        img_w, img_h = info["width"], info["height"]
        passes = passes_fn(iid)
        if len(passes) > 1:
            fused = wbf(passes, img_w, img_h, iou_thresh=wbf_iou, weights=weights)
        else:
            fused = passes[0] if passes else []
        dets = soft_nms(fused)
        if len(dets) > max_dets:
            dets.sort(key=lambda x: x["score"], reverse=True); dets = dets[:max_dets]
        gts = [{"bbox": g["bbox"], "category_id": g["category_id"]} for g in gt_by_image.get(iid, [])]
        det_p = [{"bbox": d["bbox"], "score": d["score"], "category_id": 0} for d in dets]
        det_g = [{"bbox": g["bbox"], "category_id": 0} for g in gts]
        det_aps.append(compute_ap(det_p, det_g))
        cls_aps.append(compute_ap(dets, gts, check_class=True))
    det = np.mean(det_aps); cls = np.mean(cls_aps)
    return 0.7*det + 0.3*cls, det, cls

# ============================================================================
# BASELINES
# ============================================================================
print("\n" + "="*90)
print("BASELINES")

# Current best: v3+v4 with weights=[1,1,1,2,2]
def passes_v3v4(iid):
    return [c3[iid]['full_1280'], c3[iid]['full_1408'], c3[iid]['full_1536'],
            c4[iid]['full_1280'], c4[iid]['full_1536']]

c, d, cl = evaluate(passes_v3v4, weights=[1,1,1,2,2])
print(f"  v3+v4 (current best, w=[1,1,1,2,2]):  {c:.4f} (det={d:.4f} cls={cl:.4f})")

# v6 solo at different scales
def passes_v6_1280(iid):
    return [c6[iid]['full_1280']]
c, d, cl = evaluate(passes_v6_1280)
print(f"  v6 solo @1280:                         {c:.4f} (det={d:.4f} cls={cl:.4f})")

def passes_v6_1536(iid):
    return [c6[iid]['full_1536']]
c, d, cl = evaluate(passes_v6_1536)
print(f"  v6 solo @1536:                         {c:.4f} (det={d:.4f} cls={cl:.4f})")

def passes_v6_multi(iid):
    return [c6[iid]['full_1280'], c6[iid]['full_1408'], c6[iid]['full_1536']]
c, d, cl = evaluate(passes_v6_multi)
print(f"  v6 multi (1280+1408+1536):             {c:.4f} (det={d:.4f} cls={cl:.4f})")

# ============================================================================
# TEST 1: v3+v4+v6 all scales (8 passes) 
# ============================================================================
print("\n" + "="*90)
print("TEST 1: v3+v4+v6 ALL SCALES (8 passes)")

def passes_all8(iid):
    return [c3[iid]['full_1280'], c3[iid]['full_1408'], c3[iid]['full_1536'],
            c4[iid]['full_1280'], c4[iid]['full_1536'],
            c6[iid]['full_1280'], c6[iid]['full_1408'], c6[iid]['full_1536']]

# Equal weights
c, d, cl = evaluate(passes_all8)
print(f"  equal weights:                {c:.4f} (det={d:.4f} cls={cl:.4f})")

# v3=1 v4=2 v6=1 
c, d, cl = evaluate(passes_all8, weights=[1,1,1,2,2,1,1,1])
print(f"  v3=1 v4=2 v6=1:              {c:.4f} (det={d:.4f} cls={cl:.4f})")

# v3=1 v4=2 v6=2
c, d, cl = evaluate(passes_all8, weights=[1,1,1,2,2,2,2,2])
print(f"  v3=1 v4=2 v6=2:              {c:.4f} (det={d:.4f} cls={cl:.4f})")

# v3=1 v4=2 v6=3
c, d, cl = evaluate(passes_all8, weights=[1,1,1,2,2,3,3,3])
print(f"  v3=1 v4=2 v6=3:              {c:.4f} (det={d:.4f} cls={cl:.4f})")

# v3=1 v4=3 v6=2
c, d, cl = evaluate(passes_all8, weights=[1,1,1,3,3,2,2,2])
print(f"  v3=1 v4=3 v6=2:              {c:.4f} (det={d:.4f} cls={cl:.4f})")

# ============================================================================
# TEST 2: v3+v6 only (drop v4 — save time budget for more v6 passes?)
# ============================================================================
print("\n" + "="*90)
print("TEST 2: v3+v6 (no v4)")

def passes_v3v6(iid):
    return [c3[iid]['full_1280'], c3[iid]['full_1408'], c3[iid]['full_1536'],
            c6[iid]['full_1280'], c6[iid]['full_1408'], c6[iid]['full_1536']]

c, d, cl = evaluate(passes_v3v6)
print(f"  equal weights:                {c:.4f} (det={d:.4f} cls={cl:.4f})")

c, d, cl = evaluate(passes_v3v6, weights=[1,1,1,2,2,2])
print(f"  v3=1 v6=2:                    {c:.4f} (det={d:.4f} cls={cl:.4f})")

c, d, cl = evaluate(passes_v3v6, weights=[2,2,2,1,1,1])
print(f"  v3=2 v6=1:                    {c:.4f} (det={d:.4f} cls={cl:.4f})")

# ============================================================================
# TEST 3: v4+v6 only (drop v3)
# ============================================================================
print("\n" + "="*90)
print("TEST 3: v4+v6 (no v3)")

def passes_v4v6(iid):
    return [c4[iid]['full_1280'], c4[iid]['full_1536'],
            c6[iid]['full_1280'], c6[iid]['full_1408'], c6[iid]['full_1536']]

c, d, cl = evaluate(passes_v4v6)
print(f"  equal weights:                {c:.4f} (det={d:.4f} cls={cl:.4f})")

c, d, cl = evaluate(passes_v4v6, weights=[2,2,1,1,1])
print(f"  v4=2 v6=1:                    {c:.4f} (det={d:.4f} cls={cl:.4f})")

c, d, cl = evaluate(passes_v4v6, weights=[1,1,2,2,2])
print(f"  v4=1 v6=2:                    {c:.4f} (det={d:.4f} cls={cl:.4f})")

# ============================================================================
# TEST 4: Best subset with timing consideration (5-6 passes within budget)
# ============================================================================
print("\n" + "="*90)
print("TEST 4: Practical combos (5-6 passes for time budget)")

# v3@1280 + v3@1408 + v4@1280 + v4@1536 + v6@1280 + v6@1536  (6 passes)
def passes_6a(iid):
    return [c3[iid]['full_1280'], c3[iid]['full_1408'],
            c4[iid]['full_1280'], c4[iid]['full_1536'],
            c6[iid]['full_1280'], c6[iid]['full_1536']]
c, d, cl = evaluate(passes_6a, weights=[1,1,2,2,1,1])
print(f"  6-pass v3(12,14)+v4(12,15)+v6(12,15) w=1,1,2,2,1,1:  {c:.4f} (det={d:.4f} cls={cl:.4f})")

# v3@1280 + v4@1280 + v4@1536 + v6@1280 + v6@1536  (5 passes)
def passes_5a(iid):
    return [c3[iid]['full_1280'],
            c4[iid]['full_1280'], c4[iid]['full_1536'],
            c6[iid]['full_1280'], c6[iid]['full_1536']]
c, d, cl = evaluate(passes_5a, weights=[1,2,2,1,1])
print(f"  5-pass v3(12)+v4(12,15)+v6(12,15) w=1,2,2,1,1:       {c:.4f} (det={d:.4f} cls={cl:.4f})")

c, d, cl = evaluate(passes_5a, weights=[1,2,2,2,2])
print(f"  5-pass v3(12)+v4(12,15)+v6(12,15) w=1,2,2,2,2:       {c:.4f} (det={d:.4f} cls={cl:.4f})")

# v3@1280 + v3@1536 + v4@1280 + v6@1280 + v6@1536  (5 passes)
def passes_5b(iid):
    return [c3[iid]['full_1280'], c3[iid]['full_1536'],
            c4[iid]['full_1280'],
            c6[iid]['full_1280'], c6[iid]['full_1536']]
c, d, cl = evaluate(passes_5b, weights=[1,1,2,2,2])
print(f"  5-pass v3(12,15)+v4(12)+v6(12,15) w=1,1,2,2,2:       {c:.4f} (det={d:.4f} cls={cl:.4f})")

# ============================================================================
# TEST 5: WBF iou sweep with 3 models
# ============================================================================
print("\n" + "="*90)
print("TEST 5: WBF iou sweep with 8-pass all models")
for wiou in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
    c, d, cl = evaluate(passes_all8, weights=[1,1,1,2,2,2,2,2], wbf_iou=wiou)
    print(f"  wbf_iou={wiou:.2f} w=[1,1,1,2,2,2,2,2]: {c:.4f} (det={d:.4f} cls={cl:.4f})")

print("\n" + "="*90)
print("DONE")
