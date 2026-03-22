"""Fine-tune the winners from cls_boost3:
1. WBF weights v3=1, v4=2 (cls +0.0033)
2. No soft-NMS vote (cls +0.0013)
3. Combine them + sweep finer weight ratios
"""
import pickle, json, time
import numpy as np
from ensemble_boxes import weighted_boxes_fusion

print("Loading caches...")
c3 = pickle.load(open('cache_v3_29.pkl', 'rb'))
c4 = pickle.load(open('cache_v4_29.pkl', 'rb'))
image_ids = sorted(c3.keys())
gt_data = json.load(open('data/coco/train/annotations.json'))
gt_by_image = {}
for ann in gt_data["annotations"]:
    gt_by_image.setdefault(ann["image_id"], []).append(ann)
img_info = {img["id"]: img for img in gt_data["images"]}

def _iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax+aw, bx+bw) - max(ax, bx))
    iy = max(0, min(ay+ah, by+bh) - max(ay, by))
    inter = ix * iy
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0

def soft_nms_vote(dets, iou_thresh=0.45, sigma=5.0, score_thresh=0.001):
    if not dets: return dets
    dets = [d.copy() for d in dets]
    dets.sort(key=lambda x: x["score"], reverse=True)
    kept, absorbed = [], []
    while dets:
        best = dets.pop(0)
        kept.append(best)
        cur_abs = [best.copy()]
        remaining = []
        for d in dets:
            iou = _iou(best["bbox"], d["bbox"])
            if iou >= iou_thresh: cur_abs.append(d.copy())
            d["score"] *= np.exp(-(iou**2)/sigma)
            if d["score"] >= score_thresh: remaining.append(d)
        absorbed.append(cur_abs)
        dets = sorted(remaining, key=lambda x: x["score"], reverse=True)
    for ki, k in enumerate(kept):
        if len(absorbed[ki]) > 1:
            cs = {}
            for ab in absorbed[ki]: cs[ab["category_id"]] = cs.get(ab["category_id"],0) + ab["score"]
            kept[ki]["category_id"] = max(cs, key=cs.get)
    return kept

def soft_nms_no_vote(dets, iou_thresh=0.45, sigma=5.0, score_thresh=0.001):
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

def evaluate(wbf_fn, snms_fn, max_dets=300):
    det_aps, cls_aps = [], []
    for iid in image_ids:
        info = img_info.get(iid)
        if not info: continue
        img_w, img_h = info["width"], info["height"]
        passes = [c3[iid]['full_1280'], c3[iid]['full_1408'], c3[iid]['full_1536'],
                  c4[iid]['full_1280'], c4[iid]['full_1536']]
        fused = wbf_fn(passes, img_w, img_h)
        dets = snms_fn(fused)
        if len(dets) > max_dets:
            dets.sort(key=lambda x: x["score"], reverse=True)
            dets = dets[:max_dets]
        gts = [{"bbox": g["bbox"], "category_id": g["category_id"]} for g in gt_by_image.get(iid, [])]
        det_p = [{"bbox": d["bbox"], "score": d["score"], "category_id": 0} for d in dets]
        det_g = [{"bbox": g["bbox"], "category_id": 0} for g in gts]
        det_aps.append(compute_ap(det_p, det_g))
        cls_aps.append(compute_ap(dets, gts, check_class=True))
    det = np.mean(det_aps); cls = np.mean(cls_aps)
    return 0.7*det+0.3*cls, det, cls

t0 = time.time()

print("="*80)
print("FINE-GRAINED WEIGHT SWEEP (with score vote)")
# 5 passes: [v3@1280, v3@1408, v3@1536, v4@1280, v4@1536]
best = (0, None, None)
for w_v4 in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
    for w_v3 in [0.5, 1.0, 1.5, 2.0]:
        wts = [w_v3, w_v3, w_v3, w_v4, w_v4]
        def make_wbf(w):
            def fn(p, iw, ih): return wbf(p, iw, ih, weights=w)
            return fn
        c, d, cl = evaluate(make_wbf(wts), soft_nms_vote)
        marker = " ***" if c > best[0] else ""
        print(f"  v3={w_v3:.1f} v4={w_v4:.1f}: {c:.4f} (det={d:.4f} cls={cl:.4f}){marker}")
        if c > best[0]: best = (c, wts, (d, cl))

print(f"\n  BEST: {best[0]:.4f} weights={best[1]} det={best[2][0]:.4f} cls={best[2][1]:.4f}")

print("\n" + "="*80)
print("FINE-GRAINED WEIGHT SWEEP (NO vote)")
best2 = (0, None, None)
for w_v4 in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
    for w_v3 in [0.5, 1.0, 1.5, 2.0]:
        wts = [w_v3, w_v3, w_v3, w_v4, w_v4]
        def make_wbf(w):
            def fn(p, iw, ih): return wbf(p, iw, ih, weights=w)
            return fn
        c, d, cl = evaluate(make_wbf(wts), soft_nms_no_vote)
        marker = " ***" if c > best2[0] else ""
        print(f"  v3={w_v3:.1f} v4={w_v4:.1f}: {c:.4f} (det={d:.4f} cls={cl:.4f}){marker}")
        if c > best2[0]: best2 = (c, wts, (d, cl))

print(f"\n  BEST: {best2[0]:.4f} weights={best2[1]} det={best2[2][0]:.4f} cls={best2[2][1]:.4f}")

# Also try per-scale weights: maybe v3@1280 has different weight than v3@1408
print("\n" + "="*80)
print("PER-SCALE WEIGHTS (v4=2, varying v3 per scale)")
for w_1280 in [0.5, 1.0, 1.5]:
    for w_1408 in [0.5, 1.0, 1.5]:
        for w_1536 in [0.5, 1.0, 1.5]:
            wts = [w_1280, w_1408, w_1536, 2.0, 2.0]
            def make_wbf(w):
                def fn(p, iw, ih): return wbf(p, iw, ih, weights=w)
                return fn
            c, d, cl = evaluate(make_wbf(wts), soft_nms_vote)
            if c >= 0.9440:
                print(f"  v3_1280={w_1280} v3_1408={w_1408} v3_1536={w_1536} v4=2: {c:.4f} (det={d:.4f} cls={cl:.4f})")

# Combined best: weights + WBF iou sweep
print("\n" + "="*80)
print("BEST WEIGHTS + WBF IOU SWEEP")
for wiou in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
    def make_wbf_iou(wi):
        def fn(p, iw, ih): return wbf(p, iw, ih, iou_thresh=wi, weights=[1,1,1,2,2])
        return fn
    c, d, cl = evaluate(make_wbf_iou(wiou), soft_nms_vote)
    print(f"  wiou={wiou:.2f} w=[1,1,1,2,2] vote: {c:.4f} (det={d:.4f} cls={cl:.4f})")
    c2, d2, cl2 = evaluate(make_wbf_iou(wiou), soft_nms_no_vote)
    print(f"  wiou={wiou:.2f} w=[1,1,1,2,2] novt: {c2:.4f} (det={d2:.4f} cls={cl2:.4f})")

print(f"\nTotal: {time.time()-t0:.1f}s")
print("DONE!")
