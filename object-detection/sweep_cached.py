"""Fast parameter sweep using cached detections — NO inference needed.
Tests WBF iou_thresh, skip_box_thresh, conf_type, soft-NMS sigma, score_thresh.
Also tests different pass subsets from cached data.
"""
import pickle, json, time
import numpy as np
from ensemble_boxes import weighted_boxes_fusion

# Load caches
print("Loading caches...")
c3 = pickle.load(open('cache_v3_29.pkl', 'rb'))
c4 = pickle.load(open('cache_v4_29.pkl', 'rb'))
image_ids = sorted(c3.keys())

# Load GT
gt_data = json.load(open('data/coco/train/annotations.json'))
gt_by_image = {}
for ann in gt_data["annotations"]:
    gt_by_image.setdefault(ann["image_id"], []).append(ann)
img_info = {img["id"]: img for img in gt_data["images"]}

print(f"Images: {len(image_ids)}, GT images with annotations: {len(gt_by_image)}")

def _iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax+aw, bx+bw) - max(ax, bx))
    iy = max(0, min(ay+ah, by+bh) - max(ay, by))
    inter = ix * iy
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0

def soft_nms(dets, iou_thresh=0.45, sigma=5.0, score_thresh=0.01):
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
            if iou >= iou_thresh:
                cur_abs.append(d.copy())
            d["score"] *= np.exp(-(iou**2) / sigma)
            if d["score"] >= score_thresh:
                remaining.append(d)
        absorbed.append(cur_abs)
        dets = sorted(remaining, key=lambda x: x["score"], reverse=True)
    for ki, k in enumerate(kept):
        if len(absorbed[ki]) > 1:
            cs = {}
            for ab in absorbed[ki]:
                cs[ab["category_id"]] = cs.get(ab["category_id"], 0) + ab["score"]
            kept[ki]["category_id"] = max(cs, key=cs.get)
    return kept

def wbf(passes, img_w, img_h, iou_thresh=0.35, skip=0.005, ct='box_and_model_avg'):
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
            bl.append(np.array(b, dtype=np.float32))
            sl.append(np.array(s, dtype=np.float32))
            ll.append(np.array(l, dtype=np.int32))
    if not bl: return []
    fb, fs, fl = weighted_boxes_fusion(bl, sl, ll, iou_thr=iou_thresh, skip_box_thr=skip, conf_type=ct)
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
            if check_class and p["category_id"] != g["category_id"]:
                continue
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
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1])

def evaluate(passes_fn, wbf_iou=0.35, wbf_skip=0.005, wbf_ct='box_and_model_avg',
             snms_iou=0.45, snms_sigma=5.0, snms_thresh=0.01, max_dets=300):
    det_aps, cls_aps = [], []
    for iid in image_ids:
        info = img_info.get(iid)
        if not info: continue
        img_w, img_h = info["width"], info["height"]
        passes = passes_fn(iid)
        
        if len(passes) > 1:
            fused = wbf(passes, img_w, img_h, iou_thresh=wbf_iou, skip=wbf_skip, ct=wbf_ct)
        else:
            fused = passes[0] if passes else []
        
        dets = soft_nms(fused, iou_thresh=snms_iou, sigma=snms_sigma, score_thresh=snms_thresh)
        if len(dets) > max_dets:
            dets.sort(key=lambda x: x["score"], reverse=True)
            dets = dets[:max_dets]
        
        gts = [{"bbox": g["bbox"], "category_id": g["category_id"]} for g in gt_by_image.get(iid, [])]
        det_p = [{"bbox": d["bbox"], "score": d["score"], "category_id": 0} for d in dets]
        det_g = [{"bbox": g["bbox"], "category_id": 0} for g in gts]
        det_aps.append(compute_ap(det_p, det_g))
        cls_aps.append(compute_ap(dets, gts, check_class=True))
    
    det = np.mean(det_aps); cls = np.mean(cls_aps)
    return 0.7*det + 0.3*cls, det, cls

# Current best: 5-pass
def passes_5(iid):
    return [c3[iid]['full_1280'], c3[iid]['full_1408'], c3[iid]['full_1536'],
            c4[iid]['full_1280'], c4[iid]['full_1536']]

print("\n" + "="*90)
print("BASELINE (current submitted config):")
comb, det, cls = evaluate(passes_5, wbf_iou=0.35, wbf_skip=0.005, wbf_ct='box_and_model_avg',
                           snms_iou=0.45, snms_sigma=5.0, snms_thresh=0.01)
print(f"  5-pass, WBF=0.35, skip=0.005, ct=bma, sigma=5.0 -> {comb:.4f} (det={det:.4f} cls={cls:.4f})")

print("\n" + "="*90)
print("SWEEP 1: WBF iou_thresh")
for wiou in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
    c, d, cl = evaluate(passes_5, wbf_iou=wiou)
    print(f"  wbf_iou={wiou:.2f} -> {c:.4f} (det={d:.4f} cls={cl:.4f})")

print("\n" + "="*90)
print("SWEEP 2: WBF skip_box_thresh")
for skip in [0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.02, 0.03, 0.05]:
    c, d, cl = evaluate(passes_5, wbf_skip=skip)
    print(f"  skip={skip:.3f} -> {c:.4f} (det={d:.4f} cls={cl:.4f})")

print("\n" + "="*90)
print("SWEEP 3: Soft-NMS sigma")
for sig in [0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 50.0]:
    c, d, cl = evaluate(passes_5, snms_sigma=sig)
    print(f"  sigma={sig:5.1f} -> {c:.4f} (det={d:.4f} cls={cl:.4f})")

print("\n" + "="*90)
print("SWEEP 4: Soft-NMS iou_thresh")
for siou in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
    c, d, cl = evaluate(passes_5, snms_iou=siou)
    print(f"  snms_iou={siou:.2f} -> {c:.4f} (det={d:.4f} cls={cl:.4f})")

print("\n" + "="*90)
print("SWEEP 5: Soft-NMS score_thresh")
for st in [0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08]:
    c, d, cl = evaluate(passes_5, snms_thresh=st)
    print(f"  score_thresh={st:.3f} -> {c:.4f} (det={d:.4f} cls={cl:.4f})")

print("\n" + "="*90)
print("SWEEP 6: conf_type")
for ct in ['avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg']:
    c, d, cl = evaluate(passes_5, wbf_ct=ct)
    print(f"  conf_type={ct:30s} -> {c:.4f} (det={d:.4f} cls={cl:.4f})")

print("\n" + "="*90)
print("SWEEP 7: max_dets per image")
for md in [100, 150, 200, 250, 300, 400, 500, 700, 1000]:
    c, d, cl = evaluate(passes_5, max_dets=md)
    print(f"  max_dets={md:4d} -> {c:.4f} (det={d:.4f} cls={cl:.4f})")

print("\n" + "="*90)
print("SWEEP 8: Pass subsets")
configs = {
    "v3_1280_only": lambda iid: [c3[iid]['full_1280']],
    "v3_3pass": lambda iid: [c3[iid]['full_1280'], c3[iid]['full_1408'], c3[iid]['full_1536']],
    "v4_2pass": lambda iid: [c4[iid]['full_1280'], c4[iid]['full_1536']],
    "5pass (current)": passes_5,
    "v3_3+v4_1280": lambda iid: [c3[iid]['full_1280'], c3[iid]['full_1408'], c3[iid]['full_1536'], c4[iid]['full_1280']],
    "v3_1280_1536+v4": lambda iid: [c3[iid]['full_1280'], c3[iid]['full_1536'], c4[iid]['full_1280'], c4[iid]['full_1536']],
}
for name, fn in configs.items():
    c, d, cl = evaluate(fn)
    print(f"  {name:25s} -> {c:.4f} (det={d:.4f} cls={cl:.4f})")

# Fine-grained sweep around the best params
print("\n" + "="*90)
print("SWEEP 9: Fine-grained combo (WBF iou x sigma x skip x score_thresh)")
best_score = 0
best_params = {}
count = 0
for wiou in [0.25, 0.30, 0.35, 0.40, 0.45]:
    for sig in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
        for skip in [0.002, 0.005, 0.008, 0.01]:
            for st in [0.005, 0.01, 0.02]:
                for siou in [0.40, 0.45, 0.50]:
                    count += 1
                    c, d, cl = evaluate(passes_5, wbf_iou=wiou, snms_sigma=sig,
                                        wbf_skip=skip, snms_thresh=st, snms_iou=siou)
                    if c > best_score:
                        best_score = c
                        best_params = dict(wbf_iou=wiou, sigma=sig, skip=skip,
                                          score_thresh=st, snms_iou=siou)
                        print(f"  NEW BEST: {c:.4f} (det={d:.4f} cls={cl:.4f}) | "
                              f"wiou={wiou} sig={sig} skip={skip} st={st} siou={siou}")

print(f"\nTested {count} configs in sweep 9")
print(f"\n{'='*90}")
print(f"OVERALL BEST CONFIG: {best_score:.4f}")
print(f"  Params: {best_params}")
print("Done!")
