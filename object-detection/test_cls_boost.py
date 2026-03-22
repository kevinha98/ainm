"""Test classification improvement strategies using cached detections.
Focus: cls score is 0.8943 vs det 0.9599 — cls is the bottleneck.
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

print(f"Images: {len(image_ids)}")

def _iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax+aw, bx+bw) - max(ax, bx))
    iy = max(0, min(ay+ah, by+bh) - max(ay, by))
    inter = ix * iy
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0

def _iou_xyxy(a, b):
    """IoU for [x1,y1,x2,y2] format."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0

def soft_nms(dets, iou_thresh=0.45, sigma=5.0, score_thresh=0.001):
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
    # Score-weighted category voting
    for ki, k in enumerate(kept):
        if len(absorbed[ki]) > 1:
            cs = {}
            for ab in absorbed[ki]:
                cs[ab["category_id"]] = cs.get(ab["category_id"], 0) + ab["score"]
            kept[ki]["category_id"] = max(cs, key=cs.get)
    return kept

def soft_nms_count_vote(dets, iou_thresh=0.45, sigma=5.0, score_thresh=0.001):
    """Soft-NMS with COUNT-based voting instead of score-based."""
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
    # COUNT-based voting
    for ki, k in enumerate(kept):
        if len(absorbed[ki]) > 1:
            cs = {}
            for ab in absorbed[ki]:
                cs[ab["category_id"]] = cs.get(ab["category_id"], 0) + 1
            kept[ki]["category_id"] = max(cs, key=cs.get)
    return kept

def soft_nms_hybrid_vote(dets, iou_thresh=0.45, sigma=5.0, score_thresh=0.001, count_weight=1.0):
    """Soft-NMS with hybrid voting: count_weight * count + (1-count_weight) * score_sum."""
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
            cat_count = {}
            cat_score = {}
            for ab in absorbed[ki]:
                c = ab["category_id"]
                cat_count[c] = cat_count.get(c, 0) + 1
                cat_score[c] = cat_score.get(c, 0) + ab["score"]
            max_count = max(cat_count.values())
            max_score = max(cat_score.values()) if max(cat_score.values()) > 0 else 1
            hybrid = {}
            for c in cat_count:
                hybrid[c] = count_weight * (cat_count[c] / max_count) + (1-count_weight) * (cat_score[c] / max_score)
            kept[ki]["category_id"] = max(hybrid, key=hybrid.get)
    return kept


# ---------- CLASS-AGNOSTIC WBF ----------
def wbf_class_agnostic(passes, img_w, img_h, iou_thresh=0.55, skip=0.005, ct='box_and_model_avg'):
    """Class-agnostic WBF: fuse all boxes regardless of class, then vote."""
    # Prepare inputs with original labels saved
    all_boxes = []  # per-model list of boxes in normalized [x1,y1,x2,y2]
    all_scores = []
    all_orig_labels = []  # the REAL class labels
    
    bl, sl, ll = [], [], []
    for p in passes:
        b, s, orig = [], [], []
        for d in p:
            x, y, w, h = d["bbox"]
            x1, y1 = max(0, x/img_w), max(0, y/img_h)
            x2, y2 = min(1, (x+w)/img_w), min(1, (y+h)/img_h)
            if x2 <= x1 or y2 <= y1: continue
            b.append([x1,y1,x2,y2])
            s.append(d["score"])
            orig.append(d["category_id"])
        if b:
            bl.append(np.array(b, dtype=np.float32))
            sl.append(np.array(s, dtype=np.float32))
            ll.append(np.zeros(len(b), dtype=np.int32))  # all class=0
            all_boxes.append(b)
            all_scores.append(s)
            all_orig_labels.append(orig)
    
    if not bl: return []
    
    fb, fs, fl = weighted_boxes_fusion(bl, sl, ll, iou_thr=iou_thresh, skip_box_thr=skip, conf_type=ct)
    
    # For each fused box, vote on class using overlapping original detections
    results = []
    for i in range(len(fb)):
        fbox = fb[i]  # [x1, y1, x2, y2] normalized
        cat_scores = {}
        for model_idx in range(len(all_boxes)):
            for j in range(len(all_boxes[model_idx])):
                iou = _iou_xyxy(fbox, all_boxes[model_idx][j])
                if iou > 0.3:
                    orig_cat = all_orig_labels[model_idx][j]
                    cat_scores[orig_cat] = cat_scores.get(orig_cat, 0) + all_scores[model_idx][j]
        
        best_cat = max(cat_scores, key=cat_scores.get) if cat_scores else 0
        results.append({
            "category_id": best_cat,
            "bbox": [round(fbox[0]*img_w,1), round(fbox[1]*img_h,1),
                     round((fbox[2]-fbox[0])*img_w,1), round((fbox[3]-fbox[1])*img_h,1)],
            "score": round(float(fs[i]), 3)
        })
    return results


def wbf_class_agnostic_count(passes, img_w, img_h, iou_thresh=0.55, skip=0.005, ct='box_and_model_avg'):
    """Class-agnostic WBF with COUNT-based voting."""
    all_boxes = []
    all_scores = []
    all_orig_labels = []
    
    bl, sl, ll = [], [], []
    for p in passes:
        b, s, orig = [], [], []
        for d in p:
            x, y, w, h = d["bbox"]
            x1, y1 = max(0, x/img_w), max(0, y/img_h)
            x2, y2 = min(1, (x+w)/img_w), min(1, (y+h)/img_h)
            if x2 <= x1 or y2 <= y1: continue
            b.append([x1,y1,x2,y2])
            s.append(d["score"])
            orig.append(d["category_id"])
        if b:
            bl.append(np.array(b, dtype=np.float32))
            sl.append(np.array(s, dtype=np.float32))
            ll.append(np.zeros(len(b), dtype=np.int32))
            all_boxes.append(b)
            all_scores.append(s)
            all_orig_labels.append(orig)
    
    if not bl: return []
    
    fb, fs, fl = weighted_boxes_fusion(bl, sl, ll, iou_thr=iou_thresh, skip_box_thr=skip, conf_type=ct)
    
    results = []
    for i in range(len(fb)):
        fbox = fb[i]
        cat_counts = {}
        for model_idx in range(len(all_boxes)):
            for j in range(len(all_boxes[model_idx])):
                iou = _iou_xyxy(fbox, all_boxes[model_idx][j])
                if iou > 0.3:
                    orig_cat = all_orig_labels[model_idx][j]
                    cat_counts[orig_cat] = cat_counts.get(orig_cat, 0) + 1
        
        best_cat = max(cat_counts, key=cat_counts.get) if cat_counts else 0
        results.append({
            "category_id": best_cat,
            "bbox": [round(fbox[0]*img_w,1), round(fbox[1]*img_h,1),
                     round((fbox[2]-fbox[0])*img_w,1), round((fbox[3]-fbox[1])*img_h,1)],
            "score": round(float(fs[i]), 3)
        })
    return results


# ---------- STANDARD WBF (class-aware, current approach) ----------
def wbf_standard(passes, img_w, img_h, iou_thresh=0.55, skip=0.005, ct='box_and_model_avg'):
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


# ---------- POST-WBF CLASS VOTING (keep WBF class-aware, then re-vote) ----------
def wbf_with_post_vote(passes, img_w, img_h, iou_thresh=0.55, skip=0.005, ct='box_and_model_avg', vote_iou=0.5):
    """Standard class-aware WBF, then re-assign classes by voting from original detections."""
    # First run standard WBF
    fused = wbf_standard(passes, img_w, img_h, iou_thresh, skip, ct)
    
    # Flatten all original detections
    all_dets = []
    for p in passes:
        for d in p:
            all_dets.append(d)
    
    # For each fused box, vote on class
    for f in fused:
        cat_scores = {}
        for d in all_dets:
            iou = _iou(f["bbox"], d["bbox"])
            if iou > vote_iou:
                c = d["category_id"]
                cat_scores[c] = cat_scores.get(c, 0) + d["score"]
        if cat_scores:
            f["category_id"] = max(cat_scores, key=cat_scores.get)
    
    return fused


# ---------- EVALUATION ----------
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


def evaluate_pipeline(wbf_fn, snms_fn, wbf_iou=0.55, snms_sigma=5.0, snms_thresh=0.001, max_dets=300):
    det_aps, cls_aps = [], []
    for iid in image_ids:
        info = img_info.get(iid)
        if not info: continue
        img_w, img_h = info["width"], info["height"]
        
        passes = [c3[iid]['full_1280'], c3[iid]['full_1408'], c3[iid]['full_1536'],
                  c4[iid]['full_1280'], c4[iid]['full_1536']]
        
        fused = wbf_fn(passes, img_w, img_h, iou_thresh=wbf_iou)
        dets = snms_fn(fused, sigma=snms_sigma, score_thresh=snms_thresh)
        
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


# ==================== RUN TESTS ====================
print("\n" + "="*90)
print("TEST 1: BASELINE (class-aware WBF + score-weighted soft-NMS vote)")
c, d, cl = evaluate_pipeline(wbf_standard, soft_nms)
print(f"  Combined: {c:.4f} (det={d:.4f} cls={cl:.4f})")

print("\n" + "="*90)
print("TEST 2: Class-aware WBF + COUNT-based soft-NMS vote")
c, d, cl = evaluate_pipeline(wbf_standard, soft_nms_count_vote)
print(f"  Combined: {c:.4f} (det={d:.4f} cls={cl:.4f})")

print("\n" + "="*90)
print("TEST 3: Class-agnostic WBF (score vote) + score-weighted soft-NMS vote")
c, d, cl = evaluate_pipeline(wbf_class_agnostic, soft_nms)
print(f"  Combined: {c:.4f} (det={d:.4f} cls={cl:.4f})")

print("\n" + "="*90)
print("TEST 4: Class-agnostic WBF (count vote) + score-weighted soft-NMS vote")
c, d, cl = evaluate_pipeline(wbf_class_agnostic_count, soft_nms)
print(f"  Combined: {c:.4f} (det={d:.4f} cls={cl:.4f})")

print("\n" + "="*90)
print("TEST 5: Post-WBF vote (class-aware WBF, then re-vote from raw dets)")
for vote_iou in [0.3, 0.4, 0.5, 0.6]:
    def make_wbf(viou):
        def fn(passes, img_w, img_h, iou_thresh=0.55):
            return wbf_with_post_vote(passes, img_w, img_h, iou_thresh, vote_iou=viou)
        return fn
    c, d, cl = evaluate_pipeline(make_wbf(vote_iou), soft_nms)
    print(f"  vote_iou={vote_iou:.1f}: Combined={c:.4f} (det={d:.4f} cls={cl:.4f})")

print("\n" + "="*90)
print("TEST 6: Hybrid voting (soft-NMS) — sweep count_weight")
for cw in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    def make_snms(w):
        def fn(dets, sigma=5.0, score_thresh=0.001):
            return soft_nms_hybrid_vote(dets, sigma=sigma, score_thresh=score_thresh, count_weight=w)
        return fn
    c, d, cl = evaluate_pipeline(wbf_standard, make_snms(cw))
    print(f"  count_weight={cw:.1f}: Combined={c:.4f} (det={d:.4f} cls={cl:.4f})")

print("\n" + "="*90)
print("TEST 7: Class-agnostic WBF + COUNT vote + COUNT soft-NMS vote")
c, d, cl = evaluate_pipeline(wbf_class_agnostic_count, soft_nms_count_vote)
print(f"  Combined: {c:.4f} (det={d:.4f} cls={cl:.4f})")

# Test varying the overlap threshold for class-agnostic WBF voting
print("\n" + "="*90)
print("TEST 8: Class-agnostic WBF — sweep vote overlap threshold")
for vt in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    def make_ca_wbf(threshold):
        def fn(passes, img_w, img_h, iou_thresh=0.55, skip=0.005, ct='box_and_model_avg'):
            all_boxes, all_scores, all_orig_labels = [], [], []
            bl, sl, ll = [], [], []
            for p in passes:
                b, s, orig = [], [], []
                for d in p:
                    x, y, w, h = d["bbox"]
                    x1, y1 = max(0, x/img_w), max(0, y/img_h)
                    x2, y2 = min(1, (x+w)/img_w), min(1, (y+h)/img_h)
                    if x2 <= x1 or y2 <= y1: continue
                    b.append([x1,y1,x2,y2]); s.append(d["score"]); orig.append(d["category_id"])
                if b:
                    bl.append(np.array(b, dtype=np.float32))
                    sl.append(np.array(s, dtype=np.float32))
                    ll.append(np.zeros(len(b), dtype=np.int32))
                    all_boxes.append(b); all_scores.append(s); all_orig_labels.append(orig)
            if not bl: return []
            fb, fs, fl = weighted_boxes_fusion(bl, sl, ll, iou_thr=iou_thresh, skip_box_thr=skip, conf_type=ct)
            results = []
            for i in range(len(fb)):
                fbox = fb[i]
                cat_scores = {}
                for mi in range(len(all_boxes)):
                    for j in range(len(all_boxes[mi])):
                        iou = _iou_xyxy(fbox, all_boxes[mi][j])
                        if iou > threshold:
                            c = all_orig_labels[mi][j]
                            cat_scores[c] = cat_scores.get(c, 0) + all_scores[mi][j]
                best_cat = max(cat_scores, key=cat_scores.get) if cat_scores else 0
                results.append({"category_id": best_cat,
                    "bbox": [round(fbox[0]*img_w,1), round(fbox[1]*img_h,1),
                             round((fbox[2]-fbox[0])*img_w,1), round((fbox[3]-fbox[1])*img_h,1)],
                    "score": round(float(fs[i]), 3)})
            return results
        return fn
    c, d, cl = evaluate_pipeline(make_ca_wbf(vt), soft_nms)
    print(f"  vote_thresh={vt:.1f}: Combined={c:.4f} (det={d:.4f} cls={cl:.4f})")

print("\n" + "="*90)
print("DONE!")
