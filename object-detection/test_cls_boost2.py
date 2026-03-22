"""Test classification improvement strategies using cached detections.
Focus: cls score is 0.8943 vs det 0.9599 — cls is the bottleneck.
OPTIMIZED version with vectorized IoU.
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

def _iou_matrix_xyxy(a, b):
    """Vectorized IoU between two sets of [x1,y1,x2,y2] boxes. Returns (len(a), len(b)) matrix."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    
    ix1 = np.maximum(a[:, 0:1], b[:, 0:1].T)
    iy1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
    ix2 = np.minimum(a[:, 2:3], b[:, 2:3].T)
    iy2 = np.minimum(a[:, 3:4], b[:, 3:4].T)
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0)


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
    """Soft-NMS with COUNT-based voting."""
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
                cs[ab["category_id"]] = cs.get(ab["category_id"], 0) + 1
            kept[ki]["category_id"] = max(cs, key=cs.get)
    return kept

def soft_nms_no_vote(dets, iou_thresh=0.45, sigma=5.0, score_thresh=0.001):
    """Soft-NMS WITHOUT any category voting — keep original WBF class."""
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
            d["score"] *= np.exp(-(iou**2) / sigma)
            if d["score"] >= score_thresh:
                remaining.append(d)
        dets = sorted(remaining, key=lambda x: x["score"], reverse=True)
    return kept


# ---------- CLASS-AGNOSTIC WBF with vectorized IoU ----------
def wbf_class_agnostic(passes, img_w, img_h, iou_thresh=0.55, skip=0.005, ct='box_and_model_avg', vote_thresh=0.3, vote_mode='score'):
    """Class-agnostic WBF: fuse all boxes regardless of class, then vote."""
    all_boxes_norm = []  # list of np arrays [N, 4] in xyxy normalized
    all_scores_flat = []
    all_labels_flat = []
    
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
            b_arr = np.array(b, dtype=np.float32)
            bl.append(b_arr)
            sl.append(np.array(s, dtype=np.float32))
            ll.append(np.zeros(len(b), dtype=np.int32))  # all class=0
            all_boxes_norm.append(b_arr)
            all_scores_flat.extend(s)
            all_labels_flat.extend(orig)
    
    if not bl: return []
    
    fb, fs, fl = weighted_boxes_fusion(bl, sl, ll, iou_thr=iou_thresh, skip_box_thr=skip, conf_type=ct)
    
    if len(fb) == 0: return []
    
    # Stack all original boxes and scores for vectorized IoU
    orig_boxes = np.vstack(all_boxes_norm)  # [M, 4]
    orig_scores = np.array(all_scores_flat, dtype=np.float32)  # [M]
    orig_labels = np.array(all_labels_flat, dtype=np.int32)   # [M]
    
    # Vectorized IoU: fused [N,4] vs original [M,4] => [N, M]
    iou_mat = _iou_matrix_xyxy(fb, orig_boxes)
    
    results = []
    for i in range(len(fb)):
        mask = iou_mat[i] > vote_thresh
        if mask.any():
            matched_labels = orig_labels[mask]
            matched_scores = orig_scores[mask]
            cat_agg = {}
            if vote_mode == 'score':
                for lbl, sc in zip(matched_labels, matched_scores):
                    cat_agg[lbl] = cat_agg.get(lbl, 0) + sc
            else:  # count
                for lbl in matched_labels:
                    cat_agg[lbl] = cat_agg.get(lbl, 0) + 1
            best_cat = max(cat_agg, key=cat_agg.get)
        else:
            best_cat = 0
        
        fbox = fb[i]
        results.append({
            "category_id": int(best_cat),
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


# ---------- POST-WBF CLASS VOTING ----------
def wbf_with_post_vote(passes, img_w, img_h, iou_thresh=0.55, skip=0.005, ct='box_and_model_avg', vote_iou=0.5, vote_mode='score'):
    """Standard class-aware WBF, then re-assign classes by voting from original detections."""
    fused = wbf_standard(passes, img_w, img_h, iou_thresh, skip, ct)
    if not fused: return fused
    
    # Collect all original detections with normalized boxes
    orig_boxes_list = []
    orig_scores_list = []
    orig_labels_list = []
    for p in passes:
        for d in p:
            x, y, w, h = d["bbox"]
            x1, y1 = x/img_w, y/img_h
            x2, y2 = (x+w)/img_w, (y+h)/img_h
            orig_boxes_list.append([x1, y1, x2, y2])
            orig_scores_list.append(d["score"])
            orig_labels_list.append(d["category_id"])
    
    orig_boxes = np.array(orig_boxes_list, dtype=np.float32)
    orig_scores = np.array(orig_scores_list, dtype=np.float32)
    orig_labels = np.array(orig_labels_list, dtype=np.int32)
    
    # Fused boxes in normalized xyxy
    fused_boxes = np.array([[d["bbox"][0]/img_w, d["bbox"][1]/img_h,
                             (d["bbox"][0]+d["bbox"][2])/img_w, (d["bbox"][1]+d["bbox"][3])/img_h]
                            for d in fused], dtype=np.float32)
    
    iou_mat = _iou_matrix_xyxy(fused_boxes, orig_boxes)
    
    for i, f in enumerate(fused):
        mask = iou_mat[i] > vote_iou
        if mask.any():
            matched_labels = orig_labels[mask]
            matched_scores = orig_scores[mask]
            cat_agg = {}
            if vote_mode == 'score':
                for lbl, sc in zip(matched_labels, matched_scores):
                    cat_agg[lbl] = cat_agg.get(lbl, 0) + sc
            else:
                for lbl in matched_labels:
                    cat_agg[lbl] = cat_agg.get(lbl, 0) + 1
            f["category_id"] = int(max(cat_agg, key=cat_agg.get))
    
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
t0 = time.time()

print("\n" + "="*80)
print("TEST 1: BASELINE (class-aware WBF + score-weighted soft-NMS vote)")
c, d, cl = evaluate_pipeline(wbf_standard, soft_nms)
print(f"  Combined: {c:.4f} (det={d:.4f} cls={cl:.4f})  [{time.time()-t0:.1f}s]")

print("\n" + "="*80)
print("TEST 2: Class-aware WBF + COUNT-based soft-NMS vote")
c, d, cl = evaluate_pipeline(wbf_standard, soft_nms_count_vote)
print(f"  Combined: {c:.4f} (det={d:.4f} cls={cl:.4f})  [{time.time()-t0:.1f}s]")

print("\n" + "="*80)
print("TEST 3: Class-aware WBF + NO soft-NMS vote (keep WBF class)")
c, d, cl = evaluate_pipeline(wbf_standard, soft_nms_no_vote)
print(f"  Combined: {c:.4f} (det={d:.4f} cls={cl:.4f})  [{time.time()-t0:.1f}s]")

print("\n" + "="*80)
print("TEST 4: Class-AGNOSTIC WBF (score vote, thresh=0.3) + score soft-NMS vote")
def ca_score_03(passes, img_w, img_h, iou_thresh=0.55):
    return wbf_class_agnostic(passes, img_w, img_h, iou_thresh, vote_thresh=0.3, vote_mode='score')
c, d, cl = evaluate_pipeline(ca_score_03, soft_nms)
print(f"  Combined: {c:.4f} (det={d:.4f} cls={cl:.4f})  [{time.time()-t0:.1f}s]")

print("\n" + "="*80)
print("TEST 5: Class-AGNOSTIC WBF (count vote, thresh=0.3) + score soft-NMS vote")
def ca_count_03(passes, img_w, img_h, iou_thresh=0.55):
    return wbf_class_agnostic(passes, img_w, img_h, iou_thresh, vote_thresh=0.3, vote_mode='count')
c, d, cl = evaluate_pipeline(ca_count_03, soft_nms)
print(f"  Combined: {c:.4f} (det={d:.4f} cls={cl:.4f})  [{time.time()-t0:.1f}s]")

print("\n" + "="*80)
print("TEST 6: Post-WBF re-vote (class-aware WBF, then re-assign from raw dets)")
for vote_iou in [0.3, 0.4, 0.5, 0.6]:
    for vm in ['score', 'count']:
        def make_wbf(vi, mode):
            def fn(passes, img_w, img_h, iou_thresh=0.55):
                return wbf_with_post_vote(passes, img_w, img_h, iou_thresh, vote_iou=vi, vote_mode=mode)
            return fn
        c, d, cl = evaluate_pipeline(make_wbf(vote_iou, vm), soft_nms)
        print(f"  vote_iou={vote_iou:.1f} mode={vm:5s}: {c:.4f} (det={d:.4f} cls={cl:.4f})")

print("\n" + "="*80)
print("TEST 7: Class-AGNOSTIC WBF — sweep vote overlap threshold")
for vt in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for vm in ['score', 'count']:
        def make_ca(threshold, mode):
            def fn(passes, img_w, img_h, iou_thresh=0.55):
                return wbf_class_agnostic(passes, img_w, img_h, iou_thresh, vote_thresh=threshold, vote_mode=mode)
            return fn
        c, d, cl = evaluate_pipeline(make_ca(vt, vm), soft_nms)
        print(f"  thresh={vt:.1f} mode={vm:5s}: {c:.4f} (det={d:.4f} cls={cl:.4f})")

print("\n" + "="*80)
print("TEST 8: Class-AGNOSTIC WBF + NO soft-NMS vote")
def ca_score_03_novote(passes, img_w, img_h, iou_thresh=0.55):
    return wbf_class_agnostic(passes, img_w, img_h, iou_thresh, vote_thresh=0.3, vote_mode='score')
c, d, cl = evaluate_pipeline(ca_score_03_novote, soft_nms_no_vote)
print(f"  Combined: {c:.4f} (det={d:.4f} cls={cl:.4f})  [{time.time()-t0:.1f}s]")

# Test with lower WBF iou for class-agnostic (might be different sweet spot)
print("\n" + "="*80)
print("TEST 9: Class-AGNOSTIC WBF — sweep WBF iou threshold")
for wiou in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
    def make_ca_wiou(wi):
        def fn(passes, img_w, img_h, iou_thresh=0.55):
            return wbf_class_agnostic(passes, img_w, img_h, iou_thresh=wi, vote_thresh=0.3, vote_mode='score')
        return fn
    c, d, cl = evaluate_pipeline(make_ca_wiou(wiou), soft_nms)
    print(f"  wbf_iou={wiou:.2f}: {c:.4f} (det={d:.4f} cls={cl:.4f})")

print(f"\nTotal time: {time.time()-t0:.1f}s")
print("DONE!")
