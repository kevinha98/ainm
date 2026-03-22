"""Fast classification improvement tests — NO class-agnostic WBF (too slow).
Focus on: post-WBF re-vote, WBF model weights, voting strategies.
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
print(f"Images: {len(image_ids)}")

def _iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax+aw, bx+bw) - max(ax, bx))
    iy = max(0, min(ay+ah, by+bh) - max(ay, by))
    inter = ix * iy
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0

def _iou_matrix_norm(a, b):
    """Vectorized IoU for normalized [x1,y1,x2,y2]."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if len(a) == 0 or len(b) == 0: return np.zeros((len(a), len(b)), np.float32)
    ix1 = np.maximum(a[:,0:1], b[:,0:1].T)
    iy1 = np.maximum(a[:,1:2], b[:,1:2].T)
    ix2 = np.minimum(a[:,2:3], b[:,2:3].T)
    iy2 = np.minimum(a[:,3:4], b[:,3:4].T)
    inter = np.maximum(0, ix2-ix1) * np.maximum(0, iy2-iy1)
    aa = (a[:,2]-a[:,0])*(a[:,3]-a[:,1])
    ab = (b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    union = aa[:,None]+ab[None,:]-inter
    return np.where(union>0, inter/union, 0)


# ===================== SOFT-NMS VARIANTS =====================

def soft_nms_score_vote(dets, iou_thresh=0.45, sigma=5.0, score_thresh=0.001):
    """Current: score-weighted category voting."""
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
    """No voting — keep original class from WBF."""
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

def soft_nms_top_conf_vote(dets, iou_thresh=0.45, sigma=5.0, score_thresh=0.001):
    """Vote using the single highest-confidence absorbed detection's class."""
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
            # Use class from the single highest-confidence detection
            best_ab = max(absorbed[ki], key=lambda x: x["score"])
            kept[ki]["category_id"] = best_ab["category_id"]
    return kept


# ===================== WBF VARIANTS =====================

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


def wbf_post_vote(passes, img_w, img_h, iou_thresh=0.55, skip=0.005, ct='box_and_model_avg',
                  vote_iou=0.5, vote_mode='score'):
    """Class-aware WBF, then re-assign classes by voting from all raw detections."""
    fused = wbf(passes, img_w, img_h, iou_thresh, skip, ct)
    if not fused: return fused
    
    # Collect original detections with normalized boxes
    orig_b, orig_s, orig_l = [], [], []
    for p in passes:
        for d in p:
            x, y, w, h = d["bbox"]
            orig_b.append([x/img_w, y/img_h, (x+w)/img_w, (y+h)/img_h])
            orig_s.append(d["score"])
            orig_l.append(d["category_id"])
    orig_b = np.array(orig_b, np.float32)
    orig_s = np.array(orig_s, np.float32)
    orig_l = np.array(orig_l, np.int32)
    
    fbox = np.array([[d["bbox"][0]/img_w, d["bbox"][1]/img_h,
                      (d["bbox"][0]+d["bbox"][2])/img_w, (d["bbox"][1]+d["bbox"][3])/img_h]
                     for d in fused], np.float32)
    
    iou_mat = _iou_matrix_norm(fbox, orig_b)
    
    for i, f in enumerate(fused):
        mask = iou_mat[i] > vote_iou
        if mask.any():
            ml = orig_l[mask]; ms = orig_s[mask]
            cat = {}
            if vote_mode == 'score':
                for lbl, sc in zip(ml, ms): cat[lbl] = cat.get(lbl,0) + sc
            else:
                for lbl in ml: cat[lbl] = cat.get(lbl,0) + 1
            f["category_id"] = int(max(cat, key=cat.get))
    return fused


# ===================== EVALUATION =====================

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


# ==================== RUN ====================
t0 = time.time()

# Standard WBF wrapper
def wbf_std(passes, img_w, img_h):
    return wbf(passes, img_w, img_h)

# Standard soft-NMS wrapper
def snms_std(dets):
    return soft_nms_score_vote(dets)

print("\n" + "="*80)
print("TEST 1: BASELINE (class-aware WBF + score vote)")
c, d, cl = evaluate(wbf_std, snms_std)
print(f"  {c:.4f} (det={d:.4f} cls={cl:.4f})  [{time.time()-t0:.1f}s]")

print("\n" + "="*80)
print("TEST 2: No soft-NMS vote (keep WBF class)")
c, d, cl = evaluate(wbf_std, soft_nms_no_vote)
print(f"  {c:.4f} (det={d:.4f} cls={cl:.4f})  [{time.time()-t0:.1f}s]")

print("\n" + "="*80)
print("TEST 3: Top-confidence vote (use highest-conf absorbed detection's class)")
c, d, cl = evaluate(wbf_std, soft_nms_top_conf_vote)
print(f"  {c:.4f} (det={d:.4f} cls={cl:.4f})  [{time.time()-t0:.1f}s]")

print("\n" + "="*80)
print("TEST 4: WBF model weights (v3 more weight since better model)")
# 5 passes: v3@1280, v3@1408, v3@1536, v4@1280, v4@1536
# Currently all weight=1
for w_v3, w_v4 in [(1,1), (2,1), (3,1), (1,2), (3,2)]:
    wts = [w_v3, w_v3, w_v3, w_v4, w_v4]
    def make_wbf(w):
        def fn(passes, img_w, img_h):
            return wbf(passes, img_w, img_h, weights=w)
        return fn
    c, d, cl = evaluate(make_wbf(wts), snms_std)
    print(f"  v3={w_v3} v4={w_v4}: {c:.4f} (det={d:.4f} cls={cl:.4f})")

print("\n" + "="*80)
print("TEST 5: Post-WBF re-vote — sweep vote_iou and mode")
for vote_iou in [0.3, 0.4, 0.5]:
    for vm in ['score', 'count']:
        def make_pv(vi, mode):
            def fn(passes, img_w, img_h):
                return wbf_post_vote(passes, img_w, img_h, vote_iou=vi, vote_mode=mode)
            return fn
        c, d, cl = evaluate(make_pv(vote_iou, vm), snms_std)
        print(f"  viou={vote_iou:.1f} mode={vm:5s}: {c:.4f} (det={d:.4f} cls={cl:.4f})")

print("\n" + "="*80)
print("TEST 6: Post-WBF re-vote + no soft-NMS vote (avoid double voting)")
for vote_iou in [0.3, 0.4, 0.5]:
    for vm in ['score', 'count']:
        def make_pv(vi, mode):
            def fn(passes, img_w, img_h):
                return wbf_post_vote(passes, img_w, img_h, vote_iou=vi, vote_mode=mode)
            return fn
        c, d, cl = evaluate(make_pv(vote_iou, vm), soft_nms_no_vote)
        print(f"  viou={vote_iou:.1f} mode={vm:5s}: {c:.4f} (det={d:.4f} cls={cl:.4f})")

print("\n" + "="*80)
print("TEST 7: WBF model weights + post-vote (combined)")
best_combo = None
for w_v3, w_v4 in [(2,1), (3,1), (3,2)]:
    for vote_iou in [0.3, 0.5]:
        wts = [w_v3, w_v3, w_v3, w_v4, w_v4]
        def make_combo(w, vi):
            def fn(passes, img_w, img_h):
                return wbf_post_vote(passes, img_w, img_h, weights=wts, vote_iou=vi)
            return fn
        # need to fix: pass weights to wbf_post_vote
        def make_combo2(w, vi):
            def fn(passes, img_w, img_h):
                fused = wbf(passes, img_w, img_h, weights=w)
                if not fused: return fused
                orig_b, orig_s, orig_l = [], [], []
                for p in passes:
                    for d in p:
                        x, y, ww, h = d["bbox"]
                        orig_b.append([x/img_w, y/img_h, (x+ww)/img_w, (y+h)/img_h])
                        orig_s.append(d["score"]); orig_l.append(d["category_id"])
                orig_b = np.array(orig_b, np.float32)
                orig_s = np.array(orig_s, np.float32)
                orig_l = np.array(orig_l, np.int32)
                fb = np.array([[d["bbox"][0]/img_w, d["bbox"][1]/img_h,
                              (d["bbox"][0]+d["bbox"][2])/img_w, (d["bbox"][1]+d["bbox"][3])/img_h]
                             for d in fused], np.float32)
                iou_mat = _iou_matrix_norm(fb, orig_b)
                for i, f in enumerate(fused):
                    mask = iou_mat[i] > vi
                    if mask.any():
                        ml, ms = orig_l[mask], orig_s[mask]
                        cat = {}
                        for lbl, sc in zip(ml, ms): cat[lbl] = cat.get(lbl,0) + sc
                        f["category_id"] = int(max(cat, key=cat.get))
                return fused
            return fn
        c, d, cl = evaluate(make_combo2(wts, vote_iou), snms_std)
        print(f"  v3={w_v3} v4={w_v4} viou={vote_iou:.1f}: {c:.4f} (det={d:.4f} cls={cl:.4f})")
        if best_combo is None or c > best_combo[0]:
            best_combo = (c, d, cl, w_v3, w_v4, vote_iou)

print("\n" + "="*80)
print("TEST 8: WBF conf threshold (model conf before WBF)")
# What if we increase model conf to suppress low-confidence wrong-class detections?
for conf in [0.03, 0.05, 0.08, 0.10, 0.15]:
    def make_wbf_conf(c_thresh):
        def fn(passes, img_w, img_h):
            # Filter passes by confidence
            filtered = [[d for d in p if d["score"] >= c_thresh] for p in passes]
            return wbf(filtered, img_w, img_h)
        return fn
    c, d, cl = evaluate(make_wbf_conf(conf), snms_std)
    print(f"  pre_conf={conf:.2f}: {c:.4f} (det={d:.4f} cls={cl:.4f})")

print(f"\n{'='*80}")
if best_combo:
    print(f"Best combo from TEST 7: {best_combo[0]:.4f} (v3={best_combo[3]} v4={best_combo[4]} viou={best_combo[5]})")

print(f"\nTotal: {time.time()-t0:.1f}s")
print("DONE!")
