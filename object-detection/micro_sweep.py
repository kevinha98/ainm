"""Micro-sweep around PHASE2_BEST to find the absolute best parameters.

PHASE2_BEST: 0.95155 (det=0.9703, cls=0.9078)
weights=[1,2,1,2,3,4,1,2], wbf=0.4989, sigma=0.949, iou=0.303, score=6e-6, max_dets=450, skip_box=0.004705
"""
import pickle, json, time, os
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Load cached data ──
print("Loading cached detections...", flush=True)
c3 = pickle.load(open('cache_v3_29.pkl', 'rb'))
c4 = pickle.load(open('cache_v4_29.pkl', 'rb'))
c6 = pickle.load(open('cache_v6_29.pkl', 'rb'))
image_ids = sorted(set(c3.keys()) & set(c4.keys()) & set(c6.keys()))
gt_data = json.load(open('data/coco/train/annotations.json'))
gt_by_image = {}
for ann in gt_data["annotations"]:
    gt_by_image.setdefault(ann["image_id"], []).append(ann)
img_info = {img["id"]: img for img in gt_data["images"]}
print(f"Loaded {len(image_ids)} images", flush=True)


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax+aw, bx+bw) - max(ax, bx))
    iy = max(0, min(ay+ah, by+bh) - max(ay, by))
    inter = ix * iy
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0


def soft_nms(dets, iou_thresh=0.45, sigma=0.5, score_thresh=0.001, vote_mode='none'):
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
            iou = _iou(best["bbox"], d["bbox"])
            if iou >= iou_thresh:
                grp.append(d.copy())
            d["score"] *= np.exp(-(iou ** 2) / sigma)
            if d["score"] >= score_thresh:
                rem.append(d)
        groups.append(grp)
        dets = sorted(rem, key=lambda x: x["score"], reverse=True)
    if vote_mode == 'score':
        for ki, k in enumerate(kept):
            if len(groups[ki]) > 1:
                cs = {}
                for ab in groups[ki]:
                    cs[ab["category_id"]] = cs.get(ab["category_id"], 0) + ab["score"]
                kept[ki]["category_id"] = max(cs, key=cs.get)
    return kept


def wbf_fuse(passes, img_w, img_h, iou_thresh=0.50, skip=0.005, ct='box_and_model_avg', weights=None):
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
    kw = dict(iou_thr=iou_thresh, skip_box_thr=skip, conf_type=ct)
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
            iou = _iou(p["bbox"], g["bbox"])
            if iou > bi:
                bi, bj = iou, j
        if bi >= iou_thresh and bj >= 0 and not matched[bj]:
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
    for i in range(len(mp)-2, -1, -1):
        mp[i] = max(mp[i], mp[i+1])
    idx = np.where(mr[1:] != mr[:-1])[0]
    return np.sum((mr[idx+1] - mr[idx]) * mp[idx+1])


def get_passes(iid):
    return [
        c3[iid]['full_1280'], c3[iid]['full_1408'], c3[iid]['full_1536'],
        c4[iid]['full_1280'], c4[iid]['full_1536'],
        c6[iid]['full_1280'], c6[iid]['full_1408'], c6[iid]['full_1536']
    ]


def evaluate(weights=[1, 2, 1, 2, 3, 4, 1, 2], wbf_iou=0.4989, snms_sigma=0.949,
             snms_iou=0.303, snms_score=6e-6, max_dets=450,
             skip_box=0.004705, conf_type='box_and_model_avg', vote_mode='none'):
    da, ca = [], []
    for iid in image_ids:
        info = img_info.get(iid)
        if not info:
            continue
        iw, ih = info["width"], info["height"]
        ap = get_passes(iid)
        w = list(weights)
        if len(ap) > 1:
            fused = wbf_fuse(ap, iw, ih, iou_thresh=wbf_iou, skip=skip_box,
                           ct=conf_type, weights=w)
        elif len(ap) == 1:
            fused = ap[0]
        else:
            fused = []
        dets = soft_nms(fused, iou_thresh=snms_iou, sigma=snms_sigma,
                       score_thresh=snms_score, vote_mode=vote_mode)
        if len(dets) > max_dets:
            dets.sort(key=lambda x: x["score"], reverse=True)
            dets = dets[:int(max_dets)]
        gts = [{"bbox": g["bbox"], "category_id": g["category_id"]} for g in gt_by_image.get(iid, [])]
        dp = [{"bbox": d["bbox"], "score": d["score"], "category_id": 0} for d in dets]
        dg = [{"bbox": g["bbox"], "category_id": 0} for g in gts]
        da.append(compute_ap(dp, dg))
        ca.append(compute_ap(dets, gts, check_class=True))
    det = np.mean(da)
    cls = np.mean(ca)
    return 0.7 * det + 0.3 * cls, det, cls


if __name__ == '__main__':
    t0 = time.time()

    # PHASE2_BEST baseline
    base = dict(
        weights=[1, 2, 1, 2, 3, 4, 1, 2],
        wbf_iou=0.4989,
        snms_sigma=0.949,
        snms_iou=0.303,
        snms_score=6e-6,
        max_dets=450,
        skip_box=0.004705,
        vote_mode='none',
    )

    c, d, cl = evaluate(**base)
    print(f"BASELINE: {c:.5f} (det={d:.5f} cls={cl:.5f})")
    best_score = c
    best_params = dict(base)

    results = []

    # 1. Ultra-fine sigma sweep (0.85 to 1.05, step 0.01)
    print("\n--- Sigma micro-sweep ---")
    for sigma in np.arange(0.85, 1.06, 0.01):
        kw = dict(base)
        kw['snms_sigma'] = round(float(sigma), 3)
        c, d, cl = evaluate(**kw)
        tag = " ***" if c > best_score else ""
        print(f"  sigma={kw['snms_sigma']:.3f} -> {c:.5f} (det={d:.5f} cls={cl:.5f}){tag}")
        if c > best_score:
            best_score = c
            best_params = dict(kw)
        results.append({'sigma': kw['snms_sigma'], 'combined': round(c, 6), 'det': round(d, 6), 'cls': round(cl, 6)})

    # 2. Ultra-fine WBF IoU sweep (0.48 to 0.52, step 0.002)
    print("\n--- WBF IoU micro-sweep ---")
    for wbf in np.arange(0.480, 0.520, 0.002):
        kw = dict(best_params)
        kw['wbf_iou'] = round(float(wbf), 4)
        c, d, cl = evaluate(**kw)
        tag = " ***" if c > best_score else ""
        print(f"  wbf_iou={kw['wbf_iou']:.4f} -> {c:.5f} (det={d:.5f} cls={cl:.5f}){tag}")
        if c > best_score:
            best_score = c
            best_params = dict(kw)
        results.append({'wbf_iou': kw['wbf_iou'], 'combined': round(c, 6), 'det': round(d, 6), 'cls': round(cl, 6)})

    # 3. SNMS IoU micro-sweep (0.25 to 0.40, step 0.01)
    print("\n--- SNMS IoU micro-sweep ---")
    for snms in np.arange(0.25, 0.41, 0.01):
        kw = dict(best_params)
        kw['snms_iou'] = round(float(snms), 3)
        c, d, cl = evaluate(**kw)
        tag = " ***" if c > best_score else ""
        print(f"  snms_iou={kw['snms_iou']:.3f} -> {c:.5f} (det={d:.5f} cls={cl:.5f}){tag}")
        if c > best_score:
            best_score = c
            best_params = dict(kw)
        results.append({'snms_iou': kw['snms_iou'], 'combined': round(c, 6), 'det': round(d, 6), 'cls': round(cl, 6)})

    # 4. Score threshold micro-sweep
    print("\n--- Score threshold micro-sweep ---")
    for st in [1e-7, 5e-7, 1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 1e-5, 2e-5, 5e-5, 1e-4]:
        kw = dict(best_params)
        kw['snms_score'] = st
        c, d, cl = evaluate(**kw)
        tag = " ***" if c > best_score else ""
        print(f"  score={st:.1e} -> {c:.5f} (det={d:.5f} cls={cl:.5f}){tag}")
        if c > best_score:
            best_score = c
            best_params = dict(kw)

    # 5. Skip box threshold micro-sweep
    print("\n--- Skip box thresh micro-sweep ---")
    for sb in [0.001, 0.002, 0.003, 0.004, 0.0045, 0.0047, 0.005, 0.0055, 0.006, 0.007, 0.008, 0.01]:
        kw = dict(best_params)
        kw['skip_box'] = sb
        c, d, cl = evaluate(**kw)
        tag = " ***" if c > best_score else ""
        print(f"  skip_box={sb:.4f} -> {c:.5f} (det={d:.5f} cls={cl:.5f}){tag}")
        if c > best_score:
            best_score = c
            best_params = dict(kw)

    # 6. Max_dets micro-sweep
    print("\n--- Max dets micro-sweep ---")
    for md in [350, 400, 420, 440, 450, 460, 480, 500, 550, 600]:
        kw = dict(best_params)
        kw['max_dets'] = md
        c, d, cl = evaluate(**kw)
        tag = " ***" if c > best_score else ""
        print(f"  max_dets={md} -> {c:.5f} (det={d:.5f} cls={cl:.5f}){tag}")
        if c > best_score:
            best_score = c
            best_params = dict(kw)

    # 7. Weight perturbation search
    print("\n--- Weight perturbation ---")
    base_w = best_params['weights']
    for pos in range(8):
        for delta in [-1, 1]:
            w = list(base_w)
            w[pos] = max(0, w[pos] + delta)
            if w == base_w:
                continue
            kw = dict(best_params)
            kw['weights'] = w
            c, d, cl = evaluate(**kw)
            tag = " ***" if c > best_score else ""
            print(f"  w={w} -> {c:.5f} (det={d:.5f} cls={cl:.5f}){tag}")
            if c > best_score:
                best_score = c
                best_params = dict(kw)
                best_params['weights'] = w

    # 8. 2D grid: sigma x wbf_iou around the final best
    print("\n--- Final 2D grid (sigma x wbf) ---")
    for sigma in np.arange(best_params['snms_sigma'] - 0.05, best_params['snms_sigma'] + 0.06, 0.01):
        for wbf in np.arange(best_params['wbf_iou'] - 0.005, best_params['wbf_iou'] + 0.006, 0.001):
            kw = dict(best_params)
            kw['snms_sigma'] = round(float(sigma), 4)
            kw['wbf_iou'] = round(float(wbf), 4)
            c, d, cl = evaluate(**kw)
            if c > best_score:
                best_score = c
                best_params = dict(kw)
                print(f"  *** NEW BEST: sigma={kw['snms_sigma']:.4f}, wbf={kw['wbf_iou']:.4f} -> {c:.5f}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"MICRO-SWEEP COMPLETE in {elapsed:.1f}s")
    print(f"BEST SCORE: {best_score:.6f}")
    print(f"BEST PARAMS: {best_params}")

    # Save results
    with open('micro_sweep_result.json', 'w') as f:
        json.dump({
            'best_score': best_score,
            'best_params': best_params,
            'elapsed': elapsed,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)
    print(f"Saved to micro_sweep_result.json")
