"""Micro-sweep: joint sigma+WBF optimization around run.py's config with weighted voting."""
import pickle, numpy as np, json
from pathlib import Path
from collections import defaultdict
from ensemble_boxes import weighted_boxes_fusion
from evaluate_local import load_coco_ground_truth, evaluate_mAP

with open('cache_v3_29.pkl','rb') as f: cv3 = pickle.load(f)
with open('cache_v4_29.pkl','rb') as f: cv4 = pickle.load(f)
with open('cache_v6_29.pkl','rb') as f: cv6 = pickle.load(f)
imgs = sorted(set(cv3.keys()) & set(cv4.keys()) & set(cv6.keys()))
weights = [1,2,1,2,3,4,1,2]

gt_path = Path('data/coco/train/annotations.json')
img_ids = set()
for k in imgs:
    img_ids.add(int(Path(k).stem.split('_')[-1]) if isinstance(k, str) else k)
gt_data = load_coco_ground_truth(gt_path, img_ids)

def wbf_fuse(passes, img_w, img_h, iou_thresh, skip):
    bl, sl, ll = [], [], []
    for p in passes:
        b, s, l = [], [], []
        for d in p:
            x,y,w,h = d['bbox']
            x1, y1 = max(0,x/img_w), max(0,y/img_h)
            x2, y2 = min(1,(x+w)/img_w), min(1,(y+h)/img_h)
            if x2<=x1 or y2<=y1: continue
            b.append([x1,y1,x2,y2]); s.append(d['score']); l.append(d['category_id'])
        if b:
            bl.append(np.array(b,np.float32)); sl.append(np.array(s,np.float32)); ll.append(np.array(l,np.int32))
    if not bl: return []
    boxes,scores,labels = weighted_boxes_fusion(bl,sl,ll,weights=weights,iou_thr=iou_thresh,skip_box_thr=skip,conf_type='box_and_model_avg')
    out = []
    for i in range(len(boxes)):
        bx = boxes[i]
        out.append({'bbox':[bx[0]*img_w,bx[1]*img_h,(bx[2]-bx[0])*img_w,(bx[3]-bx[1])*img_h],'score':float(scores[i]),'category_id':int(labels[i])})
    return out

def _iou(a,b):
    ax,ay,aw,ah = a; bx,by,bw,bh = b
    ix = max(0, min(ax+aw,bx+bw) - max(ax,bx))
    iy = max(0, min(ay+ah,by+bh) - max(ay,by))
    inter = ix*iy
    return inter / (aw*ah + bw*bh - inter + 1e-12)

def soft_nms_weighted(dets, iou_thresh=0.303, sigma=0.949, score_thresh=6e-6, max_dets=450):
    if not dets: return dets
    dets = [d.copy() for d in dets]; dets.sort(key=lambda x: x['score'], reverse=True)
    kept = []; absorbed = []
    while dets:
        best = dets.pop(0); kept.append(best)
        cur_abs = [best.copy()]; remaining = []
        for d in dets:
            iou = _iou(best['bbox'],d['bbox'])
            if iou >= iou_thresh: cur_abs.append(d.copy())
            d['score'] *= np.exp(-(iou**2)/sigma)
            if d['score'] >= score_thresh: remaining.append(d)
        absorbed.append(cur_abs)
        dets = sorted(remaining, key=lambda x: x['score'], reverse=True)
    for ki in range(len(kept)):
        if len(absorbed[ki])>1:
            cs={}
            for ab in absorbed[ki]:
                w=ab['score']**2
                cs[ab['category_id']]=cs.get(ab['category_id'],0)+w
            kept[ki]['category_id']=max(cs,key=cs.get)
    if len(kept) > max_dets:
        kept.sort(key=lambda x: x['score'], reverse=True)
        kept = kept[:max_dets]
    return kept

def evaluate(preds):
    det_r = evaluate_mAP(preds, gt_data, iou_threshold=0.5, ignore_category=True)
    cls_r = evaluate_mAP(preds, gt_data, iou_threshold=0.5, ignore_category=False)
    return 0.7*det_r['mAP']+0.3*cls_r['mAP'], det_r['mAP'], cls_r['mAP']

def run_config(wbf_iou, sigma, nms_iou, skip):
    all_preds = []
    for img_key in imgs:
        passes = [cv3[img_key]['full_1280'], cv3[img_key]['full_1408'], cv3[img_key]['full_1536'],
                  cv4[img_key]['full_1280'], cv4[img_key]['full_1536'],
                  cv6[img_key]['full_1280'], cv6[img_key]['full_1408'], cv6[img_key]['full_1536']]
        img_w = cv3[img_key].get('img_w', 1920)
        img_h = cv3[img_key].get('img_h', 1080)
        fused = wbf_fuse(passes, img_w, img_h, iou_thresh=wbf_iou, skip=skip)
        final = soft_nms_weighted(fused, iou_thresh=nms_iou, sigma=sigma)
        img_id = int(Path(img_key).stem.split('_')[-1]) if isinstance(img_key, str) else img_key
        for d in final:
            all_preds.append({'image_id':img_id,'category_id':d['category_id'],'bbox':d['bbox'],'score':d['score']})
    return evaluate(all_preds)

print(f"{'Config':>50s} | {'comb':>7s}  {'det':>7s}  {'cls':>7s}")
print("-"*80)

# Current best
comb, det, cls = run_config(0.4989, 0.949, 0.303, 0.004705)
print(f"{'CURRENT(wbf=0.4989,s=0.949,n=0.303)':>50s} | {comb:.5f}  {det:.5f}  {cls:.5f}")

# Sigma sweep around 0.949
for sigma in [0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 1.00, 1.02, 1.05, 1.10]:
    comb, det, cls = run_config(0.4989, sigma, 0.303, 0.004705)
    print(f"{'sigma='+str(sigma):>50s} | {comb:.5f}  {det:.5f}  {cls:.5f}")

# WBF IoU fine sweep near 0.4989
for wbf in [0.490, 0.495, 0.497, 0.498, 0.499, 0.500, 0.501, 0.502, 0.505, 0.510]:
    comb, det, cls = run_config(wbf, 0.949, 0.303, 0.004705)
    print(f"{'wbf='+str(wbf):>50s} | {comb:.5f}  {det:.5f}  {cls:.5f}")

# skip_box_thresh sweep
for skip in [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010]:
    comb, det, cls = run_config(0.4989, 0.949, 0.303, skip)
    print(f"{'skip='+str(skip):>50s} | {comb:.5f}  {det:.5f}  {cls:.5f}")

# NMS IoU sweep
for nms_iou in [0.20, 0.25, 0.28, 0.30, 0.303, 0.32, 0.35, 0.40]:
    comb, det, cls = run_config(0.4989, 0.949, nms_iou, 0.004705)
    print(f"{'nms='+str(nms_iou):>50s} | {comb:.5f}  {det:.5f}  {cls:.5f}")
