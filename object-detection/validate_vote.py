"""Quick validation: compare vote modes on run.py's actual params (sigma=0.949)."""
import pickle, numpy as np, json
from pathlib import Path
from ensemble_boxes import weighted_boxes_fusion
from evaluate_local import load_coco_ground_truth, evaluate_mAP

# Load caches
with open('cache_v3_29.pkl','rb') as f: cv3 = pickle.load(f)
with open('cache_v4_29.pkl','rb') as f: cv4 = pickle.load(f)
with open('cache_v6_29.pkl','rb') as f: cv6 = pickle.load(f)
imgs = sorted(set(cv3.keys()) & set(cv4.keys()) & set(cv6.keys()))
weights = [1,2,1,2,3,4,1,2]

# Load GT
gt_path = Path('data/coco/train/annotations.json')
img_ids = set()
for k in imgs:
    img_ids.add(int(Path(k).stem.split('_')[-1]) if isinstance(k, str) else k)
gt = load_coco_ground_truth(gt_path, img_ids)
print(f"Loaded {len(gt)} GT annotations for {len(imgs)} images")

def wbf_fuse(passes, img_w, img_h, iou_thresh, skip, weights):
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

def soft_nms(dets, iou_thresh, sigma, score_thresh, vote_mode, max_dets=450):
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
    if vote_mode == 'score':
        for ki in range(len(kept)):
            if len(absorbed[ki])>1:
                cs={}
                for ab in absorbed[ki]: cs[ab['category_id']]=cs.get(ab['category_id'],0)+ab['score']
                kept[ki]['category_id']=max(cs,key=cs.get)
    elif vote_mode == 'weighted':
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

configs = [
    # name, sigma, iou_t, score_t, vote_mode, max_dets
    ('run.py_OLD(score,s=0.949)', 0.949, 0.303, 6e-6, 'score', 450),
    ('run.py_NEW(weighted,s=0.949)', 0.949, 0.303, 6e-6, 'weighted', 450),
    ('run.py_none(s=0.949)', 0.949, 0.303, 6e-6, 'none', 450),
    ('elite(weighted,s=1.06)', 1.06, 0.309, 9e-6, 'weighted', 450),
    ('elite(none,s=1.06)', 1.06, 0.309, 9e-6, 'none', 450),
    ('E3788(none,s=0.949)', 0.949, 0.303, 6e-6, 'none', 400),
]

print(f"\n{'Config':>35s} | {'combined':>8s}  {'det':>7s}  {'cls':>7s}")
print("-"*75)
for name, sigma, iou_t, score_t, vote_mode, max_dets in configs:
    all_preds = []
    for img_key in imgs:
        passes = [cv3[img_key]['full_1280'], cv3[img_key]['full_1408'], cv3[img_key]['full_1536'],
                  cv4[img_key]['full_1280'], cv4[img_key]['full_1536'],
                  cv6[img_key]['full_1280'], cv6[img_key]['full_1408'], cv6[img_key]['full_1536']]
        img_w = cv3[img_key].get('img_w', 1920)
        img_h = cv3[img_key].get('img_h', 1080)
        fused = wbf_fuse(passes, img_w, img_h, iou_thresh=0.4989, skip=0.004705, weights=weights)
        final = soft_nms(fused, iou_thresh=iou_t, sigma=sigma, score_thresh=score_t, vote_mode=vote_mode, max_dets=max_dets)
        img_id = int(Path(img_key).stem.split('_')[-1]) if isinstance(img_key, str) else img_key
        for d in final:
            all_preds.append({'image_id':img_id,'category_id':d['category_id'],'bbox':d['bbox'],'score':d['score']})
    det_r = evaluate_mAP(all_preds, gt, iou_threshold=0.5, ignore_category=True)
    cls_r = evaluate_mAP(all_preds, gt, iou_threshold=0.5, ignore_category=False)
    det, cls = det_r['mAP'], cls_r['mAP']
    comb = 0.7*det + 0.3*cls
    print(f'{name:>35s} | {comb:.5f}  {det:.5f}  {cls:.5f}')
