"""Joint sigma+WBF optimization around best found params with weighted voting."""
import pickle, numpy as np, json
from pathlib import Path
from ensemble_boxes import weighted_boxes_fusion
from evaluate_local import load_coco_ground_truth, evaluate_mAP

with open('cache_v3_29.pkl','rb') as f: cv3 = pickle.load(f)
with open('cache_v4_29.pkl','rb') as f: cv4 = pickle.load(f)
with open('cache_v6_29.pkl','rb') as f: cv6 = pickle.load(f)
imgs = sorted(set(cv3.keys()) & set(cv4.keys()) & set(cv6.keys()))
W = [1,2,1,2,3,4,1,2]
gt_path = Path('data/coco/train/annotations.json')
img_ids = set()
for k in imgs: img_ids.add(int(Path(k).stem.split('_')[-1]) if isinstance(k,str) else k)
gt_data = load_coco_ground_truth(gt_path, img_ids)

def wbf_fuse(passes, iw, ih, iou_t, skip):
    bl, sl, ll = [], [], []
    for p in passes:
        b, s, l = [], [], []
        for d in p:
            x,y,ww,h = d['bbox']
            x1,y1 = max(0,x/iw), max(0,y/ih)
            x2,y2 = min(1,(x+ww)/iw), min(1,(y+h)/ih)
            if x2<=x1 or y2<=y1: continue
            b.append([x1,y1,x2,y2]); s.append(d['score']); l.append(d['category_id'])
        if b: bl.append(np.array(b,np.float32)); sl.append(np.array(s,np.float32)); ll.append(np.array(l,np.int32))
    if not bl: return []
    boxes,scores,labels = weighted_boxes_fusion(bl,sl,ll,weights=W,iou_thr=iou_t,skip_box_thr=skip,conf_type='box_and_model_avg')
    return [{'bbox':[boxes[i][0]*iw,boxes[i][1]*ih,(boxes[i][2]-boxes[i][0])*iw,(boxes[i][3]-boxes[i][1])*ih],'score':float(scores[i]),'category_id':int(labels[i])} for i in range(len(boxes))]

def _iou(a,b):
    ax,ay,aw,ah = a; bx,by,bw,bh = b
    ix = max(0, min(ax+aw,bx+bw) - max(ax,bx)); iy = max(0, min(ay+ah,by+bh) - max(ay,by))
    return ix*iy / (aw*ah + bw*bh - ix*iy + 1e-12)

def snms(dets, iou_t, sigma, score_t):
    if not dets: return dets
    dets = [d.copy() for d in dets]; dets.sort(key=lambda x: x['score'], reverse=True)
    kept = []; absorbed = []
    while dets:
        best = dets.pop(0); kept.append(best); cur = [best.copy()]; rem = []
        for d in dets:
            iou = _iou(best['bbox'],d['bbox'])
            if iou >= iou_t: cur.append(d.copy())
            d['score'] *= np.exp(-(iou**2)/sigma)
            if d['score'] >= score_t: rem.append(d)
        absorbed.append(cur); dets = sorted(rem, key=lambda x: x['score'], reverse=True)
    for ki in range(len(kept)):
        if len(absorbed[ki])>1:
            cs={}
            for ab in absorbed[ki]: cs[ab['category_id']]=cs.get(ab['category_id'],0)+ab['score']**2
            kept[ki]['category_id']=max(cs,key=cs.get)
    if len(kept)>450: kept.sort(key=lambda x:x['score'],reverse=True); kept=kept[:450]
    return kept

def run(wbf_iou, sigma, nms_iou, skip):
    all_preds = []
    for ik in imgs:
        ps = [cv3[ik]['full_1280'],cv3[ik]['full_1408'],cv3[ik]['full_1536'],cv4[ik]['full_1280'],cv4[ik]['full_1536'],cv6[ik]['full_1280'],cv6[ik]['full_1408'],cv6[ik]['full_1536']]
        iw,ih = cv3[ik].get('img_w',1920), cv3[ik].get('img_h',1080)
        f = wbf_fuse(ps, iw, ih, wbf_iou, skip)
        final = snms(f, nms_iou, sigma, 6e-6)
        img_id = int(Path(ik).stem.split('_')[-1]) if isinstance(ik,str) else ik
        for d in final: all_preds.append({'image_id':img_id,'category_id':d['category_id'],'bbox':d['bbox'],'score':d['score']})
    det_r = evaluate_mAP(all_preds, gt_data, iou_threshold=0.5, ignore_category=True)
    cls_r = evaluate_mAP(all_preds, gt_data, iou_threshold=0.5, ignore_category=False)
    return 0.7*det_r['mAP']+0.3*cls_r['mAP'], det_r['mAP'], cls_r['mAP']

combos = [
    ('CURRENT: s=0.949,w=0.4989', 0.4989, 0.949, 0.303),
    ('s=0.85,w=0.4989', 0.4989, 0.85, 0.303),
    ('s=0.85,w=0.500', 0.500, 0.85, 0.303),
    ('s=0.85,w=0.502', 0.502, 0.85, 0.303),
    ('s=0.85,w=0.505', 0.505, 0.85, 0.303),
    ('s=0.90,w=0.500', 0.500, 0.90, 0.303),
    ('s=0.90,w=0.505', 0.505, 0.90, 0.303),
    ('s=0.80,w=0.500', 0.500, 0.80, 0.303),
    ('s=0.80,w=0.505', 0.505, 0.80, 0.303),
    ('s=0.949,w=0.500', 0.500, 0.949, 0.303),
    ('s=0.949,w=0.505', 0.505, 0.949, 0.303),
    # Finer around sigma=0.85
    ('s=0.83,w=0.4989', 0.4989, 0.83, 0.303),
    ('s=0.84,w=0.4989', 0.4989, 0.84, 0.303),
    ('s=0.86,w=0.4989', 0.4989, 0.86, 0.303),
    ('s=0.87,w=0.4989', 0.4989, 0.87, 0.303),
    ('s=0.83,w=0.505', 0.505, 0.83, 0.303),
    ('s=0.85,w=0.503', 0.503, 0.85, 0.303),
    ('s=0.85,w=0.504', 0.504, 0.85, 0.303),
]

best = None
print(f"{'Config':>30s} | {'comb':>7s}  {'det':>7s}  {'cls':>7s}")
print('-'*60)
for name,wbf,sig,nms in combos:
    c,d,cl = run(wbf, sig, nms, 0.004705)
    marker = ""
    if best is None or c > best[0]:
        best = (c, d, cl, name, wbf, sig, nms)
        marker = " **BEST**"
    print(f'{name:>30s} | {c:.5f}  {d:.5f}  {cl:.5f}{marker}')

print(f"\n>>> BEST: {best[3]} → combined={best[0]:.5f} det={best[1]:.5f} cls={best[2]:.5f}")
print(f"    wbf_iou={best[4]}, sigma={best[5]}, nms_iou={best[6]}")

# Save best
import json as _json
with open('joint_v10_best.json','w') as f:
    _json.dump({'combined': best[0], 'det': best[1], 'cls': best[2], 'name': best[3],
                'wbf_iou': best[4], 'sigma': best[5], 'nms_iou': best[6]}, f, indent=2)
