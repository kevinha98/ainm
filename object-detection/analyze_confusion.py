"""Analyze classification errors: which categories are being confused most."""
import pickle, numpy as np, json
from pathlib import Path
from collections import defaultdict
from ensemble_boxes import weighted_boxes_fusion

# Load caches + GT
with open('cache_v3_29.pkl','rb') as f: cv3 = pickle.load(f)
with open('cache_v4_29.pkl','rb') as f: cv4 = pickle.load(f)
with open('cache_v6_29.pkl','rb') as f: cv6 = pickle.load(f)
imgs = sorted(set(cv3.keys()) & set(cv4.keys()) & set(cv6.keys()))
weights = [1,2,1,2,3,4,1,2]

with open('data/coco/train/annotations.json') as f:
    coco = json.load(f)
img_ids = set()
for k in imgs:
    img_ids.add(int(Path(k).stem.split('_')[-1]) if isinstance(k, str) else k)
gt = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"]}
      for a in coco["annotations"] if a["image_id"] in img_ids]
print(f"GT: {len(gt)} annotations, {len(img_ids)} images")

def _iou(a, b):
    ax,ay,aw,ah = a; bx,by,bw,bh = b
    ix = max(0, min(ax+aw,bx+bw) - max(ax,bx))
    iy = max(0, min(ay+ah,by+bh) - max(ay,by))
    inter = ix*iy
    return inter / (aw*ah + bw*bh - inter + 1e-12)

def wbf_fuse(passes, img_w, img_h):
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
    boxes,scores,labels = weighted_boxes_fusion(bl,sl,ll,weights=weights,iou_thr=0.4989,skip_box_thr=0.004705,conf_type='box_and_model_avg')
    out = []
    for i in range(len(boxes)):
        bx = boxes[i]
        out.append({'bbox':[bx[0]*img_w,bx[1]*img_h,(bx[2]-bx[0])*img_w,(bx[3]-bx[1])*img_h],'score':float(scores[i]),'category_id':int(labels[i])})
    return out

def soft_nms_weighted(dets, iou_thresh=0.303, sigma=0.949, score_thresh=6e-6, max_dets=450):
    if not dets: return dets, []
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
    return kept, absorbed

# Generate predictions and match to GT
all_preds = []
confusion_pairs = []  # (pred_cat, gt_cat, score, iou)
missed_gt = []
wrong_matches = 0
correct_matches = 0

for img_key in imgs:
    passes = [cv3[img_key]['full_1280'], cv3[img_key]['full_1408'], cv3[img_key]['full_1536'],
              cv4[img_key]['full_1280'], cv4[img_key]['full_1536'],
              cv6[img_key]['full_1280'], cv6[img_key]['full_1408'], cv6[img_key]['full_1536']]
    img_w = cv3[img_key].get('img_w', 1920)
    img_h = cv3[img_key].get('img_h', 1080)
    fused = wbf_fuse(passes, img_w, img_h)
    final, absorbed = soft_nms_weighted(fused)
    
    img_id = int(Path(img_key).stem.split('_')[-1]) if isinstance(img_key, str) else img_key
    img_gt = [g for g in gt if g['image_id'] == img_id]
    
    # Match predictions to GT (greedy, IoU >= 0.5)
    matched_gt = set()
    for pi, pred in enumerate(final):
        best_iou = 0; best_gi = -1
        for gi, g in enumerate(img_gt):
            if gi in matched_gt: continue
            iou = _iou(pred['bbox'], g['bbox'])
            if iou > best_iou:
                best_iou = iou; best_gi = gi
        if best_iou >= 0.5 and best_gi >= 0:
            matched_gt.add(best_gi)
            gt_cat = img_gt[best_gi]['category_id']
            pred_cat = pred['category_id']
            if pred_cat != gt_cat:
                wrong_matches += 1
                confusion_pairs.append((pred_cat, gt_cat, pred['score'], best_iou))
            else:
                correct_matches += 1

print(f"\nMatched predictions: {correct_matches + wrong_matches}")
print(f"  Correct class: {correct_matches} ({100*correct_matches/(correct_matches+wrong_matches):.1f}%)")
print(f"  Wrong class:   {wrong_matches} ({100*wrong_matches/(correct_matches+wrong_matches):.1f}%)")

# Most confused category pairs
pair_counts = defaultdict(int)
for pred_cat, gt_cat, score, iou in confusion_pairs:
    pair_counts[(pred_cat, gt_cat)] += 1

print(f"\nTop 20 confused category pairs (pred→gt):")
for (p, g), cnt in sorted(pair_counts.items(), key=lambda x: -x[1])[:20]:
    print(f"  pred={p:3d} → gt={g:3d}: {cnt} times")

# Most misclassified GT categories
gt_errors = defaultdict(int)
for _, gt_cat, _, _ in confusion_pairs:
    gt_errors[gt_cat] += 1
print(f"\nTop 20 GT categories with most misclassifications:")
for cat, cnt in sorted(gt_errors.items(), key=lambda x: -x[1])[:20]:
    print(f"  category {cat:3d}: {cnt} errors")

# Score distribution of correct vs wrong
correct_scores = [s for (p,g,s,_) in confusion_pairs if False]  # placeholder
wrong_scores = [s for _,_,s,_ in confusion_pairs]
if wrong_scores:
    print(f"\nWrong-class prediction scores: mean={np.mean(wrong_scores):.4f}, median={np.median(wrong_scores):.4f}")

# Check: how many errors would be fixed by taking highest-confidence raw detection instead of voting
print("\n\nAnalyzing per-model agreement on confused detections...")
# For each confused prediction, check what each individual model said
model_agree = defaultdict(int)
model_disagree = defaultdict(int)
