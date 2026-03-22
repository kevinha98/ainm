"""Mass Simulation Factory: Generates and runs 100 individual simulation experiments.

Each sim is saved as sims/sim_NNN.py and its result as sims/sim_NNN_result.json.
All results also logged to overnight_log.md.

Best found from sim 1-3:
  Combined: 0.9508, det=0.9693, cls=0.9076
  weights=[1,1,1,2,1,2,1,1], wbf_iou=0.47, snms_sigma=1.2
"""
import pickle, json, time, os, random, itertools, hashlib
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('sims', exist_ok=True)

# ── Load data ──
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

SCALE_NAMES = ["v3@1280","v3@1408","v3@1536","v4@1280","v4@1536","v6@1280","v6@1408","v6@1536"]

# ── Core functions ──
def _iou(a, b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    ix=max(0,min(ax+aw,bx+bw)-max(ax,bx)); iy=max(0,min(ay+ah,by+bh)-max(ay,by))
    inter=ix*iy; union=aw*ah+bw*bh-inter
    return inter/union if union>0 else 0

def soft_nms(dets, iou_thresh=0.45, sigma=5.0, score_thresh=0.001, vote_mode='none'):
    if not dets: return []
    dets=[d.copy() for d in dets]; dets.sort(key=lambda x:x["score"],reverse=True)
    kept,groups=[],[]
    while dets:
        best=dets.pop(0); kept.append(best); grp=[best.copy()]; rem=[]
        for d in dets:
            iou=_iou(best["bbox"],d["bbox"])
            if iou>=iou_thresh: grp.append(d.copy())
            d["score"]*=np.exp(-(iou**2)/sigma)
            if d["score"]>=score_thresh: rem.append(d)
        groups.append(grp); dets=sorted(rem,key=lambda x:x["score"],reverse=True)
    if vote_mode=='score':
        for ki,k in enumerate(kept):
            if len(groups[ki])>1:
                cs={}
                for ab in groups[ki]: cs[ab["category_id"]]=cs.get(ab["category_id"],0)+ab["score"]
                kept[ki]["category_id"]=max(cs,key=cs.get)
    return kept

def wbf_fuse(passes, img_w, img_h, iou_thresh=0.50, skip=0.005, ct='box_and_model_avg', weights=None):
    bl,sl,ll=[],[],[]
    for p in passes:
        b,s,l=[],[],[]
        for d in p:
            x,y,w,h=d["bbox"]
            x1,y1=max(0,x/img_w),max(0,y/img_h); x2,y2=min(1,(x+w)/img_w),min(1,(y+h)/img_h)
            if x2<=x1 or y2<=y1: continue
            b.append([x1,y1,x2,y2]); s.append(d["score"]); l.append(d["category_id"])
        if b: bl.append(np.array(b,np.float32)); sl.append(np.array(s,np.float32)); ll.append(np.array(l,np.int32))
    if not bl: return []
    kw=dict(iou_thr=iou_thresh, skip_box_thr=skip, conf_type=ct)
    if weights: kw['weights']=weights
    fb,fs,fl=weighted_boxes_fusion(bl,sl,ll,**kw)
    return [{"category_id":int(la),"bbox":[round(b[0]*img_w,1),round(b[1]*img_h,1),
             round((b[2]-b[0])*img_w,1),round((b[3]-b[1])*img_h,1)],"score":round(float(sc),3)}
            for b,sc,la in zip(fb,fs,fl)]

def compute_ap(preds, gts, iou_thresh=0.5, check_class=False):
    if not gts: return 1.0 if not preds else 0.0
    if not preds: return 0.0
    preds=sorted(preds,key=lambda x:x["score"],reverse=True); matched=[False]*len(gts); tp,fp=[],[]
    for p in preds:
        bi,bj=0,-1
        for j,g in enumerate(gts):
            if check_class and p["category_id"]!=g["category_id"]: continue
            iou=_iou(p["bbox"],g["bbox"])
            if iou>bi: bi,bj=iou,j
        if bi>=iou_thresh and bj>=0 and not matched[bj]: tp.append(1); fp.append(0); matched[bj]=True
        else: tp.append(0); fp.append(1)
    tc=np.cumsum(tp); fc=np.cumsum(fp); rec=tc/len(gts); prec=tc/(tc+fc)
    mr=np.concatenate(([0.],rec,[1.])); mp=np.concatenate(([1.],prec,[0.]))
    for i in range(len(mp)-2,-1,-1): mp[i]=max(mp[i],mp[i+1])
    idx=np.where(mr[1:]!=mr[:-1])[0]; return np.sum((mr[idx+1]-mr[idx])*mp[idx+1])

def get_passes(iid):
    return [c3[iid]['full_1280'],c3[iid]['full_1408'],c3[iid]['full_1536'],
            c4[iid]['full_1280'],c4[iid]['full_1536'],
            c6[iid]['full_1280'],c6[iid]['full_1408'],c6[iid]['full_1536']]

def evaluate(weights=[1,1,1,2,1,2,1,1], wbf_iou=0.48, snms_sigma=1.0,
             snms_iou=0.45, snms_score=0.001, max_dets=400,
             skip_box=0.005, conf_type='box_and_model_avg', vote_mode='none',
             pass_mask=None):
    da,ca=[],[]
    for iid in image_ids:
        info=img_info.get(iid)
        if not info: continue
        iw,ih=info["width"],info["height"]; ap=get_passes(iid)
        if pass_mask:
            ps=[p for p,m in zip(ap,pass_mask) if m]; w=[wt for wt,m in zip(weights,pass_mask) if m]
        else: ps=ap; w=list(weights)
        if len(ps)>1: fused=wbf_fuse(ps,iw,ih,iou_thresh=wbf_iou,skip=skip_box,ct=conf_type,weights=w)
        elif len(ps)==1: fused=ps[0]
        else: fused=[]
        dets=soft_nms(fused,iou_thresh=snms_iou,sigma=snms_sigma,score_thresh=snms_score,vote_mode=vote_mode)
        if len(dets)>max_dets: dets.sort(key=lambda x:x["score"],reverse=True); dets=dets[:max_dets]
        gts=[{"bbox":g["bbox"],"category_id":g["category_id"]} for g in gt_by_image.get(iid,[])]
        dp=[{"bbox":d["bbox"],"score":d["score"],"category_id":0} for d in dets]
        dg=[{"bbox":g["bbox"],"category_id":0} for g in gts]
        da.append(compute_ap(dp,dg)); ca.append(compute_ap(dets,gts,check_class=True))
    det=np.mean(da); cls=np.mean(ca); return 0.7*det+0.3*cls, det, cls

LOG='overnight_log.md'
GLOBAL_BEST={'combined':0.0}
SIM_COUNT=0

def log(msg):
    print(msg,flush=True)
    with open(LOG,'a',encoding='utf-8') as f: f.write(msg+'\n')

def run_sim(sim_id, name, description, configs):
    """Run a simulation with multiple configs. Returns best result.
    configs: list of (experiment_name, kwargs_dict)
    Saves sim script + result JSON.
    """
    global GLOBAL_BEST, SIM_COUNT
    SIM_COUNT += 1
    
    log(f"\n### Sim {sim_id:03d}: {name}")
    log(f"*{description}*")
    log("| Time | Experiment | Combined | Det | Cls | Params |")
    log("|------|-----------|----------|-----|-----|--------|")
    
    sim_best = {'combined': 0.0}
    results = []
    
    for exp_name, kw in configs:
        c, d, cl = evaluate(**kw)
        is_global = c > GLOBAL_BEST.get('combined', 0)
        is_local = c > sim_best.get('combined', 0)
        if is_global:
            GLOBAL_BEST = {'combined': c, 'det': d, 'cls': cl, 'name': f"sim{sim_id:03d}:{exp_name}", 'kwargs': kw}
        if is_local:
            sim_best = {'combined': c, 'det': d, 'cls': cl, 'name': exp_name, 'kwargs': kw}
        
        marker = " **NEW GLOBAL BEST**" if is_global else (" *local best*" if is_local else "")
        ts = datetime.now().strftime("%H:%M:%S")
        log(f"| {ts} | {exp_name} | {c:.4f} | {d:.4f} | {cl:.4f} | {kw}{marker} |")
        results.append({'name': exp_name, 'combined': round(c,5), 'det': round(d,5), 'cls': round(cl,5), 'params': kw})
    
    # Save result JSON
    result_data = {
        'sim_id': sim_id,
        'name': name,
        'description': description,
        'best': sim_best,
        'global_best_at_end': GLOBAL_BEST.get('combined', 0),
        'num_configs': len(configs),
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    with open(f'sims/sim_{sim_id:03d}_result.json', 'w') as f:
        json.dump(result_data, f, indent=2, default=str)
    
    # Save sim script (for reproducibility)
    best_kw = sim_best.get("kwargs", {})
    script = f'''# Sim {sim_id:03d}: {name}
# {description}
# Best: {sim_best.get("combined",0):.4f} (det={sim_best.get("det",0):.4f} cls={sim_best.get("cls",0):.4f})
# Config: {best_kw}
'''
    with open(f'sims/sim_{sim_id:03d}.py', 'w') as f:
        f.write(script)
    
    log(f"**Sim {sim_id:03d} best**: {sim_best.get('combined',0):.4f} | Global best: {GLOBAL_BEST.get('combined',0):.4f}")
    return sim_best

# ══════════════════════════════════════════════════════════════════
# SIMULATION DEFINITIONS (100 sims)
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()
    random.seed(2026)
    
    log(f"\n## Mass Simulation Factory -- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log(f"**Goal: 100 simulations, each saved to sims/ folder**\n")
    
    # Best known config from sim 1-3
    BEST_W = [1,1,1,2,1,2,1,1]
    BEST_KW = dict(wbf_iou=0.47, snms_sigma=1.2, snms_iou=0.45, snms_score=0.001, max_dets=400, vote_mode='none')
    
    # ── Sims 1-10: Weight variations around best ──
    for i, w in enumerate([
        [1,1,1,2,1,2,1,1],  # baseline
        [1,1,1,3,1,2,1,1],  # v4@1280=3
        [1,1,1,2,1,3,1,1],  # v6@1280=3
        [1,1,1,2,2,2,1,1],  # v4@1536=2
        [1,1,1,2,1,2,2,1],  # v6@1408=2
        [2,1,1,2,1,2,1,1],  # v3@1280=2
        [1,2,1,2,1,2,1,1],  # v3@1408=2
        [1,1,1,3,1,3,1,1],  # both 1280=3
        [1,1,1,2,2,2,2,1],  # v4+v6 mid scales=2
        [1,1,2,2,1,2,1,2],  # v3@1536=2, v6@1536=2
    ], start=1):
        configs = []
        for sigma in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
            configs.append((f"s={sigma}", dict(weights=w, wbf_iou=0.47, snms_sigma=sigma, snms_iou=0.45, snms_score=0.001, max_dets=400, vote_mode='none')))
        run_sim(i, f"Weight variant {i}", f"weights={w} x sigma sweep", configs)

    # ── Sims 11-20: WBF IoU fine grid with best weights ──
    for i, wbf in enumerate([0.43, 0.44, 0.45, 0.46, 0.465, 0.47, 0.475, 0.48, 0.49, 0.50], start=11):
        configs = []
        for sigma in [0.9, 1.0, 1.1, 1.2, 1.3, 1.4]:
            configs.append((f"s={sigma}", dict(weights=BEST_W, wbf_iou=wbf, snms_sigma=sigma, snms_iou=0.45, snms_score=0.001, max_dets=400, vote_mode='none')))
        run_sim(i, f"WBF={wbf:.3f} grid", f"wbf_iou={wbf} x sigma sweep with best weights", configs)

    # ── Sims 21-30: SNMS IoU sweep per sigma ──
    for i, sigma in enumerate([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0], start=21):
        configs = []
        for snms in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            configs.append((f"snms={snms}", dict(weights=BEST_W, wbf_iou=0.47, snms_sigma=sigma, snms_iou=snms, snms_score=0.001, max_dets=400, vote_mode='none')))
        run_sim(i, f"SNMS@sigma={sigma}", f"snms_iou sweep at sigma={sigma}", configs)

    # ── Sims 31-40: Score threshold fine sweep ──
    for i, base_sigma in enumerate([0.9, 1.0, 1.1, 1.2, 1.3], start=31):
        configs = []
        for st in [0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.001, 0.0015, 0.002, 0.003]:
            configs.append((f"st={st}", dict(weights=BEST_W, wbf_iou=0.47, snms_sigma=base_sigma, snms_iou=0.45, snms_score=st, max_dets=400, vote_mode='none')))
        run_sim(i, f"ScoreThresh@s={base_sigma}", f"score_thresh sweep at sigma={base_sigma}", configs)
    
    for i, base_sigma in enumerate([0.9, 1.0, 1.1, 1.2, 1.3], start=36):
        configs = []
        for st in [0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.001, 0.0015, 0.002, 0.003]:
            configs.append((f"st={st}", dict(weights=BEST_W, wbf_iou=0.48, snms_sigma=base_sigma, snms_iou=0.45, snms_score=st, max_dets=400, vote_mode='none')))
        run_sim(i, f"ScoreThresh@s={base_sigma}+wbf48", f"score_thresh sweep at sigma={base_sigma} wbf=0.48", configs)

    # ── Sims 41-50: Max detections sweep ──
    for i, (wbf, sigma) in enumerate([(0.47, 1.2), (0.47, 1.0), (0.48, 1.2), (0.48, 1.0), (0.47, 0.9),
                                        (0.46, 1.2), (0.46, 1.0), (0.49, 1.2), (0.47, 1.3), (0.47, 1.1)], start=41):
        configs = []
        for md in [200, 250, 300, 350, 400, 450, 500, 600, 800, 1000]:
            configs.append((f"md={md}", dict(weights=BEST_W, wbf_iou=wbf, snms_sigma=sigma, snms_iou=0.45, snms_score=0.001, max_dets=md, vote_mode='none')))
        run_sim(i, f"MaxDets@wbf{wbf}s{sigma}", f"max_dets sweep at wbf={wbf} sigma={sigma}", configs)

    # ── Sims 51-60: Vote mode comparison ──
    for i, (wbf, sigma) in enumerate([(0.47, 1.2), (0.47, 1.0), (0.48, 1.2), (0.48, 1.0), (0.47, 0.9),
                                        (0.46, 1.2), (0.46, 1.3), (0.47, 1.3), (0.47, 1.1), (0.48, 0.9)], start=51):
        configs = []
        for vm in ['none', 'score', 'count']:
            for st in [0.0001, 0.0005, 0.001]:
                configs.append((f"vote={vm},st={st}", dict(weights=BEST_W, wbf_iou=wbf, snms_sigma=sigma, snms_iou=0.45, snms_score=st, max_dets=400, vote_mode=vm)))
        run_sim(i, f"VoteMode@wbf{wbf}s{sigma}", f"vote_mode x score_thresh at wbf={wbf} sigma={sigma}", configs)

    # ── Sims 61-70: conf_type exploration ──
    for i, w in enumerate([
        [1,1,1,2,1,2,1,1], [1,1,1,3,1,2,1,1], [1,1,1,2,1,3,1,1],
        [1,1,1,2,2,2,1,1], [2,1,1,2,1,2,1,1], [1,1,1,3,1,3,1,1],
        [1,1,1,2,2,2,2,1], [1,2,1,2,1,2,1,1], [1,1,1,2,1,2,1,2],
        [1,1,1,1,1,2,1,1]
    ], start=61):
        configs = []
        for ct in ['avg', 'box_and_model_avg', 'absent_model_aware_avg', 'max']:
            for sigma in [1.0, 1.2]:
                configs.append((f"ct={ct[:8]},s={sigma}", dict(weights=w, wbf_iou=0.47, snms_sigma=sigma, snms_iou=0.45, snms_score=0.001, max_dets=400, vote_mode='none', conf_type=ct)))
        run_sim(i, f"ConfType w={w[:3]}..{w[5:]}", f"conf_type x sigma with weights={w}", configs)

    # ── Sims 71-80: skip_box_thresh exploration ──
    for i, (wbf, sigma) in enumerate([(0.47, 1.2), (0.47, 1.0), (0.48, 1.2), (0.48, 1.0), (0.47, 0.9),
                                        (0.46, 1.2), (0.47, 1.3), (0.47, 1.1), (0.48, 1.1), (0.46, 1.0)], start=71):
        configs = []
        for sb in [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02]:
            configs.append((f"sb={sb}", dict(weights=BEST_W, wbf_iou=wbf, snms_sigma=sigma, snms_iou=0.45, snms_score=0.001, max_dets=400, vote_mode='none', skip_box=sb)))
        run_sim(i, f"SkipBox@wbf{wbf}s{sigma}", f"skip_box_thresh sweep at wbf={wbf} sigma={sigma}", configs)

    # ── Sims 81-90: Random search (hill climbing from best) ──
    for i in range(81, 91):
        configs = []
        seen = set()
        for trial in range(15):
            w = list(BEST_W)
            n_changes = random.randint(1, 3)
            for pos in random.sample(range(8), n_changes):
                w[pos] = max(0, w[pos] + random.choice([-1, 0, 1]))
            wbf = round(random.uniform(0.44, 0.52), 3)
            sigma = round(random.uniform(0.7, 1.8), 2)
            snms = round(random.choice([0.35, 0.40, 0.45, 0.50, 0.55]), 2)
            st = random.choice([0.0001, 0.0005, 0.001])
            md = random.choice([300, 400, 500])
            key = (tuple(w), wbf, sigma, snms, st, md)
            if key in seen: continue
            seen.add(key)
            configs.append((f"rnd{trial}", dict(weights=w, wbf_iou=wbf, snms_sigma=sigma, snms_iou=snms, snms_score=st, max_dets=md, vote_mode='none')))
        run_sim(i, f"RandomSearch batch {i-80}", f"Random neighborhood search around best", configs)

    # ── Sims 91-100: Targeted combos of top findings ──
    for i, (w, wbf, sigma, snms, st) in enumerate([
        ([1,1,1,2,1,2,1,1], 0.47, 1.2, 0.45, 0.0001),
        ([1,1,1,2,1,2,1,1], 0.47, 1.2, 0.40, 0.0001),
        ([1,1,1,2,1,2,1,1], 0.47, 1.2, 0.50, 0.0001),
        ([1,1,1,2,1,2,1,1], 0.47, 1.15, 0.45, 0.0001),
        ([1,1,1,2,1,2,1,1], 0.47, 1.25, 0.45, 0.0001),
        ([1,1,1,2,1,2,1,1], 0.465, 1.2, 0.45, 0.0001),
        ([1,1,1,2,1,2,1,1], 0.475, 1.2, 0.45, 0.0001),
        ([1,1,1,2,1,2,1,1], 0.47, 1.2, 0.45, 0.0005),
        ([1,1,1,2,2,2,1,1], 0.47, 1.2, 0.45, 0.0001),
        ([1,1,1,2,1,2,1,1], 0.47, 1.2, 0.45, 0.001),
    ], start=91):
        configs = []
        # For each targeted combo, sweep several nearby params
        for wbf_d in [-0.01, -0.005, 0, 0.005, 0.01]:
            for sig_d in [-0.1, 0, 0.1]:
                wbf_v = round(wbf + wbf_d, 3)
                sig_v = round(sigma + sig_d, 2)
                if sig_v <= 0 or wbf_v <= 0: continue
                configs.append((f"wbf={wbf_v},s={sig_v}", dict(weights=w, wbf_iou=wbf_v, snms_sigma=sig_v, snms_iou=snms, snms_score=st, max_dets=400, vote_mode='none')))
        run_sim(i, f"Targeted combo {i-90}", f"Fine grid around w={w} wbf={wbf} s={sigma} snms={snms} st={st}", configs)

    # ── FINAL SUMMARY ──
    elapsed = time.time() - t0
    log(f"\n## MASS SIMULATION COMPLETE")
    log(f"**Total simulations: {SIM_COUNT}**")
    log(f"**GLOBAL BEST: {GLOBAL_BEST.get('combined',0):.4f} (det={GLOBAL_BEST.get('det',0):.4f} cls={GLOBAL_BEST.get('cls',0):.4f})**")
    log(f"**Config: {GLOBAL_BEST.get('name','')}**")
    log(f"**Params: {GLOBAL_BEST.get('kwargs',{})}**")
    log(f"**Total time: {elapsed:.1f}s**")
    
    # Save global summary
    summary = {
        'total_sims': SIM_COUNT,
        'global_best': GLOBAL_BEST,
        'elapsed_seconds': elapsed,
        'timestamp': datetime.now().isoformat()
    }
    with open('sims/SUMMARY.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print(f"DONE: {SIM_COUNT} simulations completed in {elapsed:.1f}s")
    print(f"GLOBAL BEST: {GLOBAL_BEST.get('combined',0):.4f}")
    print(f"Config: {GLOBAL_BEST.get('kwargs',{})}")
    print(f"Results saved to sims/ directory")
