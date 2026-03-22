# Object Detection — Learning Log

> Last updated: 2026-03-19 ~21:30 UTC

## Current State

- **Status**: Run 1 training — epoch 32/100, **mAP50-95 = 0.514**
- **Next action**: Wait for training to finish → download best.pt → package & submit
- **Blocker**: None — training is running smoothly on GCP VM

## Submissions

| # | Date | Model | Config | Local mAP50-95 | Competition Score | Notes |
|---|------|-------|--------|----------------|-------------------|-------|
| — | — | — | — | — | — | No submissions yet |

## Training Runs

### Run 1 — YOLOv8m, imgsz=1280, 100 epochs (ACTIVE)

Started: 2026-03-19 ~20:00 UTC | Config: batch=4, patience=20, AdamW lr=0.001→0.01 cosine

| Epoch | box_loss | cls_loss | dfl_loss | mAP50 | mAP50-95 | P | R | Timestamp |
|-------|----------|----------|----------|-------|----------|-------|-------|-----------|
| 1 | 1.543 | 5.426 | 1.457 | — | — | — | — | 2026-03-19 20:02 |
| 9 | 1.322 | 1.927 | 1.324 | 0.459 | 0.303 | 0.456 | 0.485 | 2026-03-19 20:08 |
| 18 | — | — | — | 0.668 | 0.459 | 0.629 | 0.650 | 2026-03-19 20:20 |
| 31 | — | — | — | 0.743 | 0.514 | 0.692 | 0.723 | 2026-03-19 21:00 |
| 32 | 1.15 | 1.14 | 1.27 | — | — | — | — | 2026-03-19 21:01 |

Observations:
- cls_loss: 5.4 → 1.9 → 1.14 — classification improving steadily across 356 classes
- mAP50-95: 0.303 → 0.459 → 0.514 — still climbing at epoch 31, no plateau yet
- GPU mem stable at ~9.2GB / 14.6GB available — could try batch=8
- ~40 sec/epoch (53 batches at ~1.3 it/s + val)

## Key Decisions

| # | Decision | Rationale | Date |
|---|----------|-----------|------|
| 1 | YOLOv8m over YOLOv8l | M fits T4 VRAM at imgsz=1280; L would need batch=2 | 2026-03-19 |
| 2 | imgsz=1280 | Shelf images have small products — high res critical | 2026-03-19 |
| 3 | batch=4 | Safe for T4 at 1280px | 2026-03-19 |
| 4 | Heavy augmentation | mosaic=0.8, mixup=0.15, copy_paste=0.2 — only 210 train images | 2026-03-19 |
| 5 | patience=20 | 356 classes needs many epochs before convergence | 2026-03-19 |

## Failed Approaches

*(Track things that didn't work to avoid re-trying)*

| Approach | Result | Date |
|----------|--------|------|
| — | — | — |

## Improvement Ideas (Priority Order)

**Quick wins:**
1. TTA at inference — already in run.py (`augment=True`), free mAP boost
2. Lower conf threshold — try 0.15 instead of 0.25 for more recall
3. NMS tuning — experiment with iou=0.5 or 0.6

**Medium effort:**
4. Resume training with lower LR if mAP plateaus
5. Multi-scale training (640+1280) → infer at 1280
6. Pseudo-labeling with product images as extra data
7. Class-weighted loss for underrepresented categories

**Heavy lift:**
8. YOLOv8l or YOLOv8x — larger model, manage VRAM
9. Ensemble M + L predictions (within 420MB weight limit)
10. SAHI (Sliced Inference) — tile large images for small objects

## Gotchas & Lessons Learned

| # | Lesson | Date |
|---|--------|------|
| 1 | PyTorch 2.6+ breaks ultralytics 8.1.0 — needs `weights_only=False` monkey-patch | 2026-03-19 |
| 2 | gcloud SCP drops on large files — 880MB COCO zip failed at 52%, needs retry | 2026-03-19 |
| 3 | gcloud.cmd not on PATH — use full path to `gcloud.cmd` | 2026-03-19 |
| 4 | Dataloader workers (8) appear as separate python processes — don't kill them | 2026-03-19 |
| 5 | Training on T4 (14.6GB) but sandbox uses L4 (24GB) — extra VRAM at inference | 2026-03-19 |
| 6 | Only 210 train images for 356 classes — heavy augmentation essential, watch overfit | 2026-03-19 |

## Competition Links & References
- Sandbox: ultralytics 8.1.0, torch 2.6.0+cu124, onnxruntime-gpu 1.20.0
- Entry point: `python run.py --input /data/images --output /output/predictions.json`
- Metric: mAP (COCO-style, likely mAP50-95)
