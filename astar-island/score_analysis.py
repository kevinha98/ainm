"""Compute what our predictions scored vs ground truth, and what improved models would score."""
import json
import numpy as np
from src.settings import DATA_DIR, NUM_CLASSES, GRID_TO_CLASS, CLASS_NAMES

# Load ground truth
with open(DATA_DIR / "ground_truth_71451d74.json") as f:
    truth = json.load(f)

# Load our submitted predictions
with open(DATA_DIR / "improved_predictions.json") as f:
    preds = json.load(f)

print("=" * 60)
print("  SCORE ANALYSIS: Our Predictions vs Ground Truth")
print("=" * 60)

for si in range(5):
    gt = np.array(truth[str(si)]["ground_truth"])  # H x W x 6 (one-hot)
    gt_cls = np.argmax(gt, axis=-1)

    pred = np.array(preds[si]["probabilities"])  # H x W x 6

    # Compute per-cell score: prob we assigned to the correct class
    correct_prob = pred[np.arange(40)[:, None], np.arange(40)[None, :], gt_cls]

    # Log-loss
    ll = -np.log(np.clip(correct_prob, 1e-15, None)).mean()

    # Accuracy
    pred_cls = np.argmax(pred, axis=-1)
    acc = (pred_cls == gt_cls).sum() / gt_cls.size

    # Average probability assigned to correct class
    avg_correct = correct_prob.mean()

    # Per-class analysis
    print(f"\n--- Seed {si} (official score: {truth[str(si)]['score']:.4f}) ---")
    print(f"  Log-loss: {ll:.4f}")
    print(f"  Accuracy: {acc:.1%}")
    print(f"  Avg prob on correct class: {avg_correct:.4f}")

    # Where did we fail hardest?
    worst = np.unravel_index(np.argmin(correct_prob), correct_prob.shape)
    print(f"  Worst cell: ({worst[0]},{worst[1]}) — pred={pred[worst]}, truth={CLASS_NAMES[gt_cls[worst]]}, prob={correct_prob[worst]:.6f}")

    # Per-class breakdown
    for c in range(NUM_CLASSES):
        mask = (gt_cls == c)
        if mask.sum() == 0:
            continue
        avg_p = correct_prob[mask].mean()
        n = mask.sum()
        pred_on_c = pred_cls == c
        true_pred = (pred_cls == c) & mask
        print(f"    {CLASS_NAMES[c]:>20s}: n={n:4d}, avg_prob_correct={avg_p:.4f}, predicted={pred_on_c.sum():4d}, correct={true_pred.sum():4d}")

# Try to reverse-engineer the scoring formula
print(f"\n\n--- SCORING FORMULA ANALYSIS ---")
scores = [truth[str(si)]["score"] for si in range(5)]
lls = []
accs = []
avg_probs = []
for si in range(5):
    gt = np.array(truth[str(si)]["ground_truth"])
    gt_cls = np.argmax(gt, axis=-1)
    pred = np.array(preds[si]["probabilities"])
    correct_prob = pred[np.arange(40)[:, None], np.arange(40)[None, :], gt_cls]
    lls.append(-np.log(np.clip(correct_prob, 1e-15, None)).mean())
    accs.append((np.argmax(pred, axis=-1) == gt_cls).mean())
    avg_probs.append(correct_prob.mean())

print(f"  {'Seed':>5s} | {'Score':>8s} | {'LogLoss':>8s} | {'Accuracy':>8s} | {'AvgProb':>8s} | {'100*AvgP':>8s}")
for si in range(5):
    print(f"  {si:>5d} | {scores[si]:>8.4f} | {lls[si]:>8.4f} | {accs[si]:>8.4f} | {avg_probs[si]:>8.4f} | {100*avg_probs[si]:>8.4f}")

# Test if score ≈ 100 * avg_prob
print(f"\n  Hypothesis: score = 100 * avg_correct_probability")
for si in range(5):
    print(f"    Seed {si}: actual={scores[si]:.4f}, 100*avg_prob={100*avg_probs[si]:.4f}, diff={scores[si] - 100*avg_probs[si]:.4f}")
