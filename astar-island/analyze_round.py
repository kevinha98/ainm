"""
Astar Island — Post-Round Analysis
====================================
After a round completes:
  1. Download ground truth via /analysis/{round_id}/{seed}
  2. Compute actual transition matrix from initial -> final state
  3. Compare our predictions vs reality
  4. Save learned transitions for future rounds
  5. Update leaderboard data

Usage:
    python analyze_round.py                # Analyze latest completed round
    python analyze_round.py <round_id>     # Analyze specific round
"""
import json, sys
import numpy as np
from pathlib import Path
from datetime import datetime

from src.settings import DATA_DIR, NUM_CLASSES, MAP_H, MAP_W, GRID_TO_CLASS, CLASS_NAMES
from src.api import AstarAPI
from src.models import build_class_grid


def download_ground_truth(api, round_id, n_seeds):
    """Download final grids for all seeds from /analysis endpoint."""
    truth = {}
    for si in range(n_seeds):
        try:
            data = api.get_analysis(round_id, si)
            truth[si] = data
            print(f"  Seed {si}: OK ({len(str(data))} bytes)")
        except Exception as e:
            print(f"  Seed {si}: FAILED ({e})")

    if truth:
        path = DATA_DIR / f"ground_truth_{round_id[:8]}.json"
        with open(path, "w") as f:
            json.dump({str(k): v for k, v in truth.items()}, f)
        print(f"  Saved to {path}")

    return truth


def compute_transition_matrix(initial_grids, final_grids):
    """Learn real transition probabilities from initial -> final state.
    Returns NxN matrix where T[i,j] = P(final_class=j | initial_class=i)."""
    counts = np.zeros((NUM_CLASSES, NUM_CLASSES))

    for si in range(len(initial_grids)):
        if si not in final_grids:
            continue
        init = initial_grids[si]
        final_data = final_grids[si]

        # The analysis endpoint may return different structures
        if isinstance(final_data, dict):
            if "grid" in final_data:
                final = np.array(final_data["grid"])
            elif "final_grid" in final_data:
                final = np.array(final_data["final_grid"])
            else:
                print(f"  Seed {si}: unknown analysis format, keys={list(final_data.keys())}")
                continue
        elif isinstance(final_data, list):
            final = np.array(final_data)
        else:
            continue

        init_cls = build_class_grid(init)
        final_cls = build_class_grid(final)

        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                counts[i, j] += np.sum((init_cls == i) & (final_cls == j))

    # Normalize
    row_sums = counts.sum(axis=1, keepdims=True)
    T = np.where(row_sums > 0, counts / row_sums, 1.0 / NUM_CLASSES)
    return T, counts


def evaluate_predictions(predictions, final_grids, initial_grids):
    """Score our predictions against ground truth."""
    results = {}

    for si in range(len(initial_grids)):
        if si not in final_grids:
            continue

        final_data = final_grids[si]
        if isinstance(final_data, dict):
            final = np.array(final_data.get("grid", final_data.get("final_grid", [])))
        else:
            final = np.array(final_data)

        final_cls = build_class_grid(final)

        # Find our prediction for this seed
        pred_found = False
        for p in predictions:
            if p["seed_index"] == si:
                probs = np.array(p["probabilities"])  # H x W x 6

                # Log-loss
                true_probs = probs[np.arange(MAP_H)[:, None], np.arange(MAP_W)[None, :], final_cls]
                ll = -np.log(np.clip(true_probs, 1e-15, None)).mean()

                # Accuracy (argmax match)
                pred_cls = np.argmax(probs, axis=-1)
                acc = (pred_cls == final_cls).mean()

                # Per-class accuracy
                per_class = {}
                for c in range(NUM_CLASSES):
                    mask = final_cls == c
                    if mask.any():
                        per_class[CLASS_NAMES[c]] = float((pred_cls[mask] == c).mean())

                results[si] = {
                    "log_loss": float(ll),
                    "accuracy": float(acc),
                    "per_class_accuracy": per_class,
                    "n_changed": int((build_class_grid(initial_grids[si]) != final_cls).sum()),
                    "n_total": MAP_H * MAP_W,
                }
                pred_found = True
                break

        if not pred_found:
            print(f"  Seed {si}: no prediction found!")

    return results


def merge_learned_transitions(new_T, new_counts):
    """Merge new transition data with any existing learned transitions."""
    learned_file = DATA_DIR / "learned_transitions.json"
    if learned_file.exists():
        with open(learned_file) as f:
            existing = json.load(f)
        old_counts = np.array(existing["counts"])
        total_counts = old_counts + new_counts
    else:
        total_counts = new_counts

    # Normalize
    row_sums = total_counts.sum(axis=1, keepdims=True)
    T = np.where(row_sums > 0, total_counts / row_sums, 1.0 / NUM_CLASSES)

    data = {
        "matrix": T.tolist(),
        "counts": total_counts.tolist(),
        "total_cells": int(total_counts.sum()),
        "updated_at": datetime.now().isoformat(),
    }
    with open(learned_file, "w") as f:
        json.dump(data, f, indent=2)

    return T, total_counts


def main():
    api = AstarAPI()
    print("=" * 60)
    print("  ASTAR ISLAND -- Post-Round Analysis")
    print("=" * 60)

    # Find round to analyze
    if len(sys.argv) > 1:
        round_id = sys.argv[1]
    else:
        # Find latest completed round
        my_rounds = api.get_my_rounds()
        completed = [r for r in my_rounds if r["status"] == "completed"]
        if not completed:
            print("\n  No completed rounds yet. Checking status...")
            for r in my_rounds:
                print(f"  Round {r['round_number']}: {r['status']} (score={r.get('round_score')})")
            return
        round_id = completed[-1]["id"]

    # Get round info
    round_info = None
    for r in api.get_my_rounds():
        if r["id"] == round_id:
            round_info = r
            break
    if round_info is None:
        print(f"  Round {round_id} not found!")
        return

    print(f"\n  Round {round_info['round_number']}: {round_info['status']}")
    print(f"  Score: {round_info.get('round_score')}")
    print(f"  Rank: {round_info.get('rank')}")
    n_seeds = round_info.get("seeds_count", 5)

    if round_info["status"] != "completed":
        print(f"  Round not yet completed (status={round_info['status']})")
        print(f"  Scores: {round_info.get('seed_scores')}")
        return

    # Download ground truth
    print("\n[1] DOWNLOADING GROUND TRUTH")
    truth = download_ground_truth(api, round_id, n_seeds)
    if not truth:
        print("  No ground truth available!")
        return

    # Load initial grids
    print("\n[2] LOADING INITIAL STATE")
    try:
        full_round = api._get(f"/rounds/{round_id}")
        initial_grids = [np.array(st["grid"]) for st in full_round["initial_states"]]
    except Exception:
        # Try from saved data
        with open(DATA_DIR / "round_info.json") as f:
            saved = json.load(f)
        initial_grids = [np.array(st["grid"]) for st in saved["initial_states"]]

    # Compute transition matrix
    print("\n[3] COMPUTING TRANSITION MATRIX")
    T, counts = compute_transition_matrix(initial_grids, truth)
    print("\n  Learned Transition Matrix (initial -> final after 50 years):")
    print(f"  {'':>20}", end="")
    for c in range(NUM_CLASSES):
        print(f"{CLASS_NAMES[c]:>12}", end="")
    print()
    for i in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[i]:>20}", end="")
        for j in range(NUM_CLASSES):
            print(f"{T[i, j]:>12.3f}", end="")
        print(f"  (n={int(counts[i].sum())})")

    # Merge with existing
    T_merged, counts_merged = merge_learned_transitions(T, counts)
    print(f"\n  Total cells learned from: {int(counts_merged.sum())}")

    # Evaluate our predictions
    print("\n[4] EVALUATING PREDICTIONS")
    try:
        with open(DATA_DIR / "improved_predictions.json") as f:
            predictions = json.load(f)
        results = evaluate_predictions(predictions, truth, initial_grids)
        for si, r in results.items():
            print(f"\n  Seed {si}:")
            print(f"    Log-loss:  {r['log_loss']:.4f}")
            print(f"    Accuracy:  {r['accuracy']:.4f}")
            print(f"    Changed:   {r['n_changed']}/{r['n_total']} ({r['n_changed']/r['n_total']*100:.1f}%)")
            for cls, acc in r['per_class_accuracy'].items():
                print(f"    {cls:>20}: {acc:.3f}")

        # Save results
        analysis_file = DATA_DIR / f"analysis_{round_id[:8]}.json"
        with open(analysis_file, "w") as f:
            json.dump({
                "round_id": round_id,
                "analyzed_at": datetime.now().isoformat(),
                "transition_matrix": T.tolist(),
                "evaluation": {str(k): v for k, v in results.items()},
            }, f, indent=2)
        print(f"\n  Saved analysis to {analysis_file}")

    except FileNotFoundError:
        print("  No predictions file found to evaluate")

    # Check leaderboard
    print("\n[5] LEADERBOARD")
    try:
        lb = api.get_leaderboard()
        if lb:
            for entry in lb[:10]:
                print(f"  #{entry.get('rank', '?')} {entry.get('team_name', '?')}: {entry.get('weighted_score', '?')}")
        else:
            print("  Empty")
    except Exception:
        print("  Not available")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
