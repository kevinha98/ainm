"""
New prediction approach: train a regressor on Round 1 GT distributions.
The ground truth is NOT one-hot — it's a probability distribution over 6 classes.
The simulation is stochastic, and GT represents the empirical distribution over many runs.

Strategy: Train a model that maps (initial_grid_features) -> 6-class probability distribution.
Since we have Round 1 GT (5 seeds × 1600 cells = 8000 samples), train on that.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from src.settings import GRID_TO_CLASS, CLASS_NAMES, NUM_CLASSES, DATA_DIR
from src.models import extract_features, build_class_grid

def load_round1_training_data():
    """Load Round 1 ground truth as training data."""
    with open(DATA_DIR / "ground_truth_71451d74.json") as f:
        gt_data = json.load(f)
    with open(DATA_DIR / "round_info.json") as f:
        rd = json.load(f)
    
    X_all = []
    Y_all = []
    
    for si in range(5):
        gt = np.array(gt_data[str(si)]["ground_truth"])  # 40x40x6 probability distribution
        grid = np.array(rd["initial_states"][si]["grid"])
        cls_grid = build_class_grid(grid)
        
        # Find settlements and ports
        settle_set = set(map(tuple, np.argwhere(cls_grid == 1).tolist()))
        port_set = set(map(tuple, np.argwhere(cls_grid == 2).tolist()))
        
        # Extract features (same as our HGB model)
        X = extract_features(grid, settle_set, port_set)  # (1600, n_features)
        Y = gt.reshape(-1, 6)  # (1600, 6) -- the probability distribution
        
        X_all.append(X)
        Y_all.append(Y)
    
    return np.vstack(X_all), np.vstack(Y_all)


def train_distribution_model(X, Y):
    """Train one regressor per class to predict the probability distribution."""
    models = []
    for c in range(NUM_CLASSES):
        print(f"  Training class {c} ({CLASS_NAMES[c]})...")
        model = HistGradientBoostingRegressor(
            max_iter=300,
            max_depth=6,
            learning_rate=0.05,
            min_samples_leaf=20,
            random_state=42,
        )
        model.fit(X, Y[:, c])
        
        # Training MSE
        pred_c = model.predict(X)
        mse = np.mean((pred_c - Y[:, c]) ** 2)
        print(f"    MSE: {mse:.6f}, mean_target: {Y[:, c].mean():.4f}")
        models.append(model)
    
    return models


def predict_distribution(models, X):
    """Predict 6-class probability distribution for each cell."""
    n = X.shape[0]
    pred = np.zeros((n, NUM_CLASSES))
    for c in range(NUM_CLASSES):
        pred[:, c] = models[c].predict(X)
    
    # Ensure valid probability distribution
    pred = np.clip(pred, 0.001, None)
    pred /= pred.sum(axis=-1, keepdims=True)
    return pred


def model_distribution_regressor(grid_np, settle_set, port_set, models):
    """Use trained distribution models to predict GT distributions."""
    X = extract_features(grid_np, settle_set, port_set)
    pred = predict_distribution(models, X)
    H, W = grid_np.shape
    return pred.reshape(H, W, NUM_CLASSES)


if __name__ == "__main__":
    print("Loading Round 1 training data...")
    X, Y = load_round1_training_data()
    print(f"  X: {X.shape}, Y: {Y.shape}")
    
    print("\nTraining distribution models...")
    models = train_distribution_model(X, Y)
    
    # Cross-validation: train on 4 seeds, predict 5th
    print("\n=== Leave-One-Out Cross-Validation ===")
    
    with open(DATA_DIR / "ground_truth_71451d74.json") as f:
        gt_data = json.load(f)
    with open(DATA_DIR / "round_info.json") as f:
        rd = json.load(f)
    
    def score_pred(pred, gt):
        """Score using our hypothesized KL formula."""
        pred = np.clip(pred, 1e-6, None)
        pred /= pred.sum(axis=-1, keepdims=True)
        kl = np.sum(gt * np.log((gt + 1e-15) / (pred + 1e-15)), axis=-1).mean()
        return 100 * np.exp(-kl)
    
    for test_seed in range(5):
        # Train on other seeds
        X_train, Y_train = [], []
        for si in range(5):
            if si == test_seed:
                continue
            gt = np.array(gt_data[str(si)]["ground_truth"])
            grid = np.array(rd["initial_states"][si]["grid"])
            cls_grid = build_class_grid(grid)
            settle_set = set(map(tuple, np.argwhere(cls_grid == 1).tolist()))
            port_set = set(map(tuple, np.argwhere(cls_grid == 2).tolist()))
            X_s = extract_features(grid, settle_set, port_set)
            Y_s = gt.reshape(-1, 6)
            X_train.append(X_s)
            Y_train.append(Y_s)
        
        X_train = np.vstack(X_train)
        Y_train = np.vstack(Y_train)
        
        # Train
        cv_models = []
        for c in range(NUM_CLASSES):
            m = HistGradientBoostingRegressor(
                max_iter=300, max_depth=6, learning_rate=0.05,
                min_samples_leaf=20, random_state=42
            )
            m.fit(X_train, Y_train[:, c])
            cv_models.append(m)
        
        # Predict test seed
        gt_test = np.array(gt_data[str(test_seed)]["ground_truth"])
        grid_test = np.array(rd["initial_states"][test_seed]["grid"])
        cls_test = build_class_grid(grid_test)
        settle_test = set(map(tuple, np.argwhere(cls_test == 1).tolist()))
        port_test = set(map(tuple, np.argwhere(cls_test == 2).tolist()))
        X_test = extract_features(grid_test, settle_test, port_test)
        
        pred_test = predict_distribution(cv_models, X_test).reshape(40, 40, 6)
        
        score = score_pred(pred_test, gt_test)
        
        # Compare with naive strategies 
        uniform = np.ones_like(gt_test) / 6
        score_uniform = score_pred(uniform, gt_test)
        
        # Also: what if we just use the average GT from training?
        avg_gt_train = Y_train.mean(axis=0)
        avg_pred = np.tile(avg_gt_train, (40, 40, 1))
        score_avg = score_pred(avg_pred, gt_test)
        
        actual = gt_data[str(test_seed)]["score"]
        print(f"  Seed {test_seed}: regressor={score:.1f}, uniform={score_uniform:.1f}, avg_dist={score_avg:.1f}, actual_R1={actual:.1f}")
