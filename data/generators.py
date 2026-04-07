# data/generators.py
# Synthetic dataset generators for all 3 tasks.
# ALL generators use fixed seeds — results are identical every run.
# Each generator includes _validate_*() assertions to catch regressions.

import numpy as np
from scipy import stats as scipy_stats
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — Easy: Single Bug (Label Flipping ~30%)
# Baseline acc WITH bug:    ~0.52–0.56  (near random)
# Expected acc AFTER fix:   ~0.83–0.88
# ══════════════════════════════════════════════════════════════════════════════

def generate_easy_task_data():
    """
    Binary classification with 30% training labels flipped.

    Returns:
        X_train       (800, 10)
        X_test        (200, 10)
        y_train_clean (800,)   — correct labels (used by grader)
        y_train_buggy (800,)   — 30% labels flipped (what agent sees)
        y_test        (200,)
    """
    np.random.seed(42)

    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        flip_y=0.01,        # small inherent noise — realistic
        class_sep=1.2,      # clear separation so fixed model hits >83%
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_train_buggy = y_train.copy()
    n_flip = int(0.30 * len(y_train_buggy))
    flip_idx = np.random.choice(len(y_train_buggy), n_flip, replace=False)
    y_train_buggy[flip_idx] = 1 - y_train_buggy[flip_idx]

    _validate_easy(y_train, y_train_buggy)
    return X_train, X_test, y_train.copy(), y_train_buggy, y_test


def _validate_easy(y_clean, y_buggy):
    flip_rate = (y_clean != y_buggy).mean()
    assert 0.28 <= flip_rate <= 0.32, f"Expected ~30% flip, got {flip_rate:.2%}"


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Medium: 3 Simultaneous Bugs
#
# Bug 1: Class imbalance  (90/10 split)
# Bug 2: Partial bad normalization — only first 5 features scaled ×50
#         (more realistic than all features ×100 — requires real diagnosis)
# Bug 3: Learning rate 100× too high (1.0 instead of 0.01)
#
# Baseline val_acc WITH all bugs:       ~0.50–0.62
# Expected val_acc WITH all bugs fixed: ~0.72–0.82
# ══════════════════════════════════════════════════════════════════════════════

def generate_medium_task_data():
    """
    Tabular dataset with 3 simultaneous, distinct bugs.

    Bug 2 is realistic: only features [:5] are over-scaled (×50),
    not all features. Agent must inspect to notice the skew.

    Returns:
        X_train_buggy  (800, 15) — first 5 features ×50
        X_train_clean  (800, 15) — all features at correct scale
        X_test         (200, 15) — correctly scaled test set
        y_train        (800,)    — imbalanced (~90/10)
        y_test         (200,)
        config_buggy   dict      — learning_rate=1.0, class_weight=None
        config_clean   dict      — learning_rate=0.01, class_weight="balanced"
    """
    np.random.seed(42)

    X, y = make_classification(
        n_samples=1000,
        n_features=15,
        n_informative=6,
        n_redundant=3,
        weights=[0.9, 0.1],     # Bug 1: severe class imbalance
        class_sep=0.8,           # moderate — task is meant to be hard
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Bug 2: Scale only the first 5 features by 50×
    # This is realistic — looks like a unit mismatch (km vs m, USD vs cents)
    X_train_buggy = X_train.copy()
    X_train_buggy[:, :5] *= 50.0
    X_train_clean = X_train.copy()

    # Scale test set correctly so grader is fair
    scaler = StandardScaler()
    scaler.fit(X_train_clean)
    X_test_scaled = scaler.transform(X_test)

    config_buggy = {
        "learning_rate": 1.0,        # Bug 3: 100× too high
        "batch_size": 32,
        "epochs": 50,
        "hidden_layers": [64, 32],
        "class_weight": None          # Bug 1 effect: no correction applied
    }
    config_clean = {
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 50,
        "hidden_layers": [64, 32],
        "class_weight": "balanced"
    }

    _validate_medium(y_train, X_train_buggy, X_train_clean)
    return X_train_buggy, X_train_clean, X_test_scaled, y_train, y_test, config_buggy, config_clean


def _validate_medium(y_train, X_buggy, X_clean):
    minority_frac = (y_train == 1).mean()
    assert minority_frac <= 0.15, f"Expected <15% minority, got {minority_frac:.2%}"

    # Only first 5 features should be over-scaled
    buggy_cols_ratio = abs(X_buggy[:, :5].mean()) / (abs(X_clean[:, :5].mean()) + 1e-9)
    clean_cols_ratio = abs(X_buggy[:, 5:].mean()) / (abs(X_clean[:, 5:].mean()) + 1e-9)
    assert buggy_cols_ratio > 10, f"First 5 cols should be ~50× scaled, ratio={buggy_cols_ratio:.1f}"
    assert clean_cols_ratio < 5,  f"Last 10 cols should be unscaled, ratio={clean_cols_ratio:.1f}"


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 — Hard: Silent Distribution Shift
#
# Train is StandardScaler-normalised. Test is raw (un-normalised).
# Model trains well (~85%+) but fails on raw test (~52–58%).
# No single obvious metric reveals this — agent must compare distributions.
#
# Observable signal (unlocked by inspect_data):
#   train mean ≈ 0.0,  test_raw mean ≈ original feature mean (~non-zero)
#   train std  ≈ 1.0,  test_raw std  ≈ 1.8–2.0
# ══════════════════════════════════════════════════════════════════════════════

def generate_hard_task_data():
    """
    Dataset with a silent train/test distribution shift.

    Returns:
        X_train_scaled  (900, 20)  — normalised training features
        X_test_raw      (300, 20)  — un-normalised test (the bug)
        X_test_scaled   (300, 20)  — normalised test (used by grader)
        y_train         (900,)
        y_test          (300,)
        scaler          StandardScaler — fitted on train (agent must apply to test)
        shift_stats     dict        — quantifies the shift for inspect_data signal
    """
    np.random.seed(42)

    X, y = make_classification(
        n_samples=1200,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        class_sep=1.5,      # high sep — model SHOULD train well on scaled data
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # train is normalised
    X_test_raw     = X_test.copy()                  # test is NOT normalised ← bug
    X_test_scaled  = scaler.transform(X_test)       # correct version for grader

    shift_stats = _compute_shift_stats(X_train_scaled, X_test_raw, X_test_scaled)

    _validate_hard(X_train_scaled, X_test_raw, X_test_scaled)
    return X_train_scaled, X_test_raw, X_test_scaled, y_train, y_test, scaler, shift_stats


def _compute_shift_stats(X_train, X_test_raw, X_test_scaled):
    """
    Pre-computed distribution shift statistics.
    Returned as part of the observation when agent calls inspect_data on Task 3.
    Gives a clear signal WITHOUT revealing the fix.
    """
    return {
        # Global stats — obvious difference between train and test_raw
        "train_feature_mean":    round(float(X_train.mean()), 4),
        "train_feature_std":     round(float(X_train.std()), 4),
        "test_feature_mean":     round(float(X_test_raw.mean()), 4),
        "test_feature_std":      round(float(X_test_raw.std()), 4),

        # Per-feature mean difference — strongest signal
        "mean_delta_per_feature": round(
            float(np.abs(X_train.mean(axis=0) - X_test_raw.mean(axis=0)).mean()), 4
        ),
        "std_delta_per_feature":  round(
            float(np.abs(X_train.std(axis=0) - X_test_raw.std(axis=0)).mean()), 4
        ),

        # Kolmogorov–Smirnov stat on first feature (strong drift indicator)
        "ks_stat_feature_0": round(
            float(scipy_stats.ks_2samp(X_train[:, 0], X_test_raw[:, 0]).statistic), 4
        ),

        # For grader: ground truth
        "shift_detected": True,
        "fix_method": "apply StandardScaler fitted on train to test set"
    }


def _validate_hard(X_train, X_test_raw, X_test_scaled):
    assert abs(X_train.mean()) < 0.05,      "Train mean should be ~0 after scaling"
    assert abs(X_train.std() - 1.0) < 0.05, "Train std should be ~1 after scaling"
    assert abs(X_test_raw.std() - 1.0) > 0.2, "Test raw std should differ from 1.0"
    assert abs(X_test_scaled.mean()) < 0.1,    "Scaled test mean should be ~0"


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY — Shared helpers used by all task files
# ══════════════════════════════════════════════════════════════════════════════

def get_data_summary(X, y, top_n_features=3):
    """
    Rich dataset statistics for Observation.data_summary.
    Includes skewness and per-feature stats on top_n most extreme features
    — gives the agent real diagnostic signal without overwhelming it.
    """
    unique, counts = np.unique(y, return_counts=True)
    class_dist = {str(int(k)): int(v) for k, v in zip(unique, counts)}
    minority_frac = float(counts.min() / counts.sum())

    # Per-feature means — surface the most extreme ones (normalization clue)
    feature_means = X.mean(axis=0)
    feature_stds  = X.std(axis=0)
    top_idx = np.argsort(np.abs(feature_means))[::-1][:top_n_features]
    top_features = {
        f"feature_{i}": {
            "mean": round(float(feature_means[i]), 4),
            "std":  round(float(feature_stds[i]), 4),
        }
        for i in top_idx
    }

    # Global distribution shape
    flat = X.flatten()
    skewness = round(float(scipy_stats.skew(flat)), 4)
    kurtosis = round(float(scipy_stats.kurtosis(flat)), 4)

    return {
        "n_samples":           int(X.shape[0]),
        "n_features":          int(X.shape[1]),
        "class_distribution":  class_dist,
        "minority_fraction":   round(minority_frac, 4),
        "feature_mean_global": round(float(np.mean(X)), 4),
        "feature_std_global":  round(float(np.std(X)), 4),
        "feature_min":         round(float(np.min(X)), 4),
        "feature_max":         round(float(np.max(X)), 4),
        "skewness":            skewness,
        "kurtosis":            kurtosis,
        "top_extreme_features": top_features,  # most diagnostic for scaling bugs
        "null_count":          int(np.isnan(X).sum()),
    }


def get_baseline_scores():
    """
    Expected grader score ranges for a random/no-fix agent.
    Used in README baseline comparison table and inference.py scoring context.
    """
    return {
        "task_easy":   {"min": 0.02, "max": 0.12, "note": "near-random due to label noise"},
        "task_medium": {"min": 0.10, "max": 0.25, "note": "imbalance + bad LR kills F1"},
        "task_hard":   {"min": 0.00, "max": 0.10, "note": "model trains well, test fails silently"},
    }
def generate_distribution_shift_data(n_train=500, n_test=200, n_features=10, seed=42):
    rng = np.random.RandomState(seed)

    # Raw train: mean=5, std=2 (intentionally NOT randn so stats are non-trivial)
    X_train_raw = rng.randn(n_train, n_features) * 2 + 5
    train_mean  = X_train_raw.mean(axis=0)   # ≈ 5
    train_std   = X_train_raw.std(axis=0) + 1e-8  # ≈ 2

    # Normalized train (what model is trained on)
    X_train = (X_train_raw - train_mean) / train_std   # mean≈0, std≈1

    # Labels on normalized features
    y_train = ((X_train[:, 0] + X_train[:, 1]) > 0).astype(np.float32)

    # Test raw: DIFFERENT mean (=15), SAME std (=2) → covariate shift
    # Model sees un-normalized test → features look wrong → bad accuracy
    X_test_raw    = rng.randn(n_test, n_features) * 2 + 15

    # Consistent labels (normalize first, apply same boundary)
    X_test_scaled = (X_test_raw - train_mean) / train_std
    y_test        = ((X_test_scaled[:, 0] + X_test_scaled[:, 1]) > 0).astype(np.float32)

    # Ground-truth fix: normalize test with train stats → std≈1, mean≈5
    X_test_normalized = (X_test_raw - train_mean) / train_std

    return {
        "X_train":           X_train.astype(np.float32),
        "y_train":           y_train,
        "X_test_raw":        X_test_raw.astype(np.float32),   # THE BUG
        "X_test_normalized": X_test_normalized.astype(np.float32),
        "y_test":            y_test,
        "train_mean":        train_mean.astype(np.float32),   # ≈ 5
        "train_std":         train_std.astype(np.float32),    # ≈ 2
    }