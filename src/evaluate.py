"""
evaluate.py
───────────
Model evaluation utilities: metrics, calibration, SHAP, threshold optimization.

All functions are stateless — they take fitted model + data, return results.
Nothing is logged to MLflow here — that happens in train.py so the
experiment context is controlled in one place.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for CI
import matplotlib.pyplot as plt
import shap

from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
)
from src.config import FN_COST, FP_COST, ARTIFACTS_DIR

logger = logging.getLogger(__name__)


# ── Core metrics ───────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Computes the full metric suite required for credit risk model evaluation.

    Why these metrics and not accuracy:
        Class distribution is 84.5/15.5. A model predicting all high-risk
        achieves 84.5% accuracy while being completely useless.
        F1, PR-AUC, and ROC-AUC are invariant to class distribution.

    Returns dict of metric_name → float, ready for mlflow.log_metrics().
    """
    metrics = {
        "roc_auc"      : roc_auc_score(y_true, y_prob),
        "pr_auc"       : average_precision_score(y_true, y_prob),
        "f1"           : f1_score(y_true, y_pred, zero_division=0),
        "precision"    : precision_score(y_true, y_pred, zero_division=0),
        "recall"       : recall_score(y_true, y_pred, zero_division=0),
        "f1_low_risk"  : f1_score(y_true, y_pred, pos_label=0, zero_division=0),
    }

    logger.info(
        f"Metrics:\n"
        f"  ROC-AUC : {metrics['roc_auc']:.4f}\n"
        f"  PR-AUC  : {metrics['pr_auc']:.4f}\n"
        f"  F1      : {metrics['f1']:.4f}\n"
        f"  Recall  : {metrics['recall']:.4f}\n"
        f"  F1(low) : {metrics['f1_low_risk']:.4f}"
    )
    logger.info(
        f"\nClassification report:\n"
        f"{classification_report(y_true, y_pred, zero_division=0)}"
    )
    return metrics


# ── Calibration ────────────────────────────────────────────────────────────

def calibrate_model(
    model,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    method: str = "isotonic",
):
    """
    Wraps an already-fitted model in CalibratedClassifierCV, calibrating
    on the (disjoint) validation set without re-fitting the base model.

    Why calibration is mandatory for financial models:
        Raw predict_proba from GBM is not a true probability.
        A model outputting P=0.9 that's only right 55% of the time
        causes direct financial loss through mispriced expected loss.
        Calibration aligns output probabilities with observed frequencies.

    Implementation note:
        `cv="prefit"` was deprecated in sklearn 1.6 and removed in 1.8.
        The replacement is to wrap the fitted model in FrozenEstimator,
        which signals "this estimator is already fit — do not refit it
        via cross-validation," exactly what cv="prefit" used to mean.
        With FrozenEstimator, CalibratedClassifierCV.fit() uses all of
        X_val/y_val purely for calibration (no internal CV splitting),
        so the caller is still responsible for ensuring X_val/y_val is
        disjoint from the data the base model was trained on.

    method='isotonic' for GBM (flexible, handles non-monotonic distortion)
    method='sigmoid'  for LR  (Platt scaling — appropriate for linear models)
    """
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(model), method=method
    )
    calibrated.fit(X_val, y_val)
    logger.info(f"Model calibrated using {method} method")
    return calibrated


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error — measures how well predicted probabilities
    match observed frequencies.

    ECE = 0.0  → perfect calibration
    ECE > 0.05 → model fails production calibration bar for financial use
    """
    fraction_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    ece = float(np.mean(np.abs(fraction_pos - mean_pred)))
    logger.info(f"Expected Calibration Error (ECE): {ece:.4f}")
    return ece


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob_before: np.ndarray,
    y_prob_after: np.ndarray,
    model_name: str,
    save_path: Path,
) -> Path:
    """Plots reliability diagram before and after calibration."""
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    for probs, label in [
        (y_prob_before, f"{model_name} (uncalibrated)"),
        (y_prob_after,  f"{model_name} (calibrated)"),
    ]:
        fraction_pos, mean_pred = calibration_curve(
            y_true, probs, n_bins=10, strategy="uniform"
        )
        ax.plot(mean_pred, fraction_pos, marker="o", label=label)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration curve — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    logger.info(f"Calibration curve saved to {save_path}")
    return save_path


# ── Threshold optimization ─────────────────────────────────────────────────

def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fn_cost: float = FN_COST,
    fp_cost: float = FP_COST,
) -> tuple[float, float]:
    """
    Finds the decision threshold that minimizes expected business cost.

    In credit risk:
        False Negative (FN): approving a customer who defaults
            → lose principal + interest → high cost (FN_COST=15)
        False Positive (FP): rejecting a customer who would have repaid
            → lose interest revenue → low cost (FP_COST=1)

    Default 0.5 threshold ignores this asymmetry entirely.
    This function finds the threshold that minimizes total expected cost.

    Returns (optimal_threshold, minimum_cost)
    """
    thresholds  = np.arange(0.01, 1.0, 0.01)
    costs       = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()
        total_cost = (fn * fn_cost) + (fp * fp_cost)
        costs.append(total_cost)

    optimal_idx       = int(np.argmin(costs))
    optimal_threshold = float(thresholds[optimal_idx])
    minimum_cost      = float(costs[optimal_idx])

    # Operational floor: thresholds below 0.30 represent a near-universal
    # reject policy — not a functioning credit model. With 84.5% positive
    # class and asymmetric costs, the pure cost optimizer drives the
    # threshold to near-zero. We constrain to a minimum that preserves
    # meaningful discrimination.
    THRESHOLD_FLOOR   = 0.30
    if optimal_threshold < THRESHOLD_FLOOR:
        logger.warning(
            f"Cost-optimal threshold {optimal_threshold:.2f} is below "
            f"operational floor {THRESHOLD_FLOOR} — constraining to floor. "
            f"This reflects the class imbalance (84.5% positive) interacting "
            f"with asymmetric FN/FP costs. Document in README."
        )
        optimal_threshold = THRESHOLD_FLOOR

    logger.info(
        f"Threshold optimization:\n"
        f"  FN cost weight : {fn_cost}x\n"
        f"  FP cost weight : {fp_cost}x\n"
        f"  Optimal threshold : {optimal_threshold:.2f}\n"
        f"  Expected cost at threshold : {minimum_cost:.0f}"
    )
    return optimal_threshold, minimum_cost


# ── SHAP explainability ────────────────────────────────────────────────────

def compute_shap_values(
    model,
    X: pd.DataFrame,
    model_name: str,
    save_path: Path,
) -> np.ndarray:
    """
    Computes SHAP values and saves global summary plot.

    Uses TreeExplainer for GBM (fast, exact).
    Uses LinearExplainer for LR (exact for linear models).
    Falls back to KernelExplainer if neither applies (slow — samples 100 rows).

    The summary plot is saved to reports/ and logged as MLflow artifact.
    It is also included in the README as the explainability screenshot.
    """
    try:
        if "logistic" in model_name.lower():
            explainer   = shap.LinearExplainer(model, X)
        else:
            explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    except Exception:
        logger.warning("TreeExplainer/LinearExplainer failed — using KernelExplainer (slow)")
        explainer   = shap.KernelExplainer(model.predict_proba, X.sample(100))
        shap_values = explainer.shap_values(X)

    # For binary classification, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]   # use class 1 (high-risk) values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP summary plot saved to {save_path}")
    return shap_values