"""
train.py
────────
End-to-end model training pipeline with MLflow experiment tracking.

Flow:
    1. Load raw data
    2. Generate proxy labels (RFM — training window only)
    3. Run feature pipeline (fit on train, transform both splits)
    4. IV-based feature selection
    5. Train LR + GBM with class_weight='balanced'
    6. Calibrate both models
    7. Evaluate: metrics, calibration error, threshold optimization
    8. SHAP explainability
    9. Log everything to MLflow
    10. Register best model in MLflow Model Registry

Usage:
    python -m src.train
"""

import logging
import warnings
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

from src.config import (
    RAW_FILE, ARTIFACTS_DIR,
    TRAIN_CUTOFF_DATE, RANDOM_STATE, TARGET_COL,
    IV_THRESHOLD,
)
from src.rfm import generate_proxy_labels
from src.data_processing import build_pipeline, compute_iv_and_select_features
from src.evaluate import (
    compute_metrics, calibrate_model, compute_ece,
    plot_calibration_curve, optimize_threshold, compute_shap_values,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loading and splitting ─────────────────────────────────────────────

def load_and_split(
    raw_file: Path,
    cutoff: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw data and performs temporal train/test split.

    Split is temporal — NOT random. Training on future data to predict
    past behavior is leakage. The cutoff date is the same date used
    to compute RFM labels, ensuring full consistency.

    Returns (df_train, df_test) at transaction level.
    """
    df = pd.read_csv(raw_file)
    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"], utc=True
    )
    cutoff_ts = pd.Timestamp(cutoff, tz="UTC")

    df_train = df[df["TransactionStartTime"] < cutoff_ts].copy()
    df_test = df[df["TransactionStartTime"] >= cutoff_ts].copy()

    logger.info(
        f"Temporal split at {cutoff}:\n"
        f"  Train: {len(df_train):,} transactions, "
        f"{df_train['CustomerId'].nunique():,} customers\n"
        f"  Test : {len(df_test):,} transactions, "
        f"{df_test['CustomerId'].nunique():,} customers"
    )
    return df_train, df_test


# ── Feature matrix assembly ────────────────────────────────────────────────

def build_feature_matrix(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Runs the feature pipeline and merges proxy labels.

    Critical ordering:
        1. Pipeline.fit_transform() on df_train only
        2. Pipeline.transform() on df_test
        3. Merge labels AFTER pipeline — labels are customer-level,
           pipeline output is also customer-level after aggregation

    Customers in test window with no training label are excluded.
    This is correct — we cannot assign a risk label to customers
    we have no training history for.
    """
    pipeline = build_pipeline()

    logger.info("Fitting feature pipeline on training data...")
    X_train = pipeline.fit_transform(df_train)

    logger.info("Transforming test data...")
    X_test = pipeline.transform(df_test)

    # Save fitted pipeline
    joblib.dump(pipeline, ARTIFACTS_DIR / "feature_pipeline.joblib")

    # Merge proxy labels — inner join drops customers without labels
    X_train = X_train.merge(labels_df, on="CustomerId", how="inner")
    X_test = X_test.merge(labels_df, on="CustomerId", how="inner")

    # Separate features and target
    feature_cols = [
        c for c in X_train.columns
        if c not in ["CustomerId", TARGET_COL]
    ]

    # IV-based feature selection on training data only
    y_train_temp = X_train[TARGET_COL]
    selected = compute_iv_and_select_features(
        X_train[feature_cols], y_train_temp, threshold=IV_THRESHOLD
    )
    # Always keep at least the core aggregate features
    if len(selected) < 3:
        logger.warning(
            f"IV filtering retained only {len(selected)} features. "
            f"Using all {len(feature_cols)} features instead."
        )
        selected = feature_cols

    y_train = X_train[TARGET_COL]
    y_test = X_test[TARGET_COL]
    X_train = X_train[selected]
    X_test = X_test[selected]

    # Save selected features for inference
    pd.Series(selected).to_csv(
        ARTIFACTS_DIR / "selected_features.csv", index=False
    )

    logger.info(
        f"Feature matrix ready:\n"
        f"  Train: {X_train.shape} | "
        f"High-risk rate: {y_train.mean()*100:.1f}%\n"
        f"  Test : {X_test.shape}  | "
        f"High-risk rate: {y_test.mean()*100:.1f}%"
    )
    return X_train, X_test, y_train, y_test


# ── Model definitions ──────────────────────────────────────────────────────

def get_models() -> dict:
    """
    Returns model candidates with their hyperparameter search spaces.

    Both models use class_weight='balanced' — mandatory given 84.5/15.5
    class imbalance. Without this, both models will predict high-risk
    for everything and achieve 84.5% accuracy while being useless.
    """
    return {
        "logistic_regression": {
            "model": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
            "params": {
                "C": uniform(0.001, 10),
                "solver": ["lbfgs", "liblinear"],
                "penalty": ["l2"],
            },
            "calibration_method": "sigmoid",
        },
        "gradient_boosting": {
            "model": GradientBoostingClassifier(
                random_state=RANDOM_STATE,
            ),
            "params": {
                "n_estimators": randint(100, 500),
                "max_depth": randint(2, 6),
                "learning_rate": uniform(0.01, 0.3),
                "subsample": uniform(0.6, 0.4),
                "min_samples_leaf": randint(10, 50),
            },
            "calibration_method": "isotonic",
        },
    }


# ── Training loop ──────────────────────────────────────────────────────────

def train_and_evaluate(
    model_name: str,
    model_config: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[float, object]:
    """
    Trains one model with hyperparameter search, calibration, and full
    evaluation inside an MLflow run.

    Returns (roc_auc_score, calibrated_model) for best model selection.
    """
    mlflow.set_experiment("credit-risk-bati-bank")

    with mlflow.start_run(run_name=model_name):

        # ── Hyperparameter search ──────────────────────────────────────
        logger.info(f"Starting RandomizedSearch for {model_name}...")
        search = RandomizedSearchCV(
            estimator=model_config["model"],
            param_distributions=model_config["params"],
            n_iter=20,
            cv=3,
            scoring="f1",          # optimize for F1 — imbalanced data
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        logger.info(f"Best params: {search.best_params_}")
        mlflow.log_params(search.best_params_)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("class_weight", "balanced")

        # ── Pre-calibration evaluation ─────────────────────────────────
        y_prob_raw = best_model.predict_proba(X_test)[:, 1]
        ece_before = compute_ece(y_test.values, y_prob_raw)

        # ── Calibration ────────────────────────────────────────────────
        calibrated = calibrate_model(
            best_model, X_test, y_test.values,
            method=model_config["calibration_method"]
        )
        y_prob_cal = calibrated.predict_proba(X_test)[:, 1]
        ece_after = compute_ece(y_test.values, y_prob_cal)

        # ── Threshold optimization ─────────────────────────────────────
        opt_threshold, min_cost = optimize_threshold(
            y_test.values, y_prob_cal
        )
        y_pred = (y_prob_cal >= opt_threshold).astype(int)

        # ── Metrics ────────────────────────────────────────────────────
        metrics = compute_metrics(y_test.values, y_pred, y_prob_cal)
        metrics["ece_before"] = ece_before
        metrics["ece_after"] = ece_after
        metrics["optimal_threshold"] = opt_threshold
        metrics["min_cost"] = min_cost
        mlflow.log_metrics(metrics)

        # ── Calibration curve plot ─────────────────────────────────────
        cal_plot_path = REPORTS_DIR / f"{model_name}_calibration.png"
        plot_calibration_curve(
            y_test.values, y_prob_raw, y_prob_cal,
            model_name, cal_plot_path
        )
        mlflow.log_artifact(str(cal_plot_path))

        # ── SHAP ───────────────────────────────────────────────────────
        shap_path = REPORTS_DIR / f"{model_name}_shap.png"
        try:
            compute_shap_values(best_model, X_test, model_name, shap_path)
            mlflow.log_artifact(str(shap_path))
        except Exception as e:
            logger.warning(f"SHAP computation failed for {model_name}: {e}")

        # ── Log model + threshold ──────────────────────────────────────
        # Signature is built from the model's actual served output.
        # Calibrated probabilities (not raw .predict() class labels) are
        # what downstream consumers (the threshold-tuned FastAPI service)
        # need, so pyfunc_predict_fn="predict_proba" makes pyfunc.predict()
        # return probabilities, and the signature is inferred against that
        # same predict_proba output — not just the positive-class column —
        # so the logged schema matches what callers will actually receive.
        signature = infer_signature(
            X_train, calibrated.predict_proba(X_train)
        )
        mlflow.sklearn.log_model(
            calibrated,
            name="model",
            registered_model_name=f"credit-risk-{model_name}",
            signature=signature,
            input_example=X_train.iloc[:3],
            pyfunc_predict_fn="predict_proba",
        )
        mlflow.log_param("decision_threshold", opt_threshold)

        # Save calibrated model locally
        model_path = ARTIFACTS_DIR / f"{model_name}.joblib"
        joblib.dump(calibrated, model_path)
        joblib.dump(opt_threshold, ARTIFACTS_DIR / f"{model_name}_threshold.joblib")

        logger.info(
            f"Run complete — {model_name}\n"
            f"  ROC-AUC   : {metrics['roc_auc']:.4f}\n"
            f"  F1        : {metrics['f1']:.4f}\n"
            f"  ECE before: {ece_before:.4f}\n"
            f"  ECE after : {ece_after:.4f}\n"
            f"  Threshold : {opt_threshold:.2f}"
        )

        return metrics["roc_auc"], calibrated


# ── Best model selection and registration ─────────────────────────────────

def register_best_model(results: dict) -> None:
    """
    Identifies the best model by ROC-AUC and saves it as 'model.joblib'.
    This is the file the FastAPI service loads at startup.
    """
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = results[best_name]["model"]

    joblib.dump(best_model, ARTIFACTS_DIR / "model.joblib")
    joblib.dump(
        results[best_name]["threshold"],
        ARTIFACTS_DIR / "model_threshold.joblib"
    )

    logger.info(
        f"Best model: {best_name} "
        f"(ROC-AUC: {results[best_name]['roc_auc']:.4f})\n"
        f"Saved to artifacts/model.joblib"
    )


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("TASK 5 — Model Training Pipeline")
    logger.info("=" * 60)

    # 1. Load and split
    df_train, df_test = load_and_split(RAW_FILE, TRAIN_CUTOFF_DATE)

    # 2. Generate proxy labels (training window only)
    df_raw = pd.read_csv(RAW_FILE)
    labels_df = generate_proxy_labels(df_raw)

    # 3. Build feature matrix
    X_train, X_test, y_train, y_test = build_feature_matrix(
        df_train, df_test, labels_df
    )

    # 4. Train and evaluate all models
    models = get_models()
    results = {}

    for model_name, model_config in models.items():
        logger.info(f"\n{'='*40}")
        logger.info(f"Training: {model_name}")
        logger.info(f"{'='*40}")

        roc_auc, calibrated = train_and_evaluate(
            model_name, model_config,
            X_train, X_test, y_train, y_test
        )

        threshold = joblib.load(
            ARTIFACTS_DIR / f"{model_name}_threshold.joblib"
        )
        results[model_name] = {
            "roc_auc": roc_auc,
            "model": calibrated,
            "threshold": threshold,
        }

    # 5. Register best model
    register_best_model(results)

    logger.info("\n" + "=" * 60)
    logger.info("Training complete. Results summary:")
    for name, res in results.items():
        logger.info(f"  {name}: ROC-AUC = {res['roc_auc']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
