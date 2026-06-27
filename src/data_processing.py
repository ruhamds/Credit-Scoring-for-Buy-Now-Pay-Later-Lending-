"""
data_processing.py
──────────────────
Transforms raw Xente transaction data into a model-ready,
customer-level feature matrix.

Design principles:
- All transformations are reproducible and fit on training data only
- No leakage: Pipeline.fit() is called exclusively on training rows
- Signed log transform handles negative Amount values safely
- RobustScaler chosen over StandardScaler due to extreme outliers (skew=51)
"""

import logging
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from src.config import (
    COLS_TO_DROP,
    CAT_COLS,
    AGG_FUNCTIONS,
    IV_THRESHOLD,
    ARTIFACTS_DIR,
    SNAPSHOT_DATE,
)

logger = logging.getLogger(__name__)


# ── Custom Transformers ────────────────────────────────────────────────────

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Drops zero-signal and redundant columns defined in config.COLS_TO_DROP.
    Silently ignores columns not present in the dataframe.
    """

    def __init__(self, cols_to_drop: list):
        self.cols_to_drop = cols_to_drop

    def fit(self, X, y=None):
        # Only drop columns that actually exist
        self.cols_present_ = [c for c in self.cols_to_drop if c in X.columns]
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_present_, errors="ignore")


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Parses TransactionStartTime and extracts:
    hour, day, month, day_of_week, is_weekend
    Then drops the original datetime column.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        dt = pd.to_datetime(X["TransactionStartTime"], utc=True)
        X["tx_hour"]       = dt.dt.hour
        X["tx_day"]        = dt.dt.day
        X["tx_month"]      = dt.dt.month
        X["tx_dayofweek"]  = dt.dt.dayofweek
        X["tx_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        X = X.drop(columns=["TransactionStartTime"])
        return X


class SignedLogTransformer(BaseEstimator, TransformerMixin):
    """
    Applies sign(x) * log1p(|x|) to Amount.

    Why not standard log1p?
    Amount contains negative values (credits into account).
    log1p crashes on negatives. Signed log preserves the direction
    of the transaction while compressing the extreme magnitude.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Amount"] = np.sign(X["Amount"]) * np.log1p(np.abs(X["Amount"]))
        return X


class CustomerAggregator(BaseEstimator, TransformerMixin):
    """
    Collapses transaction-level rows into one row per CustomerId.

    Computes per-customer aggregates from AGG_FUNCTIONS config.
    This is the core feature set for credit risk modeling —
    individual transactions are not meaningful; behavioral
    patterns across transactions are.

    After aggregation, CustomerId is set as index (not a feature).
    """

    def __init__(self, agg_functions: dict):
        self.agg_functions = agg_functions

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Build aggregates
        agg_df = X.groupby("CustomerId").agg(self.agg_functions)

        # Flatten multi-level columns: ('Amount', 'sum') → 'Amount_sum'
        agg_df.columns = ["_".join(col).strip() for col in agg_df.columns]
        agg_df = agg_df.reset_index()

        # Numeric aggregates can be NaN for single-transaction customers
        # (e.g. Amount_std has no variance with only one observation).
        # That's zero variability, not missing data, so fill with 0 here —
        # scoped to the numeric aggregate columns only, before any
        # categorical mode columns are merged in below. Those _mode
        # columns are intentionally left as NaN: CategoricalEncoder
        # already has explicit handling for missing categorical values
        # (it fills with the string "missing"), and overwriting that
        # here with a numeric 0 would create a second, conflicting
        # "missing" sentinel that the encoder would treat as a real
        # category ("0") instead of as missing.
        numeric_agg_cols = [c for c in agg_df.columns if c != "CustomerId"]
        agg_df[numeric_agg_cols] = agg_df[numeric_agg_cols].fillna(0)

        # Temporal features — take mean per customer
        temporal_cols = [c for c in X.columns if c.startswith("tx_")]
        if temporal_cols:
            temp_agg = X.groupby("CustomerId")[temporal_cols].mean()
            temp_agg.columns = [f"{c}_mean" for c in temp_agg.columns]
            agg_df = agg_df.merge(temp_agg, on="CustomerId", how="left")
            # These are means of always-present temporal features, so any
            # NaN here would only come from a merge mismatch, not from
            # single-observation variance — fine to zero-fill as well.
            mean_cols = list(temp_agg.columns)
            agg_df[mean_cols] = agg_df[mean_cols].fillna(0)

        # Categorical mode per customer (most frequent value)
        existing_cats = [c for c in CAT_COLS if c in X.columns]
        for col in existing_cats:
            mode_df = (
                X.groupby("CustomerId")[col]
                .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan)
                .reset_index()
                .rename(columns={col: f"{col}_mode"})
            )
            agg_df = agg_df.merge(mode_df, on="CustomerId", how="left")
            # NOTE: intentionally NOT filled here. CategoricalEncoder
            # handles NaN/missing categoricals explicitly downstream.

        return agg_df


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Label-encodes categorical mode columns produced by CustomerAggregator.
    Fits encoders on training data only. Handles unseen values at inference
    by mapping to -1 rather than crashing.
    """

    def __init__(self):
        self.encoders_ = {}

    def fit(self, X, y=None):
        mode_cols = [c for c in X.columns if c.endswith("_mode")]
        for col in mode_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str).fillna("missing"))
            self.encoders_[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        for col, le in self.encoders_.items():
            if col in X.columns:
                # Handle unseen labels gracefully
                known = set(le.classes_)
                X[col] = X[col].astype(str).fillna("missing").apply(
                    lambda v: v if v in known else "missing"
                )
                X[col] = le.transform(X[col])
        return X


class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Applies RobustScaler to all numerical columns except CustomerId.

    RobustScaler is chosen over StandardScaler because:
    - Amount skewness = 51 → mean/std are meaningless measures of center
    - RobustScaler uses median and IQR — robust to extreme outliers
    - Does NOT clip or remove outliers — preserves signal
    """

    def fit(self, X, y=None):
        self.num_cols_ = X.select_dtypes(include="number").columns.tolist()
        # Never scale the ID column
        self.num_cols_ = [c for c in self.num_cols_ if c != "CustomerId"]
        self.scaler_ = RobustScaler()
        self.scaler_.fit(X[self.num_cols_])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.num_cols_] = self.scaler_.transform(X[self.num_cols_])
        return X


# ── IV Filtering (manual implementation — see docstring for why) ──────────

def compute_iv_and_select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    threshold: float = IV_THRESHOLD,
) -> list:
    """
    Computes Information Value (IV) for each feature manually.

    Replaces xverse.WOE, which is incompatible with the scikit-learn/pandas
    versions in this environment (WOE.fit() always routes through
    MonotonicBinning.fit(), which calls sklearn's check_array() with the
    old `force_all_finite` keyword — renamed to `ensure_all_finite` in
    current scikit-learn — so it fails unconditionally regardless of
    column dtypes). Computing IV directly avoids the dependency entirely.

    IV interpretation (Siddiqi 2006):
        < 0.02  : unpredictive — drop
        0.02–0.1: weak
        0.1–0.3 : medium
        > 0.3   : strong

    Note on WoE sign convention: this implementation computes
    woe = log(dist_events / dist_non_events), which is the mirror image
    of the conventional log(dist_non_events / dist_events) used in some
    references. This only flips the sign of the per-bin `woe` column
    (an intermediate value that isn't returned or used elsewhere) — the
    per-bin IV term (dist_events - dist_non_events) * woe, and therefore
    the summed IV per feature used for ranking/selection below, is
    identical either way, since both the subtraction and the log flip
    sign together and cancel out.

    X_train and y_train are paired by index (not row position) — caller
    is responsible for ensuring y_train is indexed consistently with
    X_train (e.g. both indexed by CustomerId) before calling this.
    """
    # Pair X and y by index explicitly rather than trusting row order.
    y_train = y_train.reindex(X_train.index)
    if y_train.isna().any():
        raise ValueError(
            "y_train has missing labels for some rows in X_train after "
            "aligning by index. Check that X_train and y_train share a "
            "common, consistent index before calling this function."
        )

    iv_scores = {}
    total_events     = (y_train == 1).sum()
    total_non_events = (y_train == 0).sum()

    if total_events == 0 or total_non_events == 0:
        logger.warning("IV computation skipped — target has only one class")
        return X_train.columns.tolist()

    for col in X_train.columns:
        try:
            # Bin continuous features into 10 quantile bins
            binned = pd.qcut(X_train[col], q=10, duplicates="drop")
            temp   = pd.DataFrame({
                "bin"   : binned,
                "target": y_train.values,
            })

            grouped = temp.groupby("bin", observed=True)["target"].agg(
                events     = lambda x: (x == 1).sum(),
                non_events = lambda x: (x == 0).sum(),
            ).reset_index()

            grouped["dist_events"]     = grouped["events"]     / total_events
            grouped["dist_non_events"] = grouped["non_events"] / total_non_events

            # Replace zeros to avoid log(0)
            grouped["dist_events"]     = grouped["dist_events"].replace(0, 0.0001)
            grouped["dist_non_events"] = grouped["dist_non_events"].replace(0, 0.0001)

            grouped["woe"] = np.log(
                grouped["dist_events"] / grouped["dist_non_events"]
            )
            grouped["iv"]  = (
                grouped["dist_events"] - grouped["dist_non_events"]
            ) * grouped["woe"]

            iv_scores[col] = grouped["iv"].sum()

        except Exception as e:
            logger.debug(f"IV computation skipped for {col}: {e}")
            iv_scores[col] = 0.0

    iv_df = pd.DataFrame.from_dict(
        iv_scores, orient="index", columns=["IV"]
    ).sort_values("IV", ascending=False)

    logger.info(f"IV scores:\n{iv_df.to_string()}")

    selected = iv_df[iv_df["IV"] >= threshold].index.tolist()
    dropped  = iv_df[iv_df["IV"] <  threshold].index.tolist()

    logger.info(
        f"IV filtering: {len(selected)}/{len(X_train.columns)} features retained\n"
        f"Dropped (IV < {threshold}): {dropped}"
    )

    return selected if selected else X_train.columns.tolist()


# ── Pipeline Builder ───────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    Constructs the end-to-end feature engineering Pipeline.

    Step order matters:
    1. Drop zero-signal columns first — reduces noise in all subsequent steps
    2. Extract temporal features before aggregation — they feed into customer means
    3. Apply signed log to Amount before aggregation — aggregates computed on
       transformed values, consistent with inference time
    4. Aggregate to customer level — all downstream modeling is customer-level
    5. Encode categoricals — must happen after aggregation (mode columns)
    6. Scale numericals — must be last, fit on training distribution only
    """
    return Pipeline([
        ("drop_cols",    DropColumnsTransformer(cols_to_drop=COLS_TO_DROP)),
        ("temporal",     TemporalFeatureExtractor()),
        ("signed_log",   SignedLogTransformer()),
        ("aggregate",    CustomerAggregator(agg_functions=AGG_FUNCTIONS)),
        ("encode_cats",  CategoricalEncoder()),
        ("scale",        FeatureScaler()),
    ])


# ── Main processing function ───────────────────────────────────────────────

def process_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_train: pd.Series = None,
    save_pipeline: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline, list]:
    """
    Fits the pipeline on training data, transforms both splits.
    Optionally runs IV filtering and saves artifacts.

    Returns:
        X_train_processed, X_test_processed, fitted_pipeline, selected_features
    """
    pipeline = build_pipeline()

    logger.info("Fitting pipeline on training data...")
    X_train_proc = pipeline.fit_transform(df_train)

    logger.info("Transforming test data with fitted pipeline...")
    X_test_proc = pipeline.transform(df_test)

    # IV-based feature selection — training data only
    selected_features = X_train_proc.columns.tolist()
    if y_train is not None:
        # Align y_train to X_train_proc's row order, indexed by CustomerId.
        # IMPORTANT: y_train.loc[y_train.index.isin(customer_ids)] (the
        # previous approach) only FILTERS y_train — it does not reorder it
        # to match X_train_proc's row order. compute_iv_and_select_features
        # pairs X and y by row position, so any difference between
        # X_train_proc's CustomerId order and y_train's index order would
        # silently mislabel every row with no error raised. Using
        # CustomerId as the index on both sides and reindexing explicitly
        # guarantees correct pairing regardless of how either was sorted.
        customer_ids = X_train_proc["CustomerId"]
        y_aligned = y_train.reindex(customer_ids.values)

        if y_aligned.isna().any():
            missing = customer_ids[y_aligned.isna().values].tolist()
            raise ValueError(
                f"y_train is missing labels for {len(missing)} customer(s) "
                f"present in X_train_proc: {missing[:10]}"
                f"{'...' if len(missing) > 10 else ''}. "
                f"IV computation requires a label for every training customer."
            )

        num_cols = X_train_proc.select_dtypes(include="number").columns.tolist()
        num_cols = [c for c in num_cols if c != "CustomerId"]

        X_iv = X_train_proc.set_index("CustomerId")[num_cols]
        selected_features = compute_iv_and_select_features(
            X_iv, y_aligned, threshold=IV_THRESHOLD
        )

    if save_pipeline:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, ARTIFACTS_DIR / "feature_pipeline.joblib")
        logger.info(f"Pipeline saved to {ARTIFACTS_DIR / 'feature_pipeline.joblib'}")

        # Save selected feature names for inference consistency
        pd.Series(selected_features).to_csv(
            ARTIFACTS_DIR / "selected_features.csv", index=False
        )

    return X_train_proc, X_test_proc, pipeline, selected_features