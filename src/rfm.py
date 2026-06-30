"""
rfm.py
──────
Generates the proxy credit risk target variable (is_high_risk)
from customer RFM behavioral patterns.

Key constraint: ALL computations use training window data only.
Snapshot date and cutoff date are imported from config — never
hardcoded — so the entire pipeline is reproducible with a single
config change.

Proxy assumption (document this):
    Low RFM engagement (infrequent, low-value, distant transactions)
    is used as a behavioral proxy for credit default risk.
    This assumption is UNVALIDATED against actual default data and
    must be treated as a first-generation risk signal only.
"""

import logging
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans

from src.config import (
    SNAPSHOT_DATE,
    TRAIN_CUTOFF_DATE,
    N_CLUSTERS,
    RFM_RANDOM_STATE,
    ARTIFACTS_DIR,
)

logger = logging.getLogger(__name__)


# ── Step 1: Filter to training window ─────────────────────────────────────

def filter_training_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns only transactions that fall within the training window.

    Training window: TransactionStartTime < TRAIN_CUTOFF_DATE
    This is called BEFORE any RFM computation to prevent target leakage.
    """
    df = df.copy()
    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"], utc=True
    )
    cutoff = pd.Timestamp(TRAIN_CUTOFF_DATE, tz="UTC")
    train_df = df[df["TransactionStartTime"] < cutoff].copy()

    logger.info(
        f"Training window filter: {len(train_df):,} / {len(df):,} "
        f"transactions retained (cutoff: {TRAIN_CUTOFF_DATE})"
    )
    return train_df


# ── Step 2: Compute RFM metrics ────────────────────────────────────────────

def compute_rfm(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Recency, Frequency, Monetary per CustomerId.

    Recency:   Days between customer's last transaction and SNAPSHOT_DATE.
               Higher = less recent = potentially higher risk.

    Frequency: Total number of transactions in the training window.
               Lower = less engaged = potentially higher risk.

    Monetary:  Sum of positive transaction amounts in the training window.
               Negative amounts are credits (refunds/reversals) — excluded
               from monetary value to avoid distorting spending behavior.
               Lower = lower engagement = potentially higher risk.
    """
    snapshot = pd.Timestamp(SNAPSHOT_DATE, tz="UTC")

    # Monetary: positive amounts only (debits from customer = spending)
    train_df["Amount_positive"] = train_df["Amount"].clip(lower=0)

    rfm = train_df.groupby("CustomerId").agg(
        Recency=("TransactionStartTime",
                 lambda x: (snapshot - x.max()).days),
        Frequency=("TransactionId", "count"),
        Monetary=("Amount_positive", "sum"),
    ).reset_index()

    logger.info(
        f"RFM computed for {len(rfm):,} customers\n"
        f"Recency  — mean: {rfm['Recency'].mean():.1f} days, "
        f"max: {rfm['Recency'].max()} days\n"
        f"Frequency — mean: {rfm['Frequency'].mean():.1f} txns, "
        f"max: {rfm['Frequency'].max()}\n"
        f"Monetary  — mean: {rfm['Monetary'].mean():.2f}, "
        f"max: {rfm['Monetary'].max():.2f}"
    )
    return rfm


# ── Step 3: Cap outliers, then scale RFM features ─────────────────────────

def cap_rfm_outliers(rfm: pd.DataFrame, quantile: float = 0.99) -> pd.DataFrame:
    """
    Caps Frequency and Monetary at the 99th percentile before clustering.

    Why: A single customer with 4,091 transactions (vs mean of 23)
    anchors its own KMeans cluster, making the remaining 3,258 customers
    indistinguishable. Capping preserves the distribution shape while
    preventing one outlier from dictating cluster geometry.

    Recency is not capped — its range is bounded by the 90-day window.
    """
    rfm = rfm.copy()
    for col in ["Frequency", "Monetary"]:
        cap = rfm[col].quantile(quantile)
        n_capped = (rfm[col] > cap).sum()
        rfm[col] = rfm[col].clip(upper=cap)
        logger.info(f"{col} capped at {cap:.2f} (99th pct) — {n_capped} customers affected")
    return rfm


def scale_rfm(rfm: pd.DataFrame) -> tuple[np.ndarray, RobustScaler]:
    """
    Scales RFM features before clustering.

    Monetary is log-compressed (log1p) before scaling — same reason as
    Amount in the feature pipeline: even after capping at the 99th
    percentile (see cap_rfm_outliers), Monetary remains heavily
    right-skewed. Log-compression pulls in the long tail so the
    subsequent scaling step doesn't let a handful of high-spend
    customers dominate distance calculations.

    RobustScaler used for the same reason as the feature pipeline:
    it's based on median/IQR, so it isn't thrown off by the residual
    skew or by outliers that survived capping.
    KMeans uses Euclidean distance — unscaled features let Monetary
    dominate cluster assignments entirely, producing meaningless clusters.

    Returns scaled array and fitted scaler (saved for reproducibility).
    """
    rfm = rfm.copy()
    rfm["Monetary"] = np.log1p(rfm["Monetary"])

    scaler = RobustScaler()
    rfm_features = rfm[["Recency", "Frequency", "Monetary"]]
    rfm_scaled = scaler.fit_transform(rfm_features)
    logger.info("RFM features log-transformed (Monetary) and scaled with RobustScaler")
    return rfm_scaled, scaler


# ── Step 4: Cluster customers ──────────────────────────────────────────────

def cluster_customers(
    rfm_scaled: np.ndarray,
) -> tuple[KMeans, np.ndarray]:
    """
    Applies KMeans with k=3 to segment customers by RFM behavior.

    k=3 is specified by the assignment and is a reasonable choice:
    - Low risk (engaged, high value)
    - Medium risk
    - High risk (disengaged, low value)

    random_state=RFM_RANDOM_STATE ensures reproducible cluster assignments
    across runs. Without this, cluster labels shuffle between runs and
    the high-risk cluster identification below becomes unreliable.
    """
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RFM_RANDOM_STATE,
        n_init=10,          # run 10 times, pick best inertia
        max_iter=300,
    )
    labels = kmeans.fit_predict(rfm_scaled)
    logger.info(
        f"KMeans converged in {kmeans.n_iter_} iterations | "
        f"Inertia: {kmeans.inertia_:.2f}"
    )
    return kmeans, labels


# ── Step 5: Identify high-risk cluster ────────────────────────────────────

def identify_high_risk_cluster(
    rfm: pd.DataFrame,
    labels: np.ndarray,
) -> tuple[int, pd.DataFrame]:
    """
    Identifies the high-risk cluster using min-max normalized cluster means.

    Why normalization is required:
        Cluster means are in raw RFM units (days, count, currency).
        Without normalization, Monetary (tens of thousands) completely
        dominates Recency (tens of days) and Frequency (single digits)
        in the composite score — making the weights meaningless.
        Min-max normalization puts all three dimensions on [0, 1]
        so the weights reflect actual business intent.

    Risk score (post-normalization):
        higher Recency_norm   → more days inactive → worse  (+1)
        lower  Frequency_norm → fewer transactions → worse  (-1)
        lower  Monetary_norm  → less spending      → worse  (-1)
    """
    rfm = rfm.copy()
    rfm["cluster"] = labels

    cluster_summary = rfm.groupby("cluster").agg(
        mean_Recency=("Recency", "mean"),
        mean_Frequency=("Frequency", "mean"),
        mean_Monetary=("Monetary", "mean"),
        n_customers=("CustomerId", "count"),
    ).reset_index()

    logger.info(f"Cluster summary (raw means):\n{cluster_summary.to_string(index=False)}")

    # Min-max normalize each dimension across clusters
    for col in ["mean_Recency", "mean_Frequency", "mean_Monetary"]:
        col_min = cluster_summary[col].min()
        col_max = cluster_summary[col].max()
        col_range = col_max - col_min
        cluster_summary[f"{col}_norm"] = (
            (cluster_summary[col] - col_min) / col_range
            if col_range > 0 else 0.0
        )

    # Composite risk score on normalized dimensions
    cluster_summary["risk_score"] = (
        cluster_summary["mean_Recency_norm"] * 1 +  # higher = worse
        cluster_summary["mean_Frequency_norm"] * -1 +  # lower  = worse
        cluster_summary["mean_Monetary_norm"] * -1    # lower  = worse
    )

    summary = cluster_summary[
        [
            "cluster",
            "mean_Recency_norm",
            "mean_Frequency_norm",
            "mean_Monetary_norm",
            "risk_score",
        ]
    ]

    logger.info(
        "Cluster risk scores (normalized):\n%s",
        summary.to_string(index=False),
    )

    high_risk_cluster = int(
        cluster_summary.loc[cluster_summary["risk_score"].idxmax(), "cluster"]
    )

    high_risk_n = cluster_summary.loc[
        cluster_summary["cluster"] == high_risk_cluster, "n_customers"
    ].values[0]
    high_risk_pct = high_risk_n / len(rfm) * 100

    logger.info(
        f"High-risk cluster: {high_risk_cluster} "
        f"({high_risk_pct:.1f}% of training customers)"
    )

    if high_risk_pct < 5:
        logger.warning(
            f"High-risk rate {high_risk_pct:.1f}% is very low — "
            f"severe class imbalance expected. Use class_weight='balanced'."
        )
    if high_risk_pct > 60:
        logger.warning(
            f"High-risk rate {high_risk_pct:.1f}% is too high — "
            f"clusters are not separating well. Review preprocessing."
        )

    return high_risk_cluster, cluster_summary
# ── Step 6: Assign binary label ────────────────────────────────────────────


def assign_risk_labels(
    rfm: pd.DataFrame,
    labels: np.ndarray,
    high_risk_cluster: int,
) -> pd.DataFrame:
    """
    Creates the is_high_risk binary column.

    is_high_risk = 1 → customer is in the high-risk cluster
    is_high_risk = 0 → customer is in any other cluster

    Returns dataframe with CustomerId and is_high_risk only.
    This is merged into the processed feature matrix in train.py.
    """
    rfm = rfm.copy()
    rfm["cluster"] = labels
    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    label_dist = rfm["is_high_risk"].value_counts(normalize=True) * 100
    logger.info(
        f"Label distribution:\n"
        f"  Low risk  (0): {label_dist.get(0, 0):.1f}%\n"
        f"  High risk (1): {label_dist.get(1, 0):.1f}%"
    )

    return rfm[["CustomerId", "is_high_risk"]]


# ── Step 7: Save artifacts ─────────────────────────────────────────────────

def save_rfm_artifacts(
    rfm: pd.DataFrame,
    kmeans: KMeans,
    scaler: RobustScaler,
    cluster_summary: pd.DataFrame,
) -> None:
    """
    Persists RFM artifacts for reproducibility and audit trail.

    - rfm_with_labels.csv  → full RFM table with cluster assignments
    - rfm_kmeans.joblib    → fitted KMeans model
    - rfm_scaler.joblib    → fitted RobustScaler
    - cluster_summary.csv  → cluster profiles for documentation
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    rfm.to_csv(ARTIFACTS_DIR / "rfm_with_labels.csv", index=False)
    joblib.dump(kmeans, ARTIFACTS_DIR / "rfm_kmeans.joblib")
    joblib.dump(scaler, ARTIFACTS_DIR / "rfm_scaler.joblib")
    cluster_summary.to_csv(
        ARTIFACTS_DIR / "cluster_summary.csv", index=False
    )
    logger.info(f"RFM artifacts saved to {ARTIFACTS_DIR}")


# ── Main entry point ───────────────────────────────────────────────────────

def generate_proxy_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full proxy label generation pipeline.
    Call this with the RAW dataframe — filtering happens internally.

    Returns:
        DataFrame with columns [CustomerId, is_high_risk]
        — one row per training-window customer only.

    Usage in train.py:
        from src.rfm import generate_proxy_labels
        labels_df = generate_proxy_labels(df_raw)
        df_train  = df_train.merge(labels_df, on="CustomerId", how="inner")
    """
    logger.info("=" * 60)
    logger.info("Starting proxy label generation")
    logger.info(f"  Snapshot date  : {SNAPSHOT_DATE}")
    logger.info(f"  Train cutoff   : {TRAIN_CUTOFF_DATE}")
    logger.info(f"  N clusters     : {N_CLUSTERS}")
    logger.info("=" * 60)

    # Step 1: restrict to training window
    train_df = filter_training_window(df)

    # Step 2: compute RFM
    rfm = compute_rfm(train_df)

    # Step 3a: cap outliers (Frequency, Monetary)
    rfm = cap_rfm_outliers(rfm)

    # Step 3b: log-transform Monetary and scale
    rfm_scaled, scaler = scale_rfm(rfm)

    # Step 4: cluster
    kmeans, labels = cluster_customers(rfm_scaled)

    # Step 5: identify high-risk cluster
    high_risk_cluster, cluster_summary = identify_high_risk_cluster(rfm, labels)

    # Step 6: assign binary labels
    labels_df = assign_risk_labels(rfm, labels, high_risk_cluster)

    # Step 7: save artifacts
    save_rfm_artifacts(
        rfm.assign(
            cluster=labels,
            is_high_risk=(labels == high_risk_cluster).astype(int)
        ),
        kmeans,
        scaler,
        cluster_summary,
    )

    logger.info("Proxy label generation complete")
    return labels_df
