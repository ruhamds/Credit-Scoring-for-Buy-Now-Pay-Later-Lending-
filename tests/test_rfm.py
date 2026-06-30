"""
test_rfm.py
───────────
Tests for proxy target generation.
Focus: training window isolation, label validity, cluster stability.
"""

import pytest
import numpy as np
import pandas as pd
from src.rfm import (
    filter_training_window,
    compute_rfm,
    cap_rfm_outliers,
    scale_rfm,
    cluster_customers,
    assign_risk_labels,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def raw_df():
    """Transactions spanning both train and test windows."""
    return pd.DataFrame({
        "TransactionId": [f"T{i}" for i in range(6)],
        "CustomerId": ["C1", "C1", "C2", "C2", "C3", "C3"],
        "Amount": [1000, 500, 200, 150, 5000, 4000],
        "TransactionStartTime": [
            "2018-12-01T10:00:00Z",   # train
            "2019-01-10T10:00:00Z",   # train
            "2018-11-20T10:00:00Z",   # train
            "2019-01-15T10:00:00Z",   # train
            "2019-01-05T10:00:00Z",   # train
            "2019-02-01T10:00:00Z",   # TEST WINDOW — must be excluded
        ],
    })


# ── filter_training_window ─────────────────────────────────────────────────

def test_filter_excludes_test_window_transactions(raw_df):
    """
    Transactions after TRAIN_CUTOFF_DATE must be excluded.
    This is the core leakage prevention test.
    """
    filtered = filter_training_window(raw_df)
    filtered["TransactionStartTime"] = pd.to_datetime(
        filtered["TransactionStartTime"], utc=True
    )
    cutoff = pd.Timestamp("2019-01-29", tz="UTC")
    assert (filtered["TransactionStartTime"] < cutoff).all(), (
        "Test-window transactions survived the training filter — target leakage"
    )


def test_filter_retains_train_transactions(raw_df):
    """Training window transactions must not be dropped."""
    filtered = filter_training_window(raw_df)
    assert len(filtered) == 5   # 6 rows minus 1 test-window row


# ── compute_rfm ────────────────────────────────────────────────────────────

def test_rfm_one_row_per_customer(raw_df):
    """RFM output must have exactly one row per unique training customer."""
    train_df = filter_training_window(raw_df)
    rfm = compute_rfm(train_df)
    assert rfm["CustomerId"].nunique() == len(rfm), (
        "Duplicate CustomerId rows in RFM output"
    )


def test_rfm_recency_is_non_negative(raw_df):
    """Recency must be >= 0 — negative days would indicate future transactions."""
    train_df = filter_training_window(raw_df)
    rfm = compute_rfm(train_df)
    assert (rfm["Recency"] >= 0).all(), (
        "Negative recency detected — snapshot date may precede last transaction"
    )


def test_rfm_frequency_matches_transaction_count(raw_df):
    """C1 has 2 training transactions — Frequency must equal 2."""
    train_df = filter_training_window(raw_df)
    rfm = compute_rfm(train_df)
    c1_freq = rfm.loc[rfm["CustomerId"] == "C1", "Frequency"].values[0]
    assert c1_freq == 2, f"Expected frequency 2 for C1, got {c1_freq}"


# ── assign_risk_labels ─────────────────────────────────────────────────────

def test_labels_are_binary(raw_df):
    """is_high_risk must only contain 0 or 1 — no other values."""
    train_df = filter_training_window(raw_df)
    rfm = compute_rfm(train_df)
    labels = np.array([0, 1, 0])   # mock cluster labels
    result = assign_risk_labels(rfm, labels, high_risk_cluster=1)
    assert set(result["is_high_risk"].unique()).issubset({0, 1}), (
        "is_high_risk contains values other than 0 and 1"
    )


def test_labels_df_has_correct_columns(raw_df):
    """Output must have exactly CustomerId and is_high_risk — nothing else."""
    train_df = filter_training_window(raw_df)
    rfm = compute_rfm(train_df)
    labels = np.array([0, 1, 0])
    result = assign_risk_labels(rfm, labels, high_risk_cluster=1)
    assert list(result.columns) == ["CustomerId", "is_high_risk"], (
        f"Unexpected columns: {list(result.columns)}"
    )


def test_high_risk_cluster_assignment_is_correct(raw_df):
    """Customers in high_risk_cluster=1 must get is_high_risk=1."""
    train_df = filter_training_window(raw_df)
    rfm = compute_rfm(train_df)
    labels = np.array([0, 1, 0])
    result = assign_risk_labels(rfm, labels, high_risk_cluster=1)
    # C2 maps to label 1 → must be high risk
    c2_label = result.loc[result["CustomerId"] == "C2", "is_high_risk"].values[0]
    assert c2_label == 1, f"C2 should be high risk but got {c2_label}"


# ── cluster stability ───────────────────────────────────────────────────────

def test_no_single_cluster_dominates(raw_df):
    """
    No cluster should contain more than 80% of customers.
    If it does, clustering has failed — outliers are dominating.
    This test would have caught the 99.9% cluster failure.
    """
    train_df = filter_training_window(raw_df)
    rfm = compute_rfm(train_df)
    rfm = cap_rfm_outliers(rfm)
    rfm_scaled, _ = scale_rfm(rfm)
    _, labels = cluster_customers(rfm_scaled)

    unique, counts = np.unique(labels, return_counts=True)
    max_pct = counts.max() / len(labels)
    assert max_pct < 0.80, (
        f"Dominant cluster contains {max_pct*100:.1f}% of customers — "
        f"clustering has failed, likely due to outliers"
    )
