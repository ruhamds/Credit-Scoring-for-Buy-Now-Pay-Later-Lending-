"""
test_data_processing.py
───────────────────────
Tests for feature engineering pipeline.
Each test targets a specific failure mode that would be silent in production.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_processing import (
    DropColumnsTransformer,
    TemporalFeatureExtractor,
    SignedLogTransformer,
    CustomerAggregator,
    build_pipeline,
)
from src.config import COLS_TO_DROP, AGG_FUNCTIONS


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal transaction dataframe mimicking Xente schema."""
    return pd.DataFrame({
        "TransactionId"      : ["T1", "T2", "T3", "T4"],
        "BatchId"            : ["B1", "B1", "B2", "B2"],
        "AccountId"          : ["A1", "A1", "A2", "A2"],
        "SubscriptionId"     : ["S1", "S1", "S2", "S2"],
        "CustomerId"         : ["C1", "C1", "C2", "C2"],
        "CurrencyCode"       : ["UGX"] * 4,
        "CountryCode"        : [256] * 4,
        "ProviderId"         : ["ProviderId_4"] * 4,
        "ProductId"          : ["ProductId_6"] * 4,
        "ProductCategory"    : ["financial_services", "airtime",
                                "financial_services", "utility_bill"],
        "ChannelId"          : ["ChannelId_3"] * 4,
        "Amount"             : [1000.0, -50.0, 2800.0, 500.0],
        "Value"              : [1000, 50, 2800, 500],
        "TransactionStartTime": [
            "2018-11-15T10:00:00Z",
            "2018-11-16T14:30:00Z",
            "2018-12-01T08:00:00Z",
            "2018-12-15T20:00:00Z",
        ],
        "PricingStrategy"    : [2, 2, 2, 4],
        "FraudResult"        : [0, 0, 0, 0],
    })


@pytest.fixture
def sample_df_with_single_tx():
    """Includes a single-transaction customer — exposes Amount_std NaN bug."""
    return pd.DataFrame({
        "TransactionId"      : ["T1", "T2", "T3", "T4", "T5"],
        "BatchId"            : ["B1", "B1", "B2", "B2", "B3"],
        "AccountId"          : ["A1", "A1", "A2", "A2", "A3"],
        "SubscriptionId"     : ["S1", "S1", "S2", "S2", "S3"],
        "CustomerId"         : ["C1", "C1", "C2", "C2", "C3"],  # C3 has 1 tx
        "CurrencyCode"       : ["UGX"] * 5,
        "CountryCode"        : [256] * 5,
        "ProviderId"         : ["ProviderId_4"] * 5,
        "ProductId"          : ["ProductId_6"] * 5,
        "ProductCategory"    : ["financial_services"] * 5,
        "ChannelId"          : ["ChannelId_3"] * 5,
        "Amount"             : [1000.0, -50.0, 2800.0, 500.0, 750.0],
        "Value"              : [1000, 50, 2800, 500, 750],
        "TransactionStartTime": [
            "2018-11-15T10:00:00Z",
            "2018-11-16T14:30:00Z",
            "2018-12-01T08:00:00Z",
            "2018-12-15T20:00:00Z",
            "2019-01-10T09:00:00Z",
        ],
        "PricingStrategy"    : [2, 2, 2, 4, 2],
        "FraudResult"        : [0, 0, 0, 0, 0],
    })


def test_pipeline_handles_single_transaction_customer(sample_df_with_single_tx):
    """
    Pipeline must not produce NaN for customers with only one transaction.
    Amount_std is undefined for n=1 — must be filled with 0.
    This test would have caught the 712-customer NaN bug on real data.
    """
    pipeline = build_pipeline()
    result   = pipeline.fit_transform(sample_df_with_single_tx)

    null_counts = result.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    assert len(cols_with_nulls) == 0, (
        f"NaN produced for single-transaction customer: "
        f"{cols_with_nulls.to_dict()}"
    )

# ── DropColumnsTransformer ─────────────────────────────────────────────────

def test_drop_columns_removes_configured_cols(sample_df):
    """Zero-signal columns must not survive into the pipeline."""
    transformer = DropColumnsTransformer(cols_to_drop=COLS_TO_DROP)
    result = transformer.fit_transform(sample_df)
    for col in COLS_TO_DROP:
        assert col not in result.columns, (
            f"Column '{col}' should be dropped but is still present"
        )


def test_drop_columns_ignores_missing_cols(sample_df):
    """Transformer must not crash if a configured column is absent."""
    transformer = DropColumnsTransformer(cols_to_drop=["NonExistentColumn"])
    result = transformer.fit_transform(sample_df)
    assert result is not None


# ── TemporalFeatureExtractor ───────────────────────────────────────────────

def test_temporal_features_created(sample_df):
    """All five temporal features must be present after extraction."""
    transformer = TemporalFeatureExtractor()
    result = transformer.fit_transform(sample_df)
    expected = ["tx_hour", "tx_day", "tx_month", "tx_dayofweek", "tx_is_weekend"]
    for col in expected:
        assert col in result.columns, f"Expected temporal feature '{col}' not found"


def test_original_datetime_col_dropped(sample_df):
    """TransactionStartTime must be dropped after feature extraction."""
    transformer = TemporalFeatureExtractor()
    result = transformer.fit_transform(sample_df)
    assert "TransactionStartTime" not in result.columns


def test_is_weekend_is_binary(sample_df):
    """tx_is_weekend must only contain 0 or 1."""
    transformer = TemporalFeatureExtractor()
    result = transformer.fit_transform(sample_df)
    assert set(result["tx_is_weekend"].unique()).issubset({0, 1})


# ── SignedLogTransformer ───────────────────────────────────────────────────

def test_signed_log_handles_negatives(sample_df):
    """
    Negative Amount values must not produce NaN or crash.
    This is the key failure mode of standard log1p.
    """
    transformer = SignedLogTransformer()
    result = transformer.fit_transform(sample_df)
    assert result["Amount"].isna().sum() == 0, (
        "SignedLogTransformer produced NaN — negative values not handled"
    )


def test_signed_log_preserves_sign(sample_df):
    """Negative inputs must produce negative outputs — direction preserved."""
    transformer = SignedLogTransformer()
    result = transformer.fit_transform(sample_df)
    neg_mask = sample_df["Amount"] < 0
    assert (result.loc[neg_mask, "Amount"] < 0).all(), (
        "Sign not preserved after log transform"
    )


# ── CustomerAggregator ─────────────────────────────────────────────────────

def test_aggregator_produces_one_row_per_customer(sample_df):
    """Output must have exactly one row per unique CustomerId."""
    # Run upstream steps first
    df = DropColumnsTransformer(cols_to_drop=COLS_TO_DROP).fit_transform(sample_df)
    df = TemporalFeatureExtractor().fit_transform(df)
    df = SignedLogTransformer().fit_transform(df)
    agg = CustomerAggregator(agg_functions=AGG_FUNCTIONS).fit_transform(df)
    assert len(agg) == sample_df["CustomerId"].nunique(), (
        "Aggregator did not collapse to one row per customer"
    )


def test_aggregator_creates_amount_features(sample_df):
    """Expected aggregate columns must all be present."""
    df = DropColumnsTransformer(cols_to_drop=COLS_TO_DROP).fit_transform(sample_df)
    df = TemporalFeatureExtractor().fit_transform(df)
    df = SignedLogTransformer().fit_transform(df)
    agg = CustomerAggregator(agg_functions=AGG_FUNCTIONS).fit_transform(df)
    expected_cols = ["Amount_sum", "Amount_mean", "Amount_count",
                     "Amount_std", "Amount_max", "Amount_min"]
    for col in expected_cols:
        assert col in agg.columns, f"Missing aggregate column: {col}"


# ── Full pipeline ──────────────────────────────────────────────────────────

def test_pipeline_fit_transform_produces_dataframe(sample_df):
    """Full pipeline must return a DataFrame without crashing."""
    pipeline = build_pipeline()
    result = pipeline.fit_transform(sample_df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_pipeline_no_nulls_after_transform(sample_df):
    """
    Processed output must have no nulls.
    Silent nulls in features cause silent model degradation.
    """
    pipeline = build_pipeline()
    result = pipeline.fit_transform(sample_df)
    null_counts = result.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    assert len(cols_with_nulls) == 0, (
        f"Nulls present after pipeline: {cols_with_nulls.to_dict()}"
    )


def test_pipeline_transform_matches_fit_transform_shape(sample_df):
    """
    Pipeline.transform() on new data must produce same column count
    as fit_transform(). Catches feature mismatch at inference time.
    """
    pipeline = build_pipeline()
    train_result = pipeline.fit_transform(sample_df)

    # Simulate inference on new data with same schema
    new_data = sample_df.copy()
    new_data["CustomerId"] = ["C3", "C3", "C4", "C4"]
    test_result = pipeline.transform(new_data)

    assert train_result.shape[1] == test_result.shape[1], (
        "Column count mismatch between fit_transform and transform — "
        "inference pipeline is inconsistent with training pipeline"
    )