import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parent.parent
DATA_RAW       = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
ARTIFACTS_DIR  = ROOT_DIR / "artifacts"

# ── Data ───────────────────────────────────────────────────────────────────
RAW_FILE       = DATA_RAW / "data.csv"          # rename to match your filename

# ── RFM ────────────────────────────────────────────────────────────────────
# Snapshot date = 1 day after the last transaction in the dataset.
# We will confirm this exact date during EDA and update it here.
SNAPSHOT_DATE  = "2019-04-01"                   # placeholder — verify in EDA
N_CLUSTERS     = 3

# ── Model ──────────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
TEST_SIZE      = 0.2                            # temporal — last 20% of time
TARGET_COL     = "is_high_risk"

# ── Business cost for threshold optimization ───────────────────────────────
# A missed default (FN) costs ~15x more than a wrongly rejected applicant (FP)
FN_COST        = 15
FP_COST        = 1