from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# ── Data ───────────────────────────────────────────────────────────────────
RAW_FILE = DATA_RAW / "data.csv"          # rename to match your filename

# ── RFM ────────────────────────────────────────────────────────────────────
# Snapshot date = 1 day after the last transaction in the dataset.q
SNAPSHOT_DATE = "2019-02-14"   # 1 day after last transaction (confirmed)
TRAIN_CUTOFF_DATE = "2019-01-29"   # ~75 days train, ~15 days test
# adjust if distribution is too thin

# ── Columns to drop immediately (zero variance / redundant) ────────────────
COLS_TO_DROP = [
    "TransactionId",      # unique per row — no signal
    "BatchId",            # processing artifact
    "SubscriptionId",     # near-unique
    "AccountId",          # use CustomerId instead
    "CurrencyCode",       # constant
    "CountryCode",        # constant
    "Value",              # r=0.99 with Amount — redundant
]

N_CLUSTERS = 3

# ── Model ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2                            # temporal — last 20% of time
TARGET_COL = "is_high_risk"

# ── Business cost for threshold optimization ───────────────────────────────
# A missed default (FN) costs ~15x more than a wrongly rejected applicant (FP)
FN_COST = 5   # reduce from 15
FP_COST = 1

# ── Feature engineering ────────────────────────────────────────────────────
# IV threshold — features below this are dropped (industry standard)
IV_THRESHOLD = 0.02

# Categorical columns to encode
CAT_COLS = ["ProviderId", "ProductCategory", "ChannelId"]

# Numerical columns after aggregation (before scaling)
# Populated after WoE/IV selection — placeholder for now
NUM_COLS = ["Amount"]

# Aggregation feature suffixes built per CustomerId
AGG_FUNCTIONS = {
    "Amount": ["sum", "mean", "count", "std", "max", "min"]
}
# ── RFM ────────────────────────────────────────────────────────────────────
SNAPSHOT_DATE = "2019-02-14"   # already set — confirmed from EDA
TRAIN_CUTOFF_DATE = "2019-01-29"   # training window boundary
N_CLUSTERS = 3
RFM_RANDOM_STATE = 42

# High-risk cluster identification strategy:
# After clustering, rank clusters by composite RFM score.
# Cluster with lowest Frequency + lowest Monetary = highest risk.
# Recency is inverted — higher recency days = less recent = higher risk.
RISK_RANK_WEIGHTS = {
    "Recency": 1,    # higher days since last tx = worse
    "Frequency": -1,   # lower frequency = worse
    "Monetary": -1,   # lower monetary = worse
}
