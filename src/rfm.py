import pandas as pd
from sklearn.cluster import KMeans


def create_rfm_target(df: pd.DataFrame) -> pd.DataFrame:
    """Generate RFM scores and optional KMeans behavioral target."""
    df = df.copy()
    # Placeholder RFM logic
    df["recency_score"] = 0
    df["frequency_score"] = 0
    df["monetary_score"] = 0
    return df


def fit_kmeans(df: pd.DataFrame, n_clusters: int = 3) -> KMeans:
    model = KMeans(n_clusters=n_clusters, random_state=42)
    features = df[["recency_score", "frequency_score", "monetary_score"]]
    model.fit(features)
    return model
