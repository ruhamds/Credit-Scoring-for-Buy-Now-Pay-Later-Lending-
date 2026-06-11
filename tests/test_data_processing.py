import pandas as pd
from src.data_processing import build_feature_pipeline


def test_pipeline_returns_transformer():
    pipeline = build_feature_pipeline()
    df = pd.DataFrame({"feature": [1, 2, None]})
    transformed = pipeline.fit_transform(df)
    assert transformed.shape[0] == 3
