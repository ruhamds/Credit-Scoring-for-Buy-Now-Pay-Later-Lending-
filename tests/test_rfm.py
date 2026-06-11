import pandas as pd
from src.rfm import create_rfm_target


def test_rfm_target_adds_scores():
    df = pd.DataFrame({"customer_id": [1, 2]})
    result = create_rfm_target(df)
    assert "recency_score" in result.columns
    assert "monetary_score" in result.columns
