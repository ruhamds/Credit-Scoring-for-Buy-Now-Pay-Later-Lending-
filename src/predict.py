import joblib
import pandas as pd


def load_model(path: str):
    return joblib.load(path)


def predict(model, data: pd.DataFrame):
    return model.predict_proba(data)[:, 1]
