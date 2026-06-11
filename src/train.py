import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_model(X, y, params=None):
    params = params or {}
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("train_score", model.score(X_train, y_train))
        mlflow.log_metric("val_score", model.score(X_val, y_val))
        mlflow.sklearn.log_model(model, "model")

    return model
