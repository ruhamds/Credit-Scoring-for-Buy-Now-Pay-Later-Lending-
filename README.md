# Credit Risk Probability Model — Bati Bank BNPL

An end-to-end machine learning platform that predicts customer credit risk for a Buy Now Pay Later (BNPL) product using behavioral transaction data.

**Key capabilities**

- End-to-end ML pipeline
- Alternative credit scoring using RFM proxy labels
- Basel II inspired model development
- MLflow experiment tracking
- FastAPI prediction service
- Dockerized deployment
- GitHub Actions CI/CD

### Key Features

- Full pipeline: raw transactions → engineered proxy labels → trained,
  calibrated model → deployed API
- FastAPI scoring service with SHAP explainability and PostgreSQL audit logging
- Dockerized (multi-stage build) with GitHub Actions CI/CD (lint, test, build)
- MLflow experiment tracking and model registry
- Logistic Regression selected over Gradient Boosting after head-to-head evaluation
- **ROC-AUC: 0.7423** · **PR-AUC: 0.8141** · 27/27 tests passing

---

## Project Status

✅ Feature Engineering

✅ Proxy Target Engineering

✅ Model Training

✅ Model Explainability

✅ MLflow Tracking

✅ FastAPI Deployment

✅ Docker

✅ GitHub Actions

✅ PostgreSQL Prediction Logging

✅ 27 Automated Tests

## Architecture

```
Raw transactions (Xente, 95,662 rows, 90-day window)
        │
        ▼
Temporal split (train < 2019-01-29, test ≥ 2019-01-29)
        │
        ├──► RFM proxy labeling (training window only) ──► is_high_risk
        │
        ▼
Feature pipeline (sklearn Pipeline, fit on train only)
        │
        ▼
Manual IV feature selection
        │
        ▼
Model training (LR + GBM) ──► MLflow tracking + registry
        │
        ▼
Calibration + cost-based threshold optimization
        │
        ▼
FastAPI service ──► SHAP explainability ──► PostgreSQL audit log
        │
        ▼
Docker Compose (api + db + mlflow) ──► GitHub Actions CI/CD
```

---

## Results

| Metric | Logistic Regression | Gradient Boosting |
|---|---|---|
| ROC-AUC | **0.7423** | 0.6801 |
| PR-AUC | **0.8141** | 0.7534 |
| F1 (high-risk) | 0.8064 | 0.8091 |
| ECE after calibration | 0.0704 | 0.0000* |

**Logistic Regression selected** — better ROC-AUC/PR-AUC, more stable
calibration, and full coefficient interpretability for Basel II compliance.
*GBM's perfect ECE likely reflects isotonic regression overfitting on a small
calibration set — see [docs/model_evaluation.md](docs/model_evaluation.md).

**SHAP global importance (Logistic Regression)** — transaction frequency
(`Amount_count`) dominates by a wide margin, validating the core proxy
hypothesis that engagement frequency is the strongest behavioral risk signal:

![SHAP summary](docs/shap_lr_summary.png)

**Calibration curve (Logistic Regression)** — the calibrated curve tracks the
diagonal more closely than the raw model output, though variance remains
high in the 0.6–0.85 range due to the small test set (1,056 rows):

![Calibration curve](docs/calibration_lr.png)

Full evaluation detail, including the Gradient Boosting plots, is in
[docs/model_evaluation.md](docs/model_evaluation.md).

---

## Tech Stack

Python 3.11 • Scikit-learn • FastAPI • PostgreSQL • Docker • MLflow • SHAP • GitHub Actions

---

## Quickstart

```bash
git clone <https://github.com/ruhamds/Credit-Scoring-for-Buy-Now-Pay-Later-Lending-.git>
cd credit-risk-model
docker compose up --build
```

```bash
curl http://localhost:8000/health
curl http://localhost:8000/schema
```

MLflow UI: `http://localhost:5000`

**Example prediction**:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Amount_count": 12.0, "Amount_sum": 87.3, "Amount_mean": 7.3,
    "Amount_std": 2.1, "Amount_min": 3.4, "Amount_max": 11.2,
    "tx_hour_mean": 14.5, "tx_day_mean": 15.2, "tx_month_mean": 11.8,
    "tx_dayofweek_mean": 2.3, "tx_is_weekend_mean": 0.17,
    "ProviderId_mode": 3.0, "ChannelId_mode": 2.0,
    "ProductCategory_mode": 1.0
  }'
```

```json
{
  "prediction_id": "264dc039-5a1a-4e4f-8c4e-700d35708c80",
  "risk_score": 0.1791,
  "decision": "approve",
  "top_factors": [
    {"feature": "Amount_count", "impact": -30.83, "direction": "decreases_risk"},
    {"feature": "Amount_sum", "impact": -13.45, "direction": "decreases_risk"},
    {"feature": "Amount_mean", "impact": 11.41, "direction": "increases_risk"}
  ],
  "model_version": "1.0.0",
  "threshold_used": 0.3
}
```

> **API contract**: `/predict` expects pipeline-transformed, customer-level
> features (not raw transactions). See [docs/deployment.md](docs/deployment.md).

---

## Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml
├── docs/                       # full methodology — see below
├── notebooks/
├── src/
│   ├── config.py
│   ├── data_processing.py
│   ├── rfm.py
│   ├── train.py
│   ├── evaluate.py
│   └── api/
├── tests/                      # 27 tests, all passing
├── reports/                    # raw plots from training runs
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## My Contributions

- Designed proxy target generation using RFM segmentation (KMeans, k=3)
  with outlier-robust preprocessing
- Built the full feature engineering pipeline as a single `sklearn.Pipeline`
  with zero data leakage between train/test
- Implemented manual Information Value (IV) feature selection after
  diagnosing an `xverse` library incompatibility
- Trained, tuned, and calibrated both Logistic Regression and Gradient
  Boosting models with MLflow experiment tracking
- Built a cost-sensitive decision threshold optimizer reflecting real
  business asymmetry between false positives and false negatives
- Built the FastAPI prediction service with SHAP explainability and
  PostgreSQL audit logging
- Containerized the full stack (API, database, MLflow) with a multi-stage
  Dockerfile and Docker Compose
- Configured GitHub Actions CI/CD (lint, test, Docker build)

---

## Documentation

Full methodology, business rationale, and design tradeoffs:

- [docs/business_understanding.md](docs/business_understanding.md) — Basel II
  rationale, proxy label risk, LR vs. GBM tradeoffs
- [docs/proxy_target.md](docs/proxy_target.md) — RFM construction, clustering
  diagnostics, silhouette analysis, why 84.5% high-risk is accepted
- [docs/model_evaluation.md](docs/model_evaluation.md) — full metrics,
  calibration analysis, threshold optimization, SHAP plots for both models
- [docs/deployment.md](docs/deployment.md) — API contract, Docker setup,
  CI/CD pipeline, known limitations

## Limitations

This model uses an unvalidated behavioral proxy for credit risk — it has not
been validated against actual default outcomes and should be treated as a
first-generation risk signal, not a production lending decision engine. Full
discussion in [docs/proxy_target.md](docs/proxy_target.md) and
[docs/deployment.md](docs/deployment.md).

## Future Work

- Replace the proxy target with real repayment outcomes once loan performance data becomes available.
- Monitor model drift and recalibrate probabilities over time.
- Introduce automated retraining pipelines.
- Extend the API with batch prediction endpoints.
- Deploy to a cloud platform (Azure, AWS, or GCP).