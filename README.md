# Credit Risk Intelligence Platform

### End-to-End Alternative Credit Scoring & MLOps for Buy-Now-Pay-Later Lending

> Building a production-oriented credit risk scoring system using alternative transactional data, interpretable machine learning, experiment tracking, and MLOps best practices.

---

## Overview

Traditional credit scoring relies on historical loan repayment records and credit bureau information. However, many financial institutions serving emerging markets or launching new lending products lack sufficient historical credit data.

This project simulates a real-world Buy-Now-Pay-Later (BNPL) lending scenario in which **Bati Bank** partners with an e-commerce platform to offer digital credit. Because no historical loan defaults exist, a behavioral proxy target is constructed from customer transaction data using **Recency, Frequency, and Monetary (RFM)** analysis.

The objective is to build an end-to-end machine learning pipeline capable of estimating customer credit risk while emphasizing transparency, reproducibility, and production readiness.

---

# Business Problem

The bank must answer a critical question:

> **"Can we identify potentially risky customers before extending credit, even when no historical default data exists?"**

To address this challenge, this project develops an alternative credit scoring system that:

* constructs a behavioral proxy for credit risk,
* engineers customer-level behavioral features,
* compares interpretable and complex machine learning models,
* tracks experiments using MLflow,
* prepares the model for production deployment.

---

# Project Objectives

* Build an alternative credit risk model using transactional behavioral data.
* Construct a proxy target using customer engagement patterns.
* Compare interpretable and high-performance machine learning models.
* Produce calibrated probability estimates suitable for financial decision making.
* Track experiments and model versions using MLflow.
* Deploy the final model through a production-ready FastAPI service.
* Demonstrate software engineering and MLOps best practices.

---

# Repository Structure

```text
credit-risk-model/
│
├── artifacts/
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   ├── assumptions.md
│   ├── leakage_audit.md
│   └── feature_catalog.md
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_proxy_target_analysis.ipynb
│   └── 03_feature_analysis.ipynb
│
├── reports/
│
├── src/
│   ├── config.py
│   ├── data_processing.py
│   ├── rfm.py
│   ├── evaluate.py
│   ├── train.py
│   ├── predict.py
│   └── api/
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# Technology Stack

| Category            | Tools                                 |
| ------------------- | ------------------------------------- |
| Language            | Python                                |
| Data Processing     | Pandas, NumPy                         |
| Visualization       | Matplotlib, Seaborn                   |
| Machine Learning    | Scikit-Learn, XGBoost                 |
| Explainability      | SHAP                                  |
| Feature Engineering | Scikit-Learn Pipelines, RFM, WoE / IV |
| Experiment Tracking | MLflow                                |
| API                 | FastAPI                               |
| Testing             | Pytest                                |
| Containerization    | Docker                                |
| CI/CD               | GitHub Actions                        |

---

# Credit Scoring Business Understanding

## Basel II and Explainability

Financial institutions operating under Basel II and Basel III are expected to build models that are transparent, auditable, and defensible. High predictive accuracy alone is insufficient; lending decisions must also be explainable to regulators, auditors, and business stakeholders.

For this reason, the project develops both an interpretable Logistic Regression model and a Gradient Boosting model. Logistic Regression provides a transparent baseline whose predictions are easier to understand and validate, while Gradient Boosting serves as a benchmark for potential performance improvements.

---

## Why a Proxy Target?

The dataset contains transactional behavior but **does not contain actual loan repayment or default information**.

Without a true default label, supervised learning is not possible.

A proxy target is therefore constructed using customer engagement behavior:

* Recency
* Frequency
* Monetary value

Customers are segmented using K-Means clustering, and the least engaged segment is labeled as high risk.

This proxy enables supervised model development while acknowledging that behavioral disengagement is **not equivalent to confirmed credit default**.

---

## Model Selection in a Regulated Environment

Two model families are evaluated.

### Logistic Regression

* Highly interpretable
* Easy to audit
* Stable probability estimates after calibration
* Lower governance complexity

### Gradient Boosting

* Captures complex nonlinear relationships
* Often provides stronger predictive performance
* Requires post-hoc explainability techniques
* More complex to monitor and maintain

Both models are trained and evaluated using identical feature sets and compared using discrimination, calibration, and business-oriented evaluation metrics.

---

# Exploratory Data Analysis

Key findings from the dataset include:

* **95,662 transactions** across **3,742 unique customers**
* **90-day observation period** from November 2018 to February 2019
* No missing values
* Extremely right-skewed transaction amounts
* Strong correlation between `Amount` and `Value`
* Customer behavior is highly heterogeneous, motivating customer-level feature engineering
* No historical default labels, confirming the need for proxy target construction

---

# Feature Engineering

Customer-level features are generated using a production-oriented Scikit-Learn pipeline.

Features include:

* Transaction count
* Total transaction value
* Average transaction amount
* Transaction standard deviation
* Temporal features
* Behavioral aggregates
* Encoded categorical variables
* Standardized numerical variables

Feature engineering is implemented entirely within reusable Python modules rather than notebooks.

---

# Proxy Target Construction

A behavioral proxy target is created using the following workflow:

```
Transactions
      │
      ▼
Customer Aggregation
      │
      ▼
Recency • Frequency • Monetary
      │
      ▼
Robust Scaling
      │
      ▼
K-Means Clustering
      │
      ▼
High-Risk Segment
```

Silhouette analysis was performed to evaluate cluster quality before selecting the final segmentation strategy.

---

# Model Development

The project trains and compares multiple supervised learning models.

Current models include:

* Logistic Regression
* Gradient Boosting

Evaluation includes:

* ROC-AUC
* Precision-Recall AUC
* F1 Score
* Recall
* Expected Calibration Error (ECE)
* Probability calibration
* Cost-sensitive threshold optimization

Experiments are tracked using MLflow, and trained models are versioned for reproducibility.

---

# Current Results

| Model               |    ROC-AUC |         F1 |      Calibration |
| ------------------- | ---------: | ---------: | ---------------: |
| Logistic Regression | **0.7423** |     0.8064 | ECE = **0.0704** |
| Gradient Boosting   |     0.6801 | **0.8091** |     ECE ≈ 0.0000 |

Logistic Regression was selected as the primary model because it achieved the strongest discriminatory performance while remaining highly interpretable, making it more appropriate for a regulated credit-risk setting.

---

# Project Limitations

This project intentionally highlights several real-world challenges.

* No actual loan default labels are available.
* The target variable is a behavioral proxy rather than true probability of default.
* Customer disengagement does not necessarily imply future credit default.
* The proxy target produces a highly imbalanced risk distribution, which influences threshold optimization.
* Operational decision thresholds may differ from statistically optimal thresholds due to business policy and governance requirements.

These limitations are explicitly documented to ensure transparency and reproducibility.

---

# Current Project Status

| Task                         | Status |
| ---------------------------- | :----: |
| Business Understanding       |    ✅   |
| Exploratory Data Analysis    |    ✅   |
| Feature Engineering Pipeline |    ✅   |
| Proxy Target Construction    |    ✅   |
| Model Training               |    ✅   |
| MLflow Tracking              |    ✅   |
| Probability Calibration      |    ✅   |
| SHAP Explainability          |    ✅   |
| FastAPI Deployment           |    ⏳   |
| Dockerization                |    ⏳   |
| GitHub Actions CI/CD         |    ⏳   |
| Production Monitoring        |    ⏳   |

---

# Future Work

The remaining implementation focuses on production deployment.

* FastAPI inference service
* Docker containerization
* GitHub Actions CI/CD
* Prediction logging
* Health monitoring
* Model versioning
* Production-ready deployment pipeline

---

# License


