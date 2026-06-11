## Credit Risk Intelligence Platform

An end-to-end machine learning and MLOps project for building an alternative credit risk scoring system using transactional behavioral data. The project simulates a Buy-Now-Pay-Later (BNPL) lending scenario in which traditional credit history and loan default records are unavailable. Instead, customer transaction behavior is used to construct a proxy risk label and train predictive models that estimate customer risk levels.

The project covers the complete machine learning lifecycle, including exploratory data analysis, feature engineering, proxy target construction, model development, experiment tracking with MLflow, model explainability, API deployment with FastAPI, containerization with Docker, and CI/CD automation using GitHub Actions.

---

# Business Problem

Bati Bank is partnering with an e-commerce platform to launch a Buy-Now-Pay-Later (BNPL) lending product. Since the platform has no historical lending records, traditional credit scoring approaches cannot be applied directly.

The objective is to leverage alternative transactional data to:

* Identify potentially high-risk customers.
* Estimate customer risk probabilities.
* Generate credit risk scores.
* Support lending decisions in the absence of conventional credit history.

---

# Project Objectives

The project aims to:

1. Construct a proxy credit risk target using customer behavioral patterns.
2. Engineer predictive customer-level features from transaction history.
3. Train and evaluate multiple machine learning models.
4. Compare interpretable and high-performance approaches.
5. Track experiments and model versions using MLflow.
6. Deploy the best-performing model as a production-ready API.
7. Implement testing, containerization, and CI/CD practices.

---

# Project Structure

```text
credit-risk-model/
├── .github/
│   └── workflows/
│       └── ci.yml

├── artifacts/
├── data/
│   ├── raw/
│   └── processed/

├── docs/
│   ├── assumptions.md
│   ├── leakage_audit.md
│   └── feature_catalog.md

├── notebooks/
│   └── 1.0-eda.ipynb

├── reports/

├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_processing.py
│   ├── rfm.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py

├── tests/
│   └── test_data_processing.py

├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
└── .gitignore
```

---

# Technology Stack

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-Learn
* XGBoost
* Imbalanced-Learn

### Explainability

* SHAP

### Experiment Tracking

* MLflow

### API Deployment

* FastAPI
* Uvicorn

### MLOps

* Docker
* GitHub Actions
* Pytest
* Flake8

---

# Assumptions

The project relies on the following assumptions:

### Assumption 1

Customer transaction behavior contains meaningful information about future financial risk.

### Assumption 2

Customers with low engagement patterns may exhibit higher risk characteristics than highly engaged customers.

### Assumption 3

Recency, Frequency, and Monetary (RFM) metrics can be used to construct a useful proxy target in the absence of actual default labels.

These assumptions are documented and critically evaluated throughout the project.

---

# Limitations

This project does not contain actual loan repayment or default data.

As a result, the target variable used for training is a proxy constructed from customer behavioral patterns rather than true credit default outcomes.

Consequently:

* The model estimates behavioral risk rather than actual probability of default.
* Proxy labels may not perfectly represent customer creditworthiness.
* Results should be interpreted as alternative risk signals rather than final lending decisions.
* The model should be validated against real default outcomes if such data becomes available in the future.

---

# Credit Scoring Business Understanding

## 1. How does the Basel II Accord's emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II framework requires financial institutions to use risk measurement approaches that are transparent, auditable, and defensible. In a credit risk setting, model performance alone is not sufficient; institutions must also be able to explain how risk estimates are produced and justify lending decisions to regulators, auditors, and internal stakeholders.

This requirement influences the modeling approach used in this project. Alongside more complex machine learning models such as Gradient Boosting Machines (GBM), an interpretable baseline model based on Logistic Regression is developed. Logistic Regression provides clear relationships between input features and predicted risk, making it easier to understand and validate model behavior.

In addition, all assumptions, feature engineering decisions, model evaluation results, and limitations are documented throughout the project. This ensures that the model development process remains transparent and reproducible, which is essential in regulated financial environments.

---

## 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

The dataset does not contain information about actual loan repayments or customer defaults. As a result, there is no direct target variable that can be used to train a traditional credit risk model. To address this limitation, a proxy target is created using customer behavioral patterns derived from Recency, Frequency, and Monetary (RFM) metrics.

Customers are segmented using K-Means clustering based on their RFM profiles. The least engaged customer segment is then labeled as high risk (`is_high_risk = 1`), while all other customers are labeled as low risk (`is_high_risk = 0`).

This approach allows a supervised machine learning model to be trained, but it introduces important risks. Low engagement does not necessarily imply a high probability of default. Customers may become inactive for reasons unrelated to creditworthiness, such as changing platforms, seasonal purchasing behavior, or reduced spending needs.

Consequently, the resulting model should be viewed as an alternative behavioral risk assessment system rather than a true probability-of-default model. Predictions generated from this proxy target must be interpreted with caution until actual repayment and default data become available for validation.

---

## 3. What are the key trade-offs between using a simple, interpretable model and a complex, high-performance model in a regulated financial context?

In regulated credit risk environments, model selection involves balancing predictive performance against interpretability and governance requirements.

Logistic Regression offers several advantages. Its predictions are easy to explain, feature contributions can be interpreted directly, and the model is generally easier to validate and audit. These characteristics make it a common choice in traditional credit scoring systems and highly suitable for regulatory review.

Gradient Boosting Models (GBM) are often capable of achieving higher predictive performance because they can capture complex non-linear relationships and interactions between variables. However, their decision-making process is less transparent and typically requires additional explainability techniques to understand model behavior.

The trade-off is therefore between transparency and predictive power. Logistic Regression provides stronger interpretability and governance benefits, while GBM may offer improved predictive accuracy. For this reason, both approaches are evaluated in this project. Logistic Regression serves as an interpretable benchmark, while GBM is assessed to determine whether any performance improvement justifies the additional complexity.

---

# Current Project Status

* [x] Project structure initialized
* [x] Environment configured
* [x] Dependencies installed
* [x] Business understanding completed
* [ ] Exploratory Data Analysis
* [ ] Feature Engineering Pipeline
* [ ] Proxy Target Construction
* [ ] Model Training & MLflow Tracking
* [ ] Explainability Analysis
* [ ] FastAPI Deployment
* [ ] Dockerization
* [ ] CI/CD Automation
* [ ] Final Evaluation Report
