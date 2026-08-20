# MFI Loan Repayment Prediction

An end-to-end machine learning project for predicting whether a customer will repay a 5-day microloan using historical customer and loan information.

The project focuses on building a reliable binary classification pipeline with leakage-safe temporal feature engineering, robust preprocessing, imbalanced classification, systematic model comparison, hyperparameter tuning, validation-based threshold optimization, and final evaluation on an untouched test set.

## Overview

The objective is to predict the repayment outcome of a microloan:

- `1` — Repaid
- `0` — Defaulter

The complete workflow covers data preparation, feature engineering, model development, validation, model selection, threshold optimization, final evaluation, and model persistence.

## Approach

```text
Raw Data
   ↓
Data Cleaning & Validation
   ↓
Train / Validation / Test Split
   ↓
Leakage-Safe Feature Engineering
   ↓
Preprocessing
   ↓
Model Screening
   ↓
Cross-Validation & Hyperparameter Tuning
   ↓
Champion Model Selection
   ↓
Threshold Optimization
   ↓
Final Test Evaluation
   ↓
Model Persistence

Data Preparation

The dataset is split into:

70% Training
15% Validation
15% Test

Preprocessing includes:

Missing-value imputation
Winsorization of numerical features
Categorical encoding
Feature scaling using StandardScaler

All data-dependent transformations are fitted using training data only.

Feature Engineering

Customer-history features are generated chronologically so that only information available before the current loan is used.

This prevents future loan information from leaking into the prediction process and makes the engineered features representative of information available at prediction time.

Model Development

The project evaluates 45 model configurations across 12+ machine learning algorithm families, including:

Logistic Regression
Ridge Classifier
SGD Classifier
K-Nearest Neighbors
Naive Bayes
Decision Tree
Random Forest
Extra Trees
Gradient Boosting
AdaBoost
Bagging
SVM
MLP
XGBoost
LightGBM
CatBoost

Models are evaluated using Stratified K-Fold Cross-Validation.

Class Imbalance

The project accounts for class imbalance using model-specific weighting techniques such as:

class_weight
scale_pos_weight

The original class distribution is preserved rather than relying on direct resampling.

Evaluation

PR-AUC (Average Precision) is used as the primary model-selection metric because the default class is the minority and is important for repayment-risk assessment.

Additional metrics include:

ROC-AUC
Log Loss
Precision
Recall
F1 Score
Matthews Correlation Coefficient (MCC)
Model Selection

After model screening and hyperparameter tuning, CatBoost was selected as the validation champion.

Validation performance of the leading tuned models:

CatBoost
PR-AUC  : 0.9849
ROC-AUC : 0.9096


XGBoost
PR-AUC  : 0.9844
ROC-AUC : 0.9065


LightGBM
PR-AUC  : 0.9835
ROC-AUC : 0.9018

CatBoost achieved the highest validation PR-AUC among the final tuned candidates.

Threshold Optimization

Instead of using the default probability threshold of 0.50, the classification threshold was optimized using the validation set.

Selected threshold:

0.1080

Validation performance at the selected threshold:

Precision : 0.918
Recall    : 0.987
F1 Score  : 0.952

The test set was not used for threshold selection.

Final Test Results

After finalizing the model and threshold, the pipeline was evaluated once on the untouched test set.

Log Loss  : 0.3745
PR-AUC    : 0.9841
ROC-AUC   : 0.9062
Precision : 0.9186
Recall    : 0.9880
F1 Score  : 0.9520
MCC       : 0.5258

Final class-level performance:

Defaulter (0)
Precision : 0.82
Recall    : 0.39
F1 Score  : 0.53


Repaid (1)
Precision : 0.92
Recall    : 0.99
F1 Score  : 0.95
Leakage Prevention

Data leakage prevention is a core part of the project:

Train, validation, and test data are separated before preprocessing.
Imputation parameters are learned only from training data.
Winsorization limits are learned only from training data.
Scaling parameters are learned only from training data.
Categorical transformations are learned from training data.
Historical features respect chronological ordering.
Future loan information is not used in historical features.
The test set is excluded from model selection.
Threshold optimization uses validation data only.
Final test evaluation is performed only after model development is complete.
Technologies
Python
Pandas
NumPy
Scikit-learn
CatBoost
XGBoost
LightGBM
Matplotlib
Seaborn
Joblib

Key Skills

Python · Machine Learning · Binary Classification · Data Preprocessing · Feature Engineering · Temporal Feature Engineering · Cross-Validation · Hyperparameter Tuning · Imbalanced Classification · Model Evaluation · Threshold Optimization · Data Leakage Prevention · Model Persistence

Future Improvements
Deploy the model through a REST API
Build an interactive loan-risk prediction dashboard
Add SHAP-based model explainability
Implement experiment tracking
Add automated model retraining and monitoring
Develop a real-time loan-risk scoring service
Author

Akshat Sajwan
B.Tech CSE | Machine Learning

Disclaimer

This project is intended for educational and portfolio purposes. Model predictions should not be used as the sole basis for real-world lending decisions without appropriate validation, monitoring, fairness assessment, regulatory review, and domain-specific risk controls.
