
````markdown
# MFI Loan Repayment Prediction

## Machine Learning for Microloan Repayment Risk

An end-to-end machine learning system for predicting whether a customer will repay a short-term **5-day microloan** using historical customer and loan information.

This project goes beyond training a single classifier. It focuses on building a **reliable and leakage-aware machine learning pipeline** covering temporal feature engineering, robust preprocessing, imbalanced classification, systematic model comparison, hyperparameter tuning, validation-based threshold optimization, and final evaluation on an untouched test set.

The goal is not simply to maximize a single metric, but to build a model whose evaluation reflects how information would actually be available at prediction time.

---

## Overview

Microloan repayment prediction is a binary classification problem where the model estimates whether a customer is likely to repay a loan based on information available about the customer and their previous borrowing history.

### Prediction Target

| Value | Meaning |
|------:|---------|
| `1` | Repaid |
| `0` | Defaulter |

The project follows a complete machine learning lifecycle:

```text
Historical Customer + Loan Data
              │
              ▼
      Data Validation
              │
              ▼
     Train / Validation / Test
              │
              ▼
 Leakage-Safe Feature Engineering
              │
              ▼
      Data Preprocessing
              │
              ▼
      Model Screening
              │
              ▼
 Cross-Validation + Tuning
              │
              ▼
       Model Selection
              │
              ▼
    Threshold Optimization
              │
              ▼
      Final Test Evaluation
              │
              ▼
       Model Persistence
````

---

# Problem Statement

For short-term microloans, repayment behavior can depend on historical customer and loan characteristics.

A useful prediction system therefore needs to answer:

> **Given the information available before a customer's current loan, how reliably can we estimate whether that loan will be repaid?**

A major challenge is that historical customer data is naturally time-dependent.

Information from a customer's **future loans must not influence predictions about an earlier loan**.

This makes leakage-safe temporal feature engineering one of the central parts of this project.

---

# What This Project Does

The pipeline performs:

* Data cleaning and validation
* Chronological feature engineering
* Leakage-safe customer history generation
* Train / validation / test splitting
* Numerical preprocessing
* Categorical preprocessing
* Robust outlier handling
* Feature scaling
* Imbalanced classification
* Multi-model benchmarking
* Stratified cross-validation
* Hyperparameter tuning
* Validation-based threshold optimization
* Final evaluation on an untouched test set
* Model persistence for future inference

---

# Dataset

The project uses historical microloan/customer information together with the accompanying data-description file.

The dataset contains information related to customers, loans, and repayment outcomes.

The accompanying data-description workbook is:

`Micro-credit-card-Data-Description.xlsx`

It is used to understand the meaning and structure of the available variables before building the machine learning pipeline.

---

# Data Splitting

The dataset is divided into three independent subsets:

| Split      | Proportion | Purpose                                          |
| ---------- | ---------: | ------------------------------------------------ |
| Training   |        70% | Model learning and preprocessing                 |
| Validation |        15% | Model comparison, tuning and threshold selection |
| Test       |        15% | Final unbiased evaluation                        |

The test set remains untouched during model development.

This is important because using the test set for model selection or threshold optimization would make the final performance estimate overly optimistic.

---

# Leakage-Safe Feature Engineering

One of the most important parts of the project is the construction of historical customer features.

For every loan, historical features are generated **chronologically**.

The model is only allowed to use information that would have been available before the current loan.

Conceptually:

```text
Customer Loan History

Loan 1 ──► Loan 2 ──► Loan 3 ──► Loan 4
  │          │          │          │
  ▼          ▼          ▼          ▼
History     History    History    History
available   available  available  available
before      before     before     before
Loan 1      Loan 2     Loan 3     Loan 4
```

For example, when predicting Loan 3, information from Loan 4 must not be included in the features.

This prevents **future-information leakage** and makes the feature engineering process much closer to a real prediction-time scenario.

---

# Data Preprocessing

The preprocessing pipeline includes:

### Missing Values

Missing numerical and categorical values are handled through preprocessing transformations learned from the training data.

### Winsorization

Numerical features are processed using winsorization to reduce the influence of extreme observations.

### Categorical Encoding

Categorical variables are transformed into numerical representations suitable for machine learning models.

### Feature Scaling

Numerical features are standardized using:

`StandardScaler`

All data-dependent preprocessing parameters are learned from the training data rather than the validation or test sets.

---

# Model Development

Instead of assuming that one algorithm is automatically the best choice, the project performs systematic model screening across multiple machine learning families.

The evaluated algorithms/configurations include:

* Logistic Regression
* Ridge Classifier
* SGD Classifier
* K-Nearest Neighbors
* Naive Bayes
* Decision Tree
* Random Forest
* Extra Trees
* Gradient Boosting
* AdaBoost
* Bagging
* SVM
* MLP
* XGBoost
* LightGBM
* CatBoost

Overall, the project evaluates **45 model configurations across 12+ machine learning algorithm families**.

This provides a broader basis for model selection than relying on a single classifier.

---

# Cross-Validation

Candidate models are evaluated using:

**Stratified K-Fold Cross-Validation**

Stratification helps preserve the class distribution across validation folds, which is especially important when the target classes are not evenly distributed.

The objective is to identify models that perform consistently rather than selecting a model based on a single favorable split.

---

# Handling Class Imbalance

Repayment-risk datasets can contain an uneven distribution between repayment outcomes.

Instead of relying on direct resampling, the project uses model-specific weighting strategies where supported.

Examples include:

* `class_weight`
* `scale_pos_weight`

This allows the original data distribution to remain intact while giving appropriate importance to the minority class during model training.

---

# Evaluation Strategy

Accuracy is not treated as the primary measure of model quality.

The primary model-selection metric is:

### PR-AUC / Average Precision

PR-AUC is particularly useful when the class of interest is relatively less common and when precision-recall behavior matters for identifying repayment risk.

Additional evaluation metrics include:

* PR-AUC
* ROC-AUC
* Log Loss
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

Together, these metrics provide a more complete view of model discrimination, probability quality, and classification behavior.

---

# Champion Model Selection

After model screening and hyperparameter tuning, **CatBoost** was selected as the validation champion.

### Leading Tuned Models

| Model        | Validation PR-AUC | Validation ROC-AUC |
| ------------ | ----------------: | -----------------: |
| **CatBoost** |        **0.9849** |         **0.9096** |
| XGBoost      |            0.9844 |             0.9065 |
| LightGBM     |            0.9835 |             0.9018 |

CatBoost achieved the highest validation PR-AUC among the final tuned candidates.

The selection was therefore based on measured validation performance rather than simply choosing the most complex or popular algorithm.

---

# Threshold Optimization

A probability model does not automatically determine the best classification threshold.

The default threshold of:

`0.50`

was therefore not assumed to be optimal.

Instead, the classification threshold was optimized using the **validation set**.

### Selected Threshold

```text
0.1080
```

### Validation Performance

| Metric    | Validation Result |
| --------- | ----------------: |
| Precision |         **0.918** |
| Recall    |         **0.987** |
| F1 Score  |         **0.952** |

The test set was not used during threshold selection.

This separation helps prevent the final test results from becoming part of the model-development process.

---

# Final Test Evaluation

After the model and threshold were finalized, the pipeline was evaluated once on the untouched test set.

### Final Results

| Metric    | Test Result |
| --------- | ----------: |
| Log Loss  |  **0.3745** |
| PR-AUC    |  **0.9841** |
| ROC-AUC   |  **0.9062** |
| Precision |  **0.9186** |
| Recall    |  **0.9880** |
| F1 Score  |  **0.9520** |
| MCC       |  **0.5258** |

The final results show strong precision-recall performance while maintaining a clear separation between development data and final evaluation data.

---

# Class-Level Performance

The final test performance by class was:

| Class           | Precision | Recall |   F1 |
| --------------- | --------: | -----: | ---: |
| Defaulter (`0`) |      0.82 |   0.39 | 0.53 |
| Repaid (`1`)    |      0.92 |   0.99 | 0.95 |

The class-level results provide additional context that aggregate metrics alone cannot show.

In particular, the model's performance differs substantially between the repayment and defaulter classes, which is important when interpreting the practical behavior of the classifier.

---

# Why the Threshold Matters

A probability threshold changes the balance between different types of predictions.

Conceptually:

```text
Lower Threshold
      │
      ├── More observations classified as positive
      ├── Potentially higher recall
      └── Potentially lower precision
     
Higher Threshold
      │
      ├── Fewer observations classified as positive
      ├── Potentially higher precision
      └── Potentially lower recall
```

Instead of blindly using `0.50`, this project uses validation data to determine a threshold that produces a stronger precision-recall balance for the intended classification task.

---

# Leakage Prevention

Leakage prevention is treated as a first-class design requirement.

The pipeline ensures:

* Train, validation, and test sets are separated before preprocessing.
* Imputation parameters are learned only from training data.
* Winsorization limits are learned only from training data.
* Scaling parameters are learned only from training data.
* Categorical transformations are learned from training data.
* Historical features respect chronological ordering.
* Future loan information is excluded from historical features.
* The test set is excluded from model selection.
* Hyperparameter tuning uses development data rather than the final test set.
* Threshold optimization uses validation data only.
* Final test evaluation happens only after the model-development process is complete.

This is important because a model can appear highly accurate while still being invalid if future information has accidentally entered the feature set.

---

# Model Persistence

The finalized model pipeline is persisted using:

**Joblib**

This allows the trained model and preprocessing workflow to be reused for future inference without retraining from scratch.

---

# Technologies

### Programming & Data

* Python
* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* CatBoost
* XGBoost
* LightGBM

### Visualization

* Matplotlib
* Seaborn

### Model Persistence

* Joblib

---

# Key Skills Demonstrated

`Python`

`Machine Learning`

`Binary Classification`

`Data Preprocessing`

`Feature Engineering`

`Temporal Feature Engineering`

`Data Leakage Prevention`

`Cross-Validation`

`Hyperparameter Tuning`

`Imbalanced Classification`

`Model Comparison`

`Model Evaluation`

`PR-AUC`

`ROC-AUC`

`Threshold Optimization`

`Model Persistence`

---

# Project Structure

```text
MFI-Loan-Repayment-Prediction/
│
├── MFI_Loan_Repayment_Prediction.ipynb
│
├── Micro-credit-card-Data-Description.xlsx
│
└── README.md
```

The notebook contains the end-to-end experimentation and modeling workflow, while the data-description workbook provides information about the dataset variables.

---

# End-to-End Machine Learning Workflow

The complete project can be summarized as:

```text
                ┌───────────────────────┐
                │      Raw Dataset      │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ Data Validation       │
                │ & Cleaning            │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ Train / Validation /  │
                │ Test Split            │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ Temporal Feature      │
                │ Engineering           │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ Preprocessing         │
                │ Imputation / Encoding │
                │ Scaling / Winsorizing │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ Model Screening       │
                │ 45 Configurations     │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ Cross-Validation &    │
                │ Hyperparameter Tuning │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ CatBoost Champion     │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ Validation Threshold  │
                │ Optimization          │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ Untouched Test Set    │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ Final Metrics         │
                │ + Saved Model         │
                └───────────────────────┘
```

---

# Key Takeaways

### 1. Model selection should be evidence-driven

Multiple model families were compared rather than assuming that a particular algorithm would perform best.

### 2. Temporal information matters

Historical customer features must respect the order in which loans occurred.

Using future information would create leakage and produce misleading evaluation results.

### 3. Accuracy is not enough

PR-AUC, ROC-AUC, precision, recall, F1, Log Loss and MCC provide a more informative view of model behavior.

### 4. The classification threshold matters

A model's probability output and its final class decision are different stages of the pipeline.

The threshold was therefore optimized using validation data rather than automatically using `0.50`.

### 5. The test set should remain untouched

The final test results were produced only after model selection and threshold optimization were completed.

---

# Limitations

This is an educational and portfolio-oriented machine learning project rather than a production lending system.

Important limitations include:

* Model performance depends on the quality and representativeness of the available historical data.
* Historical repayment behavior may not fully represent future customer behavior.
* A model trained on one lending dataset may not generalize to another lending environment.
* The current project does not establish production-level fairness or regulatory compliance.
* Threshold selection reflects the evaluated objective and may need to change under different business costs.
* Model performance can change when the underlying customer population or lending process changes.
* The model should not be interpreted as a complete credit-risk or lending-decision system.

---

# Future Improvements

Potential extensions include:

### Deployment

* Deploy the trained pipeline through a REST API.
* Build an interactive loan-risk prediction dashboard.
* Add a production-style inference service.

### Explainability

* Add SHAP-based feature explanations.
* Provide per-prediction reasoning.
* Analyze model behavior across customer segments.

### Monitoring

* Track prediction distributions over time.
* Monitor feature drift.
* Monitor changes in model performance.
* Add automated alerts for significant distribution changes.

### Experiment Management

* Add experiment tracking.
* Record model versions and evaluation results.
* Track hyperparameter experiments systematically.

### Model Lifecycle

* Add automated retraining workflows.
* Introduce scheduled model validation.
* Compare new model versions against the existing champion model.

---

# Responsible Use

Loan repayment prediction can influence financial decisions and therefore requires careful validation beyond model performance alone.

Before using such a system in a real lending environment, additional work would be required around:

* Fairness assessment
* Explainability
* Data governance
* Privacy
* Regulatory requirements
* Model monitoring
* Human oversight
* Domain-specific risk controls

The model should therefore be treated as a **decision-support component**, not as an automatic replacement for responsible lending processes.

---

# Disclaimer

This project is intended for **educational and portfolio purposes**.

The reported results are based on the evaluated dataset and experimental methodology and should not be interpreted as evidence that the model is suitable for real-world lending decisions.

Real-world deployment would require appropriate validation, monitoring, fairness assessment, regulatory review, privacy controls, and domain-specific risk management.

---

# Author

**Akshat Sajwan**

B.Tech CSE | Machine Learning

---

## Project Summary

**MFI Loan Repayment Prediction** demonstrates an end-to-end approach to building a leakage-aware binary classification system for short-term microloan repayment prediction.

Rather than focusing only on model accuracy, the project emphasizes:

**Temporal Feature Engineering → Leakage Prevention → Model Comparison → Cross-Validation → Hyperparameter Tuning → Threshold Optimization → Untouched Test Evaluation → Model Persistence**

The result is a reproducible machine learning workflow designed to make model evaluation more realistic, transparent, and technically defensible.

```
```
