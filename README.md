# 💳 MFI Loan Repayment Prediction

### End-to-End Machine Learning Pipeline for Microloan Repayment Risk

An end-to-end machine learning project that predicts whether a customer will **repay or default on a 5-day microloan** using historical customer and loan information.

The project is designed as a **leakage-aware, evaluation-driven machine learning pipeline**, covering temporal feature engineering, robust preprocessing, imbalanced classification, systematic model comparison, hyperparameter tuning, threshold optimization, and final evaluation on an untouched test set.

> **Core Principle:** Only information that would have been available before the current loan is used for prediction.

---

## 🚀 Highlights

- 🧠 **45 model configurations** evaluated across **12+ ML algorithm families**
- 🕒 Leakage-safe **temporal feature engineering**
- 🔀 **70 / 15 / 15** train-validation-test split
- ⚖️ Imbalanced classification with model-specific weighting
- 🔬 Stratified K-Fold cross-validation
- 🔧 Hyperparameter tuning
- 🏆 **CatBoost selected as the validation champion**
- 🎯 Validation-based threshold optimization
- 📊 PR-AUC as the primary model-selection metric
- 🔒 Completely untouched test set for final evaluation
- 💾 Final model pipeline persisted with Joblib

---

## 🎯 Objective

The objective is to predict the repayment outcome of a short-term microloan:

| Label | Outcome |
|:---:|---|
| `1` | Repaid |
| `0` | Defaulter |

The project focuses not only on predictive performance, but also on ensuring that the evaluation process is **realistic, reproducible, and protected against data leakage**.

---

## 🔄 End-to-End Workflow

**Raw Data → Validation → Train/Validation/Test Split → Temporal Feature Engineering → Preprocessing → Model Screening → Cross-Validation → Hyperparameter Tuning → Champion Selection → Threshold Optimization → Final Test Evaluation → Model Persistence**

---

## 🗂️ Dataset

The project uses historical customer and microloan information together with the accompanying data-description workbook.

**Documentation:** `Micro-credit-card-Data-Description.xlsx`

The workbook is used to understand the available variables and their meaning before feature engineering and model development.

---

## 🔀 Data Split

| Split | Percentage | Purpose |
|---|---:|---|
| Training | **70%** | Model training and preprocessing |
| Validation | **15%** | Model selection, tuning and threshold optimization |
| Test | **15%** | Final unbiased evaluation |

The **test set remains untouched** until model development is complete.

---

## 🕒 Leakage-Safe Temporal Feature Engineering

Customer borrowing history is inherently chronological.

Historical features are therefore generated using only information that existed **before the current loan**.

**Example:**

`Loan 1 → Loan 2 → Loan 3 → Loan 4`

When predicting **Loan 3**, information from **Loan 4** must not be used.

This prevents future loan information from leaking into the prediction process and makes the engineered features representative of information available at prediction time.

> **If the information would not have existed when the loan was evaluated, the model should not use it.**

---

## 🛡️ Leakage Prevention

The pipeline ensures:

- Train, validation and test data are separated before preprocessing.
- Imputation parameters are learned from training data only.
- Winsorization limits are learned from training data only.
- Scaling parameters are learned from training data only.
- Categorical transformations are learned from training data only.
- Historical features respect chronological ordering.
- Future loan information is excluded from historical features.
- The test set is excluded from model selection.
- Hyperparameter tuning does not use the final test set.
- Threshold optimization uses validation data only.
- Final test evaluation occurs only after model development is complete.

---

## ⚙️ Preprocessing

The preprocessing pipeline includes:

- Missing-value imputation
- Numerical winsorization
- Categorical encoding
- Feature scaling using `StandardScaler`

All data-dependent transformations are fitted using training data only.

---

## 🤖 Model Development

Multiple machine learning families were evaluated rather than assuming a single algorithm would perform best.

**Algorithms include:**

Logistic Regression · Ridge Classifier · SGD Classifier · KNN · Naive Bayes · Decision Tree · Random Forest · Extra Trees · Gradient Boosting · AdaBoost · Bagging · SVM · MLP · XGBoost · LightGBM · CatBoost

Overall, the project evaluates:

> **45 model configurations across 12+ machine learning algorithm families.**

Models were compared using **Stratified K-Fold Cross-Validation**.

---

## ⚖️ Class Imbalance

The project accounts for class imbalance using model-specific weighting strategies where supported:

- `class_weight`
- `scale_pos_weight`

The original class distribution is preserved rather than relying on direct resampling.

---

## 📊 Evaluation Strategy

**PR-AUC / Average Precision** is used as the primary model-selection metric.

Additional metrics include:

| Metric | Purpose |
|---|---|
| **PR-AUC** | Primary model-selection metric |
| **ROC-AUC** | Ranking and discrimination |
| **Log Loss** | Probability quality |
| **Precision** | Positive prediction reliability |
| **Recall** | Positive-case coverage |
| **F1 Score** | Precision-recall balance |
| **MCC** | Correlation-based evaluation |

---

## 🏆 Champion Model

After model screening and hyperparameter tuning, **CatBoost** was selected as the validation champion.

| Rank | Model | Validation PR-AUC | Validation ROC-AUC |
|:---:|---|---:|---:|
| 🥇 | **CatBoost** | **0.9849** | **0.9096** |
| 🥈 | XGBoost | 0.9844 | 0.9065 |
| 🥉 | LightGBM | 0.9835 | 0.9018 |

CatBoost achieved the highest validation PR-AUC among the final tuned candidates.

---

## 🎯 Threshold Optimization

The default classification threshold of `0.50` was not assumed to be optimal.

Instead, the threshold was optimized using the **validation set**.

### Selected Threshold

**`0.1080`**

### Validation Performance

| Metric | Result |
|---|---:|
| Precision | **0.918** |
| Recall | **0.987** |
| F1 Score | **0.952** |

The test set was **not used** for threshold selection.

---

## 📈 Final Test Performance

After finalizing the model and threshold, the complete pipeline was evaluated once on the untouched test set.

| Metric | Test Result |
|---|---:|
| **Log Loss** | **0.3745** |
| **PR-AUC** | **0.9841** |
| **ROC-AUC** | **0.9062** |
| **Precision** | **0.9186** |
| **Recall** | **0.9880** |
| **F1 Score** | **0.9520** |
| **MCC** | **0.5258** |

### Class-Level Performance

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| **Defaulter (0)** | 0.82 | 0.39 | 0.53 |
| **Repaid (1)** | 0.92 | 0.99 | 0.95 |

Class-level metrics are reported separately to provide a more transparent view of model behavior.

---

## 💡 Why Threshold Optimization Matters

The probability threshold changes the balance between precision and recall.

**Lower threshold → more positive predictions → potentially higher recall and lower precision**

**Higher threshold → fewer positive predictions → potentially higher precision and lower recall**

The threshold was therefore treated as a **model-development decision**, rather than blindly using the conventional `0.50`.

---

## 💾 Model Persistence

The finalized model pipeline is persisted using **Joblib**, allowing the trained model and preprocessing workflow to be reused for future inference without retraining from scratch.

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, CatBoost, XGBoost, LightGBM |
| Visualization | Matplotlib, Seaborn |
| Model Persistence | Joblib |

---

## 🧠 Skills Demonstrated

`Python` · `Machine Learning` · `Binary Classification` · `Feature Engineering` · `Temporal Feature Engineering` · `Data Leakage Prevention` · `Cross-Validation` · `Hyperparameter Tuning` · `Imbalanced Classification` · `Model Comparison` · `Model Evaluation` · `PR-AUC` · `ROC-AUC` · `Threshold Optimization` · `Model Persistence`

---

## 📁 Project Structure

    MFI-Loan-Repayment-Prediction/
    │
    ├── MFI_Loan_Repayment_Prediction.ipynb
    ├── Micro-credit-card-Data-Description.xlsx
    └── README.md

---

## 🔮 Future Scope

- 🌐 REST API deployment
- 📊 Interactive loan-risk prediction dashboard
- 🔍 SHAP-based model explainability
- 🧪 Experiment tracking and model versioning
- 📡 Feature and prediction drift monitoring
- 🔄 Automated model retraining
- ⚔️ Champion-versus-challenger model evaluation
- 🚀 Production-style inference service

---

## ⚠️ Limitations

This is a **student-built educational and portfolio project**, not a production lending system.

Key limitations include:

- Performance depends on the quality and representativeness of the historical data.
- Historical behavior may not fully represent future customer behavior.
- Performance may change when applied to a different lending population.
- The project does not establish production-level fairness or regulatory compliance.
- Different business objectives may require a different classification threshold.
- Customer behavior and lending patterns can change over time.

---

## 🛡️ Responsible Use

Real-world lending systems require additional controls beyond predictive performance, including:

- Fairness assessment
- Explainability
- Privacy and data governance
- Regulatory review
- Continuous monitoring
- Human oversight
- Domain-specific risk controls

This project should therefore be treated as a **machine learning decision-support prototype**, not an automatic lending decision system.

---

## 📜 Disclaimer

This project is intended for **educational and portfolio purposes**.

The reported results are specific to the evaluated dataset and experimental methodology. They should not be interpreted as evidence that the model is suitable for real-world lending or financial decision-making.

Real-world deployment would require appropriate validation, fairness assessment, privacy controls, regulatory review, monitoring, human oversight and domain-specific risk management.

---

## 👨‍💻 Author

**Akshat Sajwan**  
**B.Tech CSE | Machine Learning**
