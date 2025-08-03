# Binary Classification with a Bank Dataset

This project performs binary classification to predict whether a customer will subscribe to a term deposit. The workflow includes data preprocessing, exploratory data analysis (EDA), model training, hyperparameter tuning, and final evaluation using XGBoost.

---

## Dataset Overview

The dataset is obtained from a Kaggle competition and consists of:

- **train.csv**: 750,000 records  
- **test.csv**: 250,000 records  

Each row contains customer attributes such as age, job, education, contact type, and the outcome of previous marketing campaigns.  
The target variable `y` indicates whether the customer subscribed to a term deposit:

- `1` = Subscribed  
- `0` = Did not subscribe

---

## Project Workflow

### 1. Data Loading & Cleaning
- Loaded train and test data using `pandas`.
- Dropped irrelevant columns like `id`.
- Checked for null values and data types.

### 2. Exploratory Data Analysis (EDA)
- **Class Distribution**: Only ~12.07% positive class (`y=1`), indicating class imbalance.
- **Categorical Feature Impact**: Features like `job`, `contact`, and `poutcome` influence the likelihood of subscription.
- **Correlation Analysis**: Heatmaps and pairplots were used to understand relationships between numerical variables.

### 3. Data Preprocessing
- Applied `LabelEncoder` to binary categorical features (`default`, `housing`, `loan`).
- Used `OneHotEncoding` for multi-class features (`job`, `education`, `contact`, `month`, `poutcome`, etc.).
- Standardized numerical features using `StandardScaler`.

---

## 4. Model Training and Evaluation

The dataset is split into training and validation sets using an 80/20 stratified split.

| Model                | Accuracy | ROC AUC Score |
|---------------------|----------|---------------|
| Logistic Regression | 91.65%   | 0.9439        |
| Decision Tree       | 90.66%   | 0.7823        |
| K-Nearest Neighbors | 91.78%   | 0.9274        |
| Random Forest       | 93.07%   | 0.9605        |
| XGBoost             | 93.29%   | 0.9654        |

The **XGBoost** model performed best across all metrics and was selected for final prediction.

---

## 5. Hyperparameter Tuning with XGBoost

Hyperparameters were tuned using `RandomizedSearchCV` with 5-fold cross-validation.

### Best Parameters:
- `subsample`: 0.8  
- `reg_lambda`: 1  
- `reg_alpha`: 0.5  
- `n_estimators`: 150  
- `min_child_weight`: 3  
- `max_depth`: 10  
- `learning_rate`: 0.1  
- `gamma`: 0.1  
- `colsample_bytree`: 0.6  

### Performance After Tuning:
- **Accuracy**: 93.42%  
- **ROC AUC Score**: 0.9664  

---

## 6. Final Prediction

- Final predictions made using the tuned XGBoost model.
- Results saved in `submission_4_xgb.csv`.
- Submitted to Kaggle for evaluation.

---

## Final Kaggle Score

- **Public Leaderboard ROC AUC Score**: **0.96450**

---
