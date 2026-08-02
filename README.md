# Loan Approval Prediction using Ensemble Learning

## Overview

This project implements a machine learning pipeline to predict whether a loan application will be **Approved** or **Rejected** using ensemble learning algorithms.

The project follows a modular architecture where data loading, preprocessing, model training, evaluation, and visualization are separated into independent Python modules.

---

## Objectives

- Predict loan approval status from applicant information.
- Compare the performance of multiple ensemble learning algorithms.
- Evaluate models using standard classification metrics.
- Visualize model performance.
- Save the best-performing model for future inference.

---

## Dataset

The project uses the **Loan Approval Prediction Dataset**.

### Features

| Feature | Description |
|----------|-------------|
| no_of_dependents | Number of dependents |
| education | Graduate / Not Graduate |
| self_employed | Self-employed status |
| income_annum | Annual income |
| loan_amount | Requested loan amount |
| loan_term | Loan duration |
| cibil_score | Credit score |
| residential_assets_value | Residential asset value |
| commercial_assets_value | Commercial asset value |
| luxury_assets_value | Luxury asset value |
| bank_asset_value | Bank asset value |

### Target

| Value | Meaning |
|--------|----------|
| Approved | Loan Approved |
| Rejected | Loan Rejected |

---

# Project Structure

```
Loan_approval_prediction/
│
├── data/
│   └── loan_approval_dataset.csv
│
├── models/
│   └── best_model.pkl
│
├── outputs/
│   └── plots/
│       ├── accuracy_comparison.png
│       ├── roc_curve.png
│       ├── *_cm.png
│       └── *_feature_importance.png
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── main.py
│
├── README.md
├── requirements.txt
└── results.txt
```

---

# Workflow

```
Dataset
    │
    ▼
Data Loading
    │
    ▼
Preprocessing
    │
    ▼
Train / Validation / Test Split
    │
    ▼
Model Training
    │
    ▼
Model Evaluation
    │
    ▼
Visualization
    │
    ▼
Best Model Selection
    │
    ▼
Save Model (.pkl)
```

---

# Data Preprocessing

The preprocessing pipeline performs the following operations:

- Remove leading/trailing whitespace from column names.
- Remove whitespace from categorical values.
- Ordinal encoding of the `education` feature.
- Binary encoding of:
  - self_employed
  - loan_status
- Train / Validation / Test split using stratified sampling.

Dataset Split:

- Training : 70%
- Validation : 15%
- Testing : 15%

---

# Machine Learning Models

The following models are trained and compared.

## Baseline

- Decision Tree Classifier

## Ensemble Models

- Random Forest Classifier
- Extra Trees Classifier
- Gradient Boosting Classifier
- Voting Classifier (Soft Voting)

The Voting Classifier combines

- Random Forest
- Extra Trees
- Gradient Boosting

---

# Evaluation Metrics

Each model is evaluated using

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

The model with the highest **F1 Score** is selected as the best model.

---

# Visualizations

The project automatically generates

- Accuracy Comparison
- Confusion Matrix for every model
- ROC Curve Comparison
- Feature Importance (Best Model)

All plots are stored inside

```
outputs/plots/
```

---

# Saved Model

The best-performing model is serialized using **Joblib**.

```
models/best_model.pkl
```

This model can later be loaded for prediction without retraining.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Joblib

---

# Installation

Clone the repository

```bash
git clone <repository-url>
```

Move into the project

```bash
cd Loan_approval_prediction
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Linux/macOS

```bash
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

Navigate to the source directory

```bash
cd src
```

Run

```bash
python main.py
```

---

# Output

After execution the project produces

- Performance comparison table
- Classification reports
- Confusion matrices
- ROC curve
- Feature importance plot
- Saved model (`best_model.pkl`)

---

# Future Improvements

Possible enhancements include:

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
- Cross-validation for more robust model evaluation.
- Handling class imbalance with resampling techniques if required.
- Feature selection to reduce dimensionality.
- Deployment as a REST API using Flask or FastAPI.
- Interactive dashboard for model predictions.

---

# Author

Ankit
