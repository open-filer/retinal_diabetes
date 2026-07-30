"""
Multi-Model Diabetes Prediction System
Training script: loads data, preprocesses, trains 5 models + a Voting ensemble,
evaluates each, and saves everything needed by the Streamlit app.
"""

import pandas as pd
import numpy as np
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("diabetes.csv")

# ---------------------------------------------------------------------------
# 2. Clean data
# These columns physically cannot be 0 in a living person, so 0 = missing data.
# Replace with the median of non-zero values for that column.
# ---------------------------------------------------------------------------
cols_with_invalid_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_with_invalid_zero:
    median_val = df.loc[df[col] != 0, col].median()
    df[col] = df[col].replace(0, median_val)

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET = "Outcome"

X = df[FEATURES]
y = df[TARGET]

# ---------------------------------------------------------------------------
# 3. Train/test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# 4. Scale features (needed for Logistic Regression, KNN, SVM)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 5. Define models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=9),
    "Support Vector Machine": SVC(probability=True, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.08,
        random_state=RANDOM_STATE, eval_metric="logloss"
    ),
}

results = {}
trained_models = {}

print("=" * 70)
print("TRAINING INDIVIDUAL MODELS")
print("=" * 70)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds).tolist()

    results[name] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
    }
    trained_models[name] = model

    print(f"{name:25s} | Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")

# ---------------------------------------------------------------------------
# 6. Build the ensemble (Voting Classifier) -- combines all 5 models
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TRAINING ENSEMBLE (VOTING CLASSIFIER)")
print("=" * 70)

voting_clf = VotingClassifier(
    estimators=[(name, model) for name, model in models.items()],
    voting="soft",  # averages predicted probabilities -> smoother decisions
)
voting_clf.fit(X_train_scaled, y_train)
ensemble_preds = voting_clf.predict(X_test_scaled)

ensemble_acc = accuracy_score(y_test, ensemble_preds)
ensemble_prec = precision_score(y_test, ensemble_preds)
ensemble_rec = recall_score(y_test, ensemble_preds)
ensemble_f1 = f1_score(y_test, ensemble_preds)
ensemble_cm = confusion_matrix(y_test, ensemble_preds).tolist()

results["Ensemble (Voting)"] = {
    "accuracy": round(ensemble_acc, 4),
    "precision": round(ensemble_prec, 4),
    "recall": round(ensemble_rec, 4),
    "f1_score": round(ensemble_f1, 4),
    "confusion_matrix": ensemble_cm,
}
trained_models["Ensemble (Voting)"] = voting_clf

print(f"{'Ensemble (Voting)':25s} | Acc: {ensemble_acc:.4f}  Prec: {ensemble_prec:.4f}  "
      f"Rec: {ensemble_rec:.4f}  F1: {ensemble_f1:.4f}")

# ---------------------------------------------------------------------------
# 7. Save everything the Streamlit app needs
# ---------------------------------------------------------------------------
joblib.dump(trained_models, "trained_models.joblib")
joblib.dump(scaler, "scaler.joblib")

with open("model_results.json", "w") as f:
    json.dump(results, f, indent=2)

with open("feature_columns.json", "w") as f:
    json.dump(FEATURES, f)

print("\nSaved: trained_models.joblib, scaler.joblib, model_results.json")
print("Training complete.")
