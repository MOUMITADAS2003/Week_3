# ---------------------------------------------
# FILE: 03_model_testing.py
# ---------------------------------------------
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

X = np.load("X_balanced.npy")
y = np.load("y_balanced.npy")

# Load final model
model = joblib.load("model_xgboost.pkl")

# Train-test split
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Predictions
y_pred = model.predict(X_test)

print("=== FINAL MODEL TEST RESULTS ===")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
