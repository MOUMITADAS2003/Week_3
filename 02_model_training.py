# ---------------------------------------------
# FILE: 02_model_training.py
# ---------------------------------------------
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load engineered features
X = np.load("X_balanced.npy")
y = np.load("y_balanced.npy")

# Train-val-test split (70-15-15)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# ----------------------- Random Forest -----------------------
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)

print("\n===== RANDOM FOREST RESULTS =====")
print(classification_report(y_val, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1]))

joblib.dump(rf, "model_random_forest.pkl")

# ----------------------- XGBoost + Grid Search -----------------------
xgb = XGBClassifier(eval_metric='logloss')

params = {
    'n_estimators': [50, 150],
    'max_depth': [4, 6],
    'learning_rate': [0.1, 0.01]
}

grid = GridSearchCV(xgb, params, scoring='roc_auc', cv=3, verbose=2)
grid.fit(X_train, y_train)

best_xgb = grid.best_estimator_
y_pred_xgb = best_xgb.predict(X_val)

print("\n===== XGBOOST BEST RESULTS =====")
print(classification_report(y_val, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_val, best_xgb.predict_proba(X_val)[:, 1]))

joblib.dump(best_xgb, "model_xgboost.pkl")

print("Training completed. Models saved!")
