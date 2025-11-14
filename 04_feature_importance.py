# ---------------------------------------------
# FILE: 04_feature_importance.py
# ---------------------------------------------
import numpy as np
import joblib
import matplotlib.pyplot as plt

model = joblib.load("model_xgboost.pkl")
ohe = joblib.load("ohe_encoder.pkl")

categorical = ohe.get_feature_names_out()
numeric = ['is_weekend', 'route_traffic', 'temperature', 'wind_speed', 'visibility']
feature_names = numeric + list(categorical)

importances = model.feature_importances_
sorted_idx = np.argsort(importances)

plt.figure(figsize=(10, 12))
plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
plt.title("Feature Importance (XGBoost)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
