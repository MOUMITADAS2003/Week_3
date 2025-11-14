# ---------------------------------------------
# FILE: 01_feature_engineering.py
# ---------------------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

# Load cleaned dataset
df = pd.read_csv("flights_cleaned.csv")

# --- 1. Derived Features ---
# Convert scheduled departure time to datetime
df['scheduled_departure_time'] = pd.to_datetime(df['scheduled_departure_time'])

# Hour of day
df['hour'] = df['scheduled_departure_time'].dt.hour

# Hour bucket
df['hour_bucket'] = pd.cut(
    df['hour'],
    bins=[0, 6, 12, 18, 24],
    labels=['LateNight', 'Morning', 'Afternoon', 'Evening'],
    include_lowest=True
)

# Weekday / weekend
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)

# --- 2. Airport route traffic avg ---
route_counts = df.groupby(['departure_airport', 'arrival_airport']).size().reset_index(name="route_traffic")
df = df.merge(route_counts, on=['departure_airport', 'arrival_airport'], how='left')

# --- 3. Select features ---
feature_columns = [
    'departure_airport', 'arrival_airport', 'hour_bucket', 'is_weekend',
    'route_traffic', 'temperature', 'wind_speed', 'visibility'
]

X = df[feature_columns]
y = df['delay_flag']   # Target: 0 = on time, 1 = delayed

# --- 4. Encode categorical variables ---
categorical_cols = ['departure_airport', 'arrival_airport', 'hour_bucket']
numeric_cols = ['is_weekend', 'route_traffic', 'temperature', 'wind_speed', 'visibility']

ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_encoded = ohe.fit_transform(X[categorical_cols])

X_numeric = df[numeric_cols].values
X_final = np.hstack([X_numeric, X_encoded])

# --- 5. Handle imbalance using SMOTE ---
sm = SMOTE(random_state=42)
X_balanced, y_balanced = sm.fit_resample(X_final, y)

# Save outputs
np.save("X_balanced.npy", X_balanced)
np.save("y_balanced.npy", y_balanced)
import joblib
joblib.dump(ohe, "ohe_encoder.pkl")

print("Feature engineering completed and saved!")
