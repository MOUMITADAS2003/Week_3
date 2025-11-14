# ---------------------------------------------
# FILE: app.py
# RUN: streamlit run app.py
# ---------------------------------------------
import streamlit as st
import joblib
import numpy as np

# Load model & encoder
model = joblib.load("model_xgboost.pkl")
ohe = joblib.load("ohe_encoder.pkl")

st.title("✈️ Flight Delay Prediction Dashboard")
st.write("Enter flight details to predict if the flight will be delayed.")

departure = st.text_input("Departure Airport (e.g., JFK)")
arrival = st.text_input("Arrival Airport (e.g., LAX)")
hour_bucket = st.selectbox("Hour Bucket", ['LateNight', 'Morning', 'Afternoon', 'Evening'])
is_weekend = st.selectbox("Is Weekend?", [0, 1])
temperature = st.number_input("Temperature")
wind_speed = st.number_input("Wind Speed")
visibility = st.number_input("Visibility")
route_traffic = st.number_input("Route Traffic")

if st.button("Predict Delay"):
    categorical = [[departure, arrival, hour_bucket]]
    numeric = [is_weekend, route_traffic, temperature, wind_speed, visibility]

    encoded = ohe.transform(categorical)
    features = np.hstack([numeric, encoded.flatten()])

    prediction = model.predict([features])[0]

    if prediction == 1:
        st.error("🚨 Flight is likely to be DELAYED")
    else:
        st.success("✔️ Flight is likely to be ON TIME")
