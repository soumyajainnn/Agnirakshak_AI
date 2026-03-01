# ------------------------------
# Agnirakshak AI - Forest Fire Risk Prediction App
# ------------------------------

import streamlit as st
import pandas as pd
import joblib
import os
from dotenv import load_dotenv

# Page Config
st.set_page_config(page_title="Agnirakshak AI 🔥", layout="wide")
load_dotenv()
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# 1. Model Column Order (As per your error message)
# Model expects: frp, daynight_N, solar_radiation_mean, mean_temp, dewpoint_mean, etc.
EXPECTED_FEATURES = [
    'frp', 'daynight_N', 'solar_radiation_mean', 'mean_temp', 'dewpoint_mean', 
    'cloud_cover_mean', 'wind_direction_mean', 'fire_weather_index', 
    'temp_range', 'lat'
]

# 2. Load Model
try:
    model = joblib.load('fire_risk_model.pkl')
    MODEL_LOADED = True
except Exception as e:
    st.error(f"Model Load Error: {e}")
    MODEL_LOADED = False

st.title("🔥 Agnirakshak AI - Forest Fire Risk Prediction")

# 3. Sidebar Inputs
st.sidebar.header("Input Parameters")
lat = st.sidebar.number_input("Latitude", value=20.5)
# Note: lon is used for map/weather but NOT sent to the model prediction
lon = st.sidebar.number_input("Longitude", value=77.5) 
temp_mean = st.sidebar.number_input("Average Temperature (°C)", value=25.0)
fwi = st.sidebar.number_input("Fire Weather Index", value=3.5)
frp_input = st.sidebar.number_input("Fire Radiative Power", value=10.0)

# 4. Prediction Logic
if st.button("Predict Fire Risk") and MODEL_LOADED:
    # Model ko exact 10 features chahiye wahi order mein
    data_dict = {
        'frp': frp_input,
        'daynight_N': 1.0,           # Default
        'solar_radiation_mean': 350.0, # Default
        'mean_temp': temp_mean,      # User Input
        'dewpoint_mean': 12.0,       # Default
        'cloud_cover_mean': 0.2,     # Default
        'wind_direction_mean': 180.0, # Default
        'fire_weather_index': fwi,   # User Input
        'temp_range': 10.0,          # Default
        'lat': lat                   # User Input
    }
    
    input_df = pd.DataFrame([data_dict])
    input_df = input_df[EXPECTED_FEATURES] # Strict Column Ordering

    try:
        # Prediction
        prob = model.predict_proba(input_df)[:, 1][0]
        risk = "HIGH" if prob >= 0.5 else "LOW"
        
        st.subheader("🔥 Prediction Result")
        if risk == "HIGH":
            st.error(f"Risk: {risk} (Likelihood: {prob:.2%})")
        else:
            st.success(f"Risk: {risk} (Likelihood: {prob:.2%})")
            
        st.progress(int(prob * 100))
    except Exception as e:
        st.error(f"Prediction failed: {e}")



