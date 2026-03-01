# ------------------------------
# Agnirakshak AI - Forest Fire Risk Prediction App
# ------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import base64
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os

# ------------------------------
# 1. PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Agnirakshak AI 🔥 Forest Fire Risk", layout="wide", page_icon="🔥")

# ------------------------------
# 2. Load environment variables
# ------------------------------
load_dotenv()
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# ------------------------------
# 3. Set background image
# ------------------------------
def set_bg_image(image_file):
    try:
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
                background-position: center;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning(f"Background image not found at: {image_file}. Using default theme.")

set_bg_image("assets/forest_peach.png")

# ------------------------------
# 4. Load model safely (Fixed Column Order)
# ------------------------------
# Error message ke basis par columns ka exact sequence:
EXPECTED_FEATURES = [
    'frp', 'daynight_N', 'solar_radiation_mean', 'mean_temp', 'dewpoint_mean', 
    'cloud_cover_mean', 'wind_direction_mean', 'fire_weather_index', 
    'temp_range', 'lat'
]

try:
    # Model file ka naam confirm kar lena (fire_risk_model.pkl ya model.joblib)
    model = joblib.load('fire_risk_model.pkl') 
    MODEL_LOADED = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    MODEL_LOADED = False

# ------------------------------
# 5. HEADER
# ------------------------------
st.title("🔥 Agnirakshak AI - Forest Fire Risk Prediction Dashboard")
st.markdown("Predict forest fire risk in real-time using environmental parameters and weather data.")

# ------------------------------
# 6. SIDEBAR INPUTS
# ------------------------------
st.sidebar.header("Input Parameters")
lat = st.sidebar.number_input("Latitude", value=20.5, step=0.1, format="%.2f")
lon = st.sidebar.number_input("Longitude", value=77.5, step=0.1, format="%.2f")
temp_mean_input = st.sidebar.number_input("Average Temperature (°C)", value=25.0, step=0.1, format="%.2f")
fire_weather_index = st.sidebar.number_input("Fire Weather Index", value=3.5, step=0.1, format="%.2f")
frp = st.sidebar.number_input("Fire Radiative Power", value=10.0, step=0.1, format="%.2f")

# Real-time weather fetch
st.sidebar.markdown("---")
st.sidebar.subheader("Fetch Real-Time Weather 🌤️")
if 'current_temp' not in st.session_state:
    st.session_state.current_temp = temp_mean_input

if st.sidebar.button("Get Current Weather"):
    if WEATHER_API_KEY:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                st.session_state.current_temp = data['main']['temp']
                st.sidebar.success(f"🌡 Temp: {st.session_state.current_temp}°C | 💧 Humidity: {data['main']['humidity']}%")
            else:
                st.sidebar.error("Error fetching weather data.")
        except:
            st.sidebar.error("Network error!")
    else:
        st.sidebar.error("API Key missing!")

# ------------------------------
# 7. PREDICTION (Fixed Logic)
# ------------------------------
if st.button("Predict Fire Risk") and MODEL_LOADED:
    st.subheader("🔥 Prediction Result")

    # Mapping inputs to ALL 10 features expected by the model
    full_input_data_dict = {
        'frp': frp,
        'daynight_N': 1.0,           # Default
        'solar_radiation_mean': 350.0, # Default
        'mean_temp': st.session_state.current_temp,
        'dewpoint_mean': 12.0,       # Default
        'cloud_cover_mean': 0.2,     # Default
        'wind_direction_mean': 160.0, # Default
        'fire_weather_index': fire_weather_index,
        'temp_range': 8.0,           # Default
        'lat': lat
    }

    # DataFrame creation with strict column order
    input_df = pd.DataFrame([full_input_data_dict])
    input_df = input_df[EXPECTED_FEATURES]

    try:
        prediction_proba = model.predict_proba(input_df)[:, 1][0]
        prediction_class = "HIGH" if prediction_proba >= 0.5 else "LOW"

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Risk Likelihood", value=f"{prediction_proba:.2%}")
        with col2:
            if prediction_class == "HIGH":
                st.error(f"Status: {prediction_class} RISK")
                st.balloons()
            else:
                st.success(f"Status: {prediction_class} RISK")

        st.progress(int(prediction_proba * 100))

        # Map display
        map_df = pd.DataFrame({'lat': [lat], 'lon': [lon], 'Risk': [prediction_class]})
        fig = px.scatter_mapbox(map_df, lat="lat", lon="lon", color="Risk", size_max=15,
                                zoom=5, mapbox_style="open-street-map",
                                color_discrete_map={"HIGH":"red","LOW":"green"})
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ------------------------------
# 8. FOOTER
# ------------------------------
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>🚀 Developed by Soumya | Agnirakshak AI</p>", unsafe_allow_html=True)




