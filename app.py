# ------------------------------
# Agnirakshak AI - Forest Fire Risk Prediction App
# ------------------------------
import streamlit as st
import pandas as pd
import joblib
import os
import base64
import requests
import plotly.express as px
from dotenv import load_dotenv

# 1. PAGE CONFIG
st.set_page_config(page_title="Agnirakshak AI 🔥", layout="wide", page_icon="🔥")
load_dotenv()
# Cloud deployment ke liye Secrets check
WEATHER_API_KEY = st.secrets.get("WEATHER_API_KEY") or os.getenv("WEATHER_API_KEY")

# 2. SET BACKGROUND IMAGE
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
    except:
        st.warning("Background image missing! Ensure 'assets/forest_peach.png' exists.")

set_bg_image("assets/forest_peach.png")

# 3. MODEL LOAD & FIXED FEATURES (Exact as per your error message)
EXPECTED_FEATURES = [
    'frp', 'daynight_N', 'solar_radiation_mean', 'mean_temp', 'dewpoint_mean', 
    'cloud_cover_mean', 'wind_direction_mean', 'fire_weather_index', 
    'temp_range', 'lat'
]

try:
    # Make sure this file name is exactly correct in your GitHub
    model = joblib.load('fire_risk_model.pkl')
    MODEL_LOADED = True
except Exception as e:
    st.error(f"Model Load Error: {e}")
    MODEL_LOADED = False

# 4. HEADER
st.title("🔥 Agnirakshak AI - Forest Fire Risk Prediction Dashboard")

# 5. SIDEBAR INPUTS
st.sidebar.header("Input Parameters")
lat = st.sidebar.number_input("Latitude", value=20.50, format="%.2f")
lon = st.sidebar.number_input("Longitude", value=77.50, format="%.2f")
avg_temp = st.sidebar.number_input("Average Temperature (°C)", value=25.00)
fwi = st.sidebar.number_input("Fire Weather Index", value=3.50)
frp_val = st.sidebar.number_input("Fire Radiative Power", value=10.00)

# --- Detailed Weather Fetch ---
st.sidebar.markdown("---")
st.sidebar.subheader("Fetch Real-Time Weather 🌤️")

if 'weather' not in st.session_state:
    st.session_state.weather = {"temp": avg_temp, "hum": "N/A", "wind": "N/A"}

if st.sidebar.button("Get Current Weather"):
    if WEATHER_API_KEY:
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
            res = requests.get(url).json()
            if res.get("main"):
                st.session_state.weather['temp'] = res['main']['temp']
                st.session_state.weather['hum'] = res['main']['humidity']
                st.session_state.weather['wind'] = res['wind']['speed']
                st.sidebar.success(f"🌡 {st.session_state.weather['temp']}°C | 💧 {st.session_state.weather['hum']}% | 🌬 {st.session_state.weather['wind']}m/s")
            else:
                st.sidebar.error("Invalid
