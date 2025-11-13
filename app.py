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
        st.error(f"Background image not found at: {image_file}. Using default Streamlit theme.")

set_bg_image("assets/forest_peach.png")

# ------------------------------
# 4. Load model safely
# ------------------------------
EXPECTED_FEATURES = [
    'frp', 'daynight_N', 'solar_radiation_mean', 'lon', 'dewpoint_mean', 
    'cloud_cover_mean', 'wind_direction_mean', 'fire_weather_index', 
    'temp_range', 'lat'
]

try:
    # Change 'model.joblib' to the actual name of your model file:
    model = joblib.load('fire_risk_model.pkl') 
    MODEL_LOADED = True
except FileNotFoundError:
    st.error("Error: model.joblib file not found. Please ensure the retrained model is in the project directory.")
    MODEL_LOADED = False
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
temp_mean = st.sidebar.number_input("Average Temperature (°C)", value=25.0, step=0.1, format="%.2f")
fire_weather_index = st.sidebar.number_input("Fire Weather Index", value=3.5, step=0.1, format="%.2f")
frp = st.sidebar.number_input("Fire Radiative Power", value=10.0, step=0.1, format="%.2f")

# ------------------------------
# Real-time weather fetch
# ------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Fetch Real-Time Weather 🌤️")
if 'current_temp' not in st.session_state:
    st.session_state.current_temp = temp_mean

if st.sidebar.button("Get Current Weather"):
    if WEATHER_API_KEY:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                fetched_temp = data['main']['temp']
                humidity = data['main']['humidity']
                wind_speed = data['wind']['speed']
                st.session_state.current_temp = fetched_temp
                st.sidebar.success(f"🌡 Temp: {fetched_temp}°C | 💧 Humidity: {humidity}% | 🌬 Wind: {wind_speed} m/s")
            else:
                st.sidebar.error(f"Error fetching weather data. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"Network error while fetching weather: {e}")
    else:
        st.sidebar.error("⚠️ WEATHER_API_KEY not found. Please add it in .env file!")

# ------------------------------
# 7. PREDICTION
# ------------------------------
if st.button("Predict Fire Risk") and MODEL_LOADED:
    st.subheader("🔥 Prediction Result")

    # Default values for missing features
    missing_features_defaults = {
        'daynight_N': 1.0,
        'solar_radiation_mean': 300.0,
        'dewpoint_mean': 10.0,
        'cloud_cover_mean': 0.0,
        'wind_direction_mean': 0.0,
        'temp_range': 5.0,
    }

    # Combine defaults with user inputs
    full_input_data_dict = {
        **missing_features_defaults,
        'lat': lat,
        'lon': lon,
        'fire_weather_index': fire_weather_index,
        'frp': frp,
    }

    # Create DataFrame and enforce correct feature order
    input_data = pd.DataFrame([full_input_data_dict])
    input_data = input_data.reindex(columns=EXPECTED_FEATURES)

    try:
        prediction_proba = model.predict_proba(input_data)[:, 1][0]
        prediction_class = "HIGH" if prediction_proba >= 0.5 else "LOW"

        st.metric(label="Predicted Fire Likelihood", value=f"{prediction_proba:.2f}")
        if prediction_class == "HIGH":
            st.error(f"Prediction: {prediction_class} RISK (Likelihood is {prediction_proba:.1%})")
            st.balloons()
        else:
            st.success(f"Prediction: {prediction_class} RISK (Likelihood is {prediction_proba:.1%})")

        # Fire Risk Meter
        st.subheader("📊 Risk Probability Meter")
        st.progress(int(prediction_proba * 100))

        # Map display
        st.subheader("📍 Location Map")
        map_df = pd.DataFrame({'lat': [lat], 'lon': [lon], 'Risk': [prediction_class]})
        fig = px.scatter_mapbox(map_df, lat="lat", lon="lon", color="Risk", size_max=15,
                                zoom=5, mapbox_style="open-street-map",
                                color_discrete_map={"HIGH":"red","LOW":"green"})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.caption(f"Model used {len(EXPECTED_FEATURES)} features for prediction.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# ------------------------------
# 8. Optional EDA
# ------------------------------
if st.checkbox("Show Sample EDA"):
    st.subheader("🔥 Sample EDA Visualizations (Mock Data)")
    df = pd.DataFrame({
        'lat': [20.5, 21.0, 19.5, 22.0, 20.8, 19.9],
        'lon': [77.5, 78.0, 76.5, 77.0, 77.3, 76.8],
        'temp_mean': [25, 30, 28, 26, 32, 24],
        'fire_weather_index': [3.5, 4.2, 3.8, 2.5, 5.0, 2.0],
        'frp': [10, 15, 12, 8, 20, 5]
    })
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
    st.pyplot(fig)

    fig_scatter = px.scatter(df, x='temp_mean', y='fire_weather_index', trendline="ols",
                             title="Temperature vs. Fire Weather Index")
    st.plotly_chart(fig_scatter, use_container_width=True)

# ------------------------------
# 9. FOOTER
# ------------------------------
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>🚀 Developed by Soumya | Agnirakshak AI</p>", unsafe_allow_html=True)




