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
WEATHER_API_KEY = st.secrets.get("WEATHER_API_KEY") or os.getenv("WEATHER_API_KEY")

# 2. BACKGROUND IMAGE
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
        pass

set_bg_image("assets/forest_peach.png")

# 3. MODEL LOAD & FEATURES
EXPECTED_FEATURES = [
    'frp', 'daynight_N', 'solar_radiation_mean', 'mean_temp', 'dewpoint_mean', 
    'cloud_cover_mean', 'wind_direction_mean', 'fire_weather_index', 
    'temp_range', 'lat'
]

try:
    model = joblib.load('fire_risk_model.pkl')
    MODEL_LOADED = True
except Exception as e:
    st.error(f"Model Load Error: {e}")
    MODEL_LOADED = False

# 4. HEADER
st.title("🔥 Agnirakshak AI - Forest Fire Risk Prediction")

# 5. SIDEBAR
st.sidebar.header("Input Parameters")
lat = st.sidebar.number_input("Latitude", value=20.50, format="%.2f")
lon = st.sidebar.number_input("Longitude", value=77.50, format="%.2f")
avg_temp = st.sidebar.number_input("Average Temperature (°C)", value=25.00)
fwi = st.sidebar.number_input("Fire Weather Index", value=3.50)
frp_val = st.sidebar.number_input("Fire Radiative Power", value=10.00)

# WEATHER FETCH LOGIC
st.sidebar.markdown("---")
st.sidebar.subheader("Fetch Real-Time Weather 🌤️")

if 'weather' not in st.session_state:
    st.session_state.weather = {"temp": avg_temp, "hum": "N/A", "wind": "N/A"}

if st.sidebar.button("Get Current Weather"):
    if WEATHER_API_KEY:
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
            res = requests.get(url).json()
            if "main" in res:
                st.session_state.weather['temp'] = res['main']['temp']
                st.session_state.weather['hum'] = res['main']['humidity']
                st.session_state.weather['wind'] = res['wind']['speed']
                st.sidebar.success(f"Fetched: {st.session_state.weather['temp']}°C")
            else:
                st.sidebar.error("API Error: Check Key or Location")
        except:
            st.sidebar.error("Connection Error!")
    else:
        st.sidebar.error("API Key missing in Secrets!")

# 6. PREDICTION
if st.button("Predict Fire Risk") and MODEL_LOADED:
    # Model needs 'mean_temp', not 'lon'
    data = {
        'frp': frp_val,
        'daynight_N': 1.0,
        'solar_radiation_mean': 350.0,
        'mean_temp': st.session_state.weather['temp'],
        'dewpoint_mean': 12.0,
        'cloud_cover_mean': 0.2,
        'wind_direction_mean': 180.0,
        'fire_weather_index': fwi,
        'temp_range': 10.0,
        'lat': lat
    }
    
    input_df = pd.DataFrame([data])[EXPECTED_FEATURES]

    try:
        prob = model.predict_proba(input_df)[:, 1][0]
        risk = "HIGH" if prob >= 0.5 else "LOW"
        
        st.subheader("📊 Prediction Result")
        col1, col2 = st.columns(2)
        col1.metric("Likelihood", f"{prob:.1%}")
        if risk == "HIGH":
            col2.error(f"Status: {risk} RISK")
            st.balloons()
        else:
            col2.success(f"Status: {risk} RISK")
        
        st.progress(int(prob * 100))
        
        map_df = pd.DataFrame({'lat': [lat], 'lon': [lon], 'Risk': [risk]})
        fig = px.scatter_mapbox(map_df, lat="lat", lon="lon", color="Risk", 
                                color_discrete_map={"HIGH":"red", "LOW":"green"},
                                zoom=5, mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.markdown("---")
st.caption("Developed by Soumya | Agnirakshak AI")