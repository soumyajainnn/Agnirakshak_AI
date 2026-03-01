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
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

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
        pass

set_bg_image("assets/forest_peach.png")

# 3. MODEL LOAD & FIXED FEATURES
# Model expects these 10 in this exact order
EXPECTED_FEATURES = [
    'frp', 'daynight_N', 'solar_radiation_mean', 'mean_temp', 'dewpoint_mean', 
    'cloud_cover_mean', 'wind_direction_mean', 'fire_weather_index', 
    'temp_range', 'lat'
]

try:
    model = joblib.load('fire_risk_model.pkl')
    MODEL_LOADED = True
except Exception as e:
    st.error(f"Error loading model: {e}")
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

# --- Updated Real-time Weather Section (Fetch all details) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Fetch Real-Time Weather 🌤️")

# Session state use kar rahe hain taaki data gayab na ho
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = {"temp": avg_temp, "hum": None, "wind": None}

if st.sidebar.button("Get Current Weather"):
    if WEATHER_API_KEY:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        try:
            res = requests.get(url)
            if res.status_code == 200:
                data = res.json()
                st.session_state.weather_data['temp'] = data['main']['temp']
                st.session_state.weather_data['hum'] = data['main']['humidity']
                st.session_state.weather_data['wind'] = data['wind']['speed']
                
                # Yeh raha tera purana detailed success message
                st.sidebar.success(f"🌡 Temp: {st.session_state.weather_data['temp']}°C | 💧 Hum: {st.session_state.weather_data['hum']}% | 🌬 Wind: {st.session_state.weather_data['wind']} m/s")
            else:
                st.sidebar.error("City not found or API error!")
        except:
            st.sidebar.error("Connection failed!")
    else:
        st.sidebar.error("API Key missing in Secrets!")

# 6. PREDICTION LOGIC
if st.button("Predict Fire Risk") and MODEL_LOADED:
    # 'mean_temp' mein fetch kiya hua ya manual temperature jaayega
    model_input_data = {
        'frp': frp_val,
        'daynight_N': 1.0,
        'solar_radiation_mean': 350.0,
        'mean_temp': st.session_state.weather_data['temp'],
        'dewpoint_mean': 12.0,
        'cloud_cover_mean': 0.2,
        'wind_direction_mean': 180.0,
        'fire_weather_index': fwi,
        'temp_range': 10.0,
        'lat': lat
    }
    
    # Strictly filtering and ordering (No 'lon' sent to model)
    input_df = pd.DataFrame([model_input_data])[EXPECTED_FEATURES]

    try:
        prob = model.predict_proba(input_df)[:, 1][0]
        risk = "HIGH" if prob >= 0.5 else "LOW"
        
        st.subheader("🔥 Prediction Result")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Risk Likelihood", f"{prob:.1%}")
        with col2:
            if risk == "HIGH":
                st.error(f"Status: {risk} RISK")
                st.balloons()
            else:
                st.success(f"Status: {risk} RISK")
        
        st.progress(int(prob * 100))
        
        # Plotly Map
        map_df = pd.DataFrame({'lat': [lat], 'lon': [lon], 'Status': [risk]})
        fig = px.scatter_mapbox(map_df, lat="lat", lon="lon", color="Status", 
                                color_discrete_map={"HIGH": "red", "LOW": "green"},
                                zoom=5, mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Developed by Soumya | Agnirakshak AI")

