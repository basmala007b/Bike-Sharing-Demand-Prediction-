import time
import base64
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="Bike Sharing Demand Predictor", page_icon="🎡", layout="wide")

# -----------------------------------------------------
# LOAD LOADING GIF AS BASE64
# -----------------------------------------------------
@st.cache_data
def get_base64_file(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

loading_gif_base64 = get_base64_file("assets/loading2.gif")

# -----------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Solid navy background with a single calm road line at the bottom */
.stApp {
    background: #10263b;
}
.stApp::before {
    content: "";
    position: fixed;
    left: 0; right: 0; bottom: 0;
    height: 120px;
    z-index: 0;
    pointer-events: none;
    background: linear-gradient(to top, rgba(255,255,255,0.05), transparent);
}
.stApp::after {
    content: "";
    position: fixed;
    left: 0; right: 0; bottom: 60px;
    height: 2px;
    z-index: 0;
    pointer-events: none;
    background-image: repeating-linear-gradient(
        to right,
        rgba(255, 255, 255, 0.22) 0px,
        rgba(255, 255, 255, 0.22) 26px,
        transparent 26px,
        transparent 52px
    );
}
.block-container { position: relative; z-index: 1; }

.block-container {
    padding-top: 3rem;
    max-width: 1000px;
}

/* Hero section: icon + title centered */
.hero-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    margin-top: 1rem;
    margin-bottom: 1.5rem;
}
.hero-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 22px;
}

/* Wheel emoji icon */
.wheel-icon {
    font-size: 4.6rem;
    line-height: 1;
    display: inline-block;
    transition: all 0.35s ease;
    filter: drop-shadow(0 0 0px rgba(255, 200, 87, 0));
}
.wheel-icon:hover {
    transform: rotate(35deg) scale(1.1);
    filter: drop-shadow(0 0 20px rgba(255, 200, 87, 0.9));
}

/* Big centered title - bright golden glow, always visible */
.hero-title {
    font-size: 3.4rem;
    font-weight: 800;
    color: #ffc857;
    letter-spacing: -1px;
    margin: 0;
    line-height: 1.1;
    text-shadow: 0 0 18px rgba(255, 200, 87, 0.55), 0 0 40px rgba(255, 200, 87, 0.25);
    transition: all 0.35s ease;
}
.hero-title:hover {
    text-shadow: 0 0 28px rgba(255, 200, 87, 0.9), 0 0 55px rgba(255, 200, 87, 0.5);
    transform: scale(1.015);
}

/* Subtitle / description - bright and readable over the pattern */
.hero-subtitle {
    max-width: 680px;
    margin: 1.3rem auto 0 auto;
    color: #ffffff;
    font-size: 1.1rem;
    font-weight: 600;
    line-height: 1.7;
    letter-spacing: 0.2px;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.55), 0 0 20px rgba(0, 0, 0, 0.35);
}

/* Center the predict button under the hero */
.predict-btn-wrap {
    display: flex;
    justify-content: center;
    margin-top: 1.8rem;
}

/* Sidebar - gray background, light text */
section[data-testid="stSidebar"] {
    background: #2b2b2b;
}
section[data-testid="stSidebar"] * {
    color: #f3e8ee !important;
}
section[data-testid="stSidebar"] h2 {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #f3e8ee !important;
}
section[data-testid="stSidebar"] label {
    font-size: 1.02rem !important;
    font-weight: 500 !important;
}

/* Expander */
div[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 14px;
    border: 1px solid rgba(243, 232, 238, 0.12);
}
div[data-testid="stExpander"] * {
    color: #f3e8ee !important;
}

/* Predict button - light colored */
.stButton > button,
.stButton > button[kind="primary"],
button[kind="primary"],
div[data-testid="stButton"] button {
    background: #f3e8ee !important;
    color: #000000 !important;
    font-weight: 700 !important;
    font-size: 1.15rem !important;
    border-radius: 14px !important;
    padding: 0.8rem 2.6rem !important;
    border: none !important;
    box-shadow: 0 4px 20px rgba(243, 232, 238, 0.25) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    white-space: normal !important;
    height: auto !important;
}
.stButton > button:hover,
.stButton > button[kind="primary"]:hover,
button[kind="primary"]:hover,
div[data-testid="stButton"] button:hover {
    transform: scale(1.04);
    box-shadow: 0 6px 28px rgba(243, 232, 238, 0.45) !important;
    color: #000000 !important;
    background: #f3e8ee !important;
}
/* Turn red while actively being pressed/clicked */
.stButton > button:active,
.stButton > button[kind="primary"]:active,
button[kind="primary"]:active,
div[data-testid="stButton"] button:active {
    background: #ff4b4b !important;
    color: #ffffff !important;
    transform: scale(0.98);
    box-shadow: 0 4px 16px rgba(255, 75, 75, 0.5) !important;
}
.stButton > button:active *,
button[kind="primary"]:active * {
    color: #ffffff !important;
}
/* Force the button's inner label to wrap instead of truncating with an ellipsis */
.stButton > button *,
button[kind="primary"] * {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
    color: #000000 !important;
}

/* Success box (prediction result) - dark glass with golden neon border */
div[data-testid="stAlertContainer"] {
    background: rgba(10, 20, 35, 0.55);
    backdrop-filter: blur(6px);
    border: 1.5px solid rgba(255, 200, 87, 0.85);
    border-radius: 16px;
    padding: 1.3rem;
    box-shadow: 0 0 18px rgba(255, 200, 87, 0.45), inset 0 0 20px rgba(255, 200, 87, 0.06);
}
div[data-testid="stAlertContainer"] p,
div[data-testid="stAlertContainer"] * {
    color: #f3e8ee !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    text-shadow: 0 0 10px rgba(243, 232, 238, 0.7);
}

/* Custom prediction result box (replaces st.success) */
.predict-result-box {
    background: rgba(10, 20, 35, 0.55);
    backdrop-filter: blur(6px);
    border: 1.5px solid rgba(255, 200, 87, 0.85);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    box-shadow: 0 0 18px rgba(255, 200, 87, 0.45), inset 0 0 20px rgba(255, 200, 87, 0.06);
    margin-bottom: 1rem;
}
.predict-result-box, .predict-result-box * {
    color: #ffffff !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
}
.predict-result-box .predict-number {
    color: #ffdd8a !important;
    font-size: 2.3rem !important;
    font-weight: 800 !important;
    text-shadow: 0 0 12px rgba(255, 200, 87, 0.8);
}

/* Metric cards - dark glass with golden neon border */
div[data-testid="stMetric"] {
    background: rgba(10, 20, 35, 0.55);
    backdrop-filter: blur(6px);
    border-radius: 14px;
    padding: 1rem;
    border: 1.5px solid rgba(255, 200, 87, 0.7);
    box-shadow: 0 0 16px rgba(255, 200, 87, 0.35), inset 0 0 16px rgba(255, 200, 87, 0.05);
}
div[data-testid="stMetricLabel"] p {
    color: #ffdd8a !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
}
div[data-testid="stMetricValue"] {
    color: #ffdd8a !important;
    font-weight: 800 !important;
    font-size: 1.7rem !important;
    text-shadow: 0 0 10px rgba(255, 200, 87, 0.7);
}

/* Divider */
hr {
    border-color: rgba(243, 232, 238, 0.15) !important;
}

/* Loading animation container */
.loading-wrap {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 1.8rem 0;
}
.loading-wrap img {
    width: 220px;
    height: auto;
    image-rendering: pixelated;
    filter: drop-shadow(0 0 25px rgba(255, 200, 87, 0.5));
    background: rgba(255, 255, 255, 0.06);
    border-radius: 18px;
    padding: 14px 22px;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# -----------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("bike_model.pkl")
    columns = joblib.load("feature_columns.pkl")
    return model, columns

model, feature_columns = load_model()

# -----------------------------------------------------
# WHEEL EMOJI ICON - glows on hover via CSS above
# -----------------------------------------------------
WHEEL_ICON = ''

# -----------------------------------------------------
# HERO SECTION
# -----------------------------------------------------
st.markdown(f"""
<div class="hero-wrap">
<div class="hero-row">
{WHEEL_ICON}
<h1 class="hero-title">Bike Sharing Demand Predictor</h1>
</div>
<p class="hero-subtitle">
Predicts the number of bikes that will be rented in a given hour,
based on weather and time conditions, using a Random Forest model
trained on Capital Bikeshare data.
</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------
# SIDEBAR INPUTS
# -----------------------------------------------------
st.sidebar.header("⚙️ Prediction Settings")

season = st.sidebar.selectbox(
    "🗓️ Season", options=[1, 2, 3, 4],
    format_func=lambda x: {1: "❄️ Winter", 2: "🌸 Spring", 3: "☀️ Summer", 4: "🍂 Fall"}[x]
)
yr = st.sidebar.selectbox("📅 Year", options=[0, 1], format_func=lambda x: "2011" if x == 0 else "2012")
mnth = st.sidebar.slider("📆 Month", 1, 12, 6)
hr = st.sidebar.slider("🕐 Hour", 0, 23, 12)
holiday = st.sidebar.selectbox("🎉 Holiday?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
weekday = st.sidebar.slider("📋 Day of week (0=Sun ... 6=Sat)", 0, 6, 3)
workingday = st.sidebar.selectbox("💼 Working day?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
weathersit = st.sidebar.selectbox(
    "🌤️ Weather condition", options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "☀️ Clear / Few clouds",
        2: "☁️ Mist / Cloudy",
        3: "🌧️ Light rain / snow",
        4: "⛈️ Heavy rain / snow"
    }[x]
)
temp = st.sidebar.slider("🌡️ Temperature (normalized 0-1)", 0.0, 1.0, 0.5)
atemp = st.sidebar.slider("🤗 Feels-like temperature (normalized 0-1)", 0.0, 1.0, 0.5)
hum = st.sidebar.slider("💧 Humidity (normalized 0-1)", 0.0, 1.0, 0.5)
windspeed = st.sidebar.slider("💨 Wind speed (normalized 0-1)", 0.0, 1.0, 0.2)
day_of_month = st.sidebar.slider("🗓️ Day of month", 1, 31, 15)
is_weekend = 1 if weekday in [0, 6] else 0

# -----------------------------------------------------
# BUILD INPUT DATAFRAME (must match training column order)
# -----------------------------------------------------
input_data = pd.DataFrame([{
    "season": season,
    "yr": yr,
    "mnth": mnth,
    "hr": hr,
    "holiday": holiday,
    "weekday": weekday,
    "workingday": workingday,
    "weathersit": weathersit,
    "temp": temp,
    "atemp": atemp,
    "hum": hum,
    "windspeed": windspeed,
    "day_of_month": day_of_month,
    "is_weekend": is_weekend,
}])[feature_columns]

# -----------------------------------------------------
# PREDICT BUTTON - centered
# -----------------------------------------------------
st.markdown('<div class="predict-btn-wrap">', unsafe_allow_html=True)
_, mid_col, _ = st.columns([1, 1, 1])
with mid_col:
    predict_clicked = st.button("Predict Demand 🚀", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------
# PREDICTION + LOADING ANIMATION
# -----------------------------------------------------
if predict_clicked:
    loading_placeholder = st.empty()
    loading_placeholder.markdown(
        f'<div class="loading-wrap"><img src="data:image/gif;base64,{loading_gif_base64}"></div>',
        unsafe_allow_html=True
    )
    time.sleep(1.2)  # let the animation play briefly
    prediction = model.predict(input_data)[0]
    loading_placeholder.empty()

    st.markdown(
        f'<div class="predict-result-box">Predicted bike rentals: '
        f'<span class="predict-number">{int(prediction)}</span> bikes</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("🕐 Hour", f"{hr}:00")
    col2.metric("🗓️ Season", {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}[season])
    col3.metric("🌤️ Weather", {1: "Clear", 2: "Cloudy", 3: "Light rain", 4: "Heavy rain"}[weathersit])

st.divider()
with st.expander("📊 About the model"):
    st.write(
        """
        - **Model:** Random Forest Regressor
        - **R² on test data:** ~0.94
        - **Data:** Capital Bikeshare (Washington D.C.), 2 years (2011-2012), 17,379 rows
        """
    )
