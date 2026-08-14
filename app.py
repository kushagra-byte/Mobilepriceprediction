import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Mobile Price Predictor 2026",
    page_icon="📱",
    layout="centered"
)

st.title("📱 Mobile Price Prediction (2026)")
st.markdown("Enter modern phone specifications and get the predicted price range.")

# -------------------------------------------------
# Load Model & Scaler
# -------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobile_price_model_2026.h5")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler_2026.pkl")

try:
    model = load_model()
    scaler = load_scaler()
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.info("Make sure these files are in the same folder:\n- mobile_price_model_2026.h5\n- scaler_2026.pkl")
    st.stop()

# -------------------------------------------------
# Price Mapping
# -------------------------------------------------
price_map = {
    0: {"range": "₹8,000 – ₹15,000",   "approx": "₹11,500", "label": "Low / Entry-level"},
    1: {"range": "₹15,000 – ₹25,000",  "approx": "₹20,000", "label": "Medium / Budget"},
    2: {"range": "₹25,000 – ₹40,000",  "approx": "₹32,500", "label": "High / Mid-range"},
    3: {"range": "₹40,000 – ₹70,000+", "approx": "₹55,000", "label": "Very High / Premium"}
}

# -------------------------------------------------
# Input Form
# -------------------------------------------------
with st.form("phone_form"):
    st.subheader("Enter Phone Specifications")

    col1, col2 = st.columns(2)

    with col1:
        ram_gb = st.number_input("RAM (GB)", min_value=2, max_value=24, value=8, step=1)
        storage_gb = st.number_input("Storage (GB)", min_value=32, max_value=1024, value=128, step=32)
        battery_mah = st.number_input("Battery (mAh)", min_value=2000, max_value=7000, value=5000, step=100)
        fast_charging_w = st.number_input("Fast Charging (Watts)", min_value=10, max_value=120, value=33, step=5)
        rear_camera_mp = st.number_input("Rear Camera (MP)", min_value=8, max_value=200, value=50, step=1)
        front_camera_mp = st.number_input("Front Camera (MP)", min_value=5, max_value=50, value=16, step=1)
        display_size_inch = st.number_input("Display Size (inches)", min_value=5.0, max_value=7.5, value=6.5, step=0.1)
        refresh_rate_hz = st.number_input("Refresh Rate (Hz)", min_value=60, max_value=144, value=120, step=30)
        weight_g = st.number_input("Weight (grams)", min_value=140, max_value=250, value=180, step=1)

    with col2:
        thickness_mm = st.number_input("Thickness (mm)", min_value=6.0, max_value=12.0, value=8.0, step=0.1)
        cpu_score = st.number_input("CPU Score (approx)", min_value=500, max_value=15000, value=5000, step=100)
        gpu_score = st.number_input("GPU Score (approx)", min_value=500, max_value=15000, value=4000, step=100)
        screen_to_body_ratio = st.number_input("Screen-to-Body Ratio (%)", min_value=70.0, max_value=95.0, value=85.0, step=0.5)
        launch_year = st.number_input("Launch Year", min_value=2018, max_value=2026, value=2024, step=1)

        st.markdown("#### Features")
        five_g = st.selectbox("5G Support", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        dual_sim = st.selectbox("Dual SIM", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        wireless_charging = st.selectbox("Wireless Charging", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        fingerprint_sensor = st.selectbox("Fingerprint Sensor", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        face_unlock = st.selectbox("Face Unlock", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

    submitted = st.form_submit_button("Predict Price", use_container_width=True)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if submitted:
    new_phone = {
        "ram_gb": ram_gb,
        "storage_gb": storage_gb,
        "battery_mah": battery_mah,
        "fast_charging_w": fast_charging_w,
        "rear_camera_mp": rear_camera_mp,
        "front_camera_mp": front_camera_mp,
        "display_size_inch": display_size_inch,
        "refresh_rate_hz": refresh_rate_hz,
        "weight_g": weight_g,
        "thickness_mm": thickness_mm,
        "cpu_score": cpu_score,
        "gpu_score": gpu_score,
        "screen_to_body_ratio": screen_to_body_ratio,
        "launch_year": launch_year,
        "5g_support": five_g,
        "dual_sim": dual_sim,
        "wireless_charging": wireless_charging,
        "fingerprint_sensor": fingerprint_sensor,
        "face_unlock": face_unlock
    }

    new_data = pd.DataFrame([new_phone])

    # Keep feature order same as training
    if hasattr(scaler, "feature_names_in_"):
        new_data = new_data.reindex(columns=scaler.feature_names_in_, fill_value=0)

    new_data_scaled = scaler.transform(new_data)

    prediction = model.predict(new_data_scaled, verbose=0)
    predicted_class = int(np.argmax(prediction))

    result = price_map[predicted_class]

    # -------------------------------------------------
    # Result Display (No Confidence)
    # -------------------------------------------------
    st.success("Prediction completed!")
    st.markdown("---")
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; text-align:center;">
                <h4 style="color:#555; margin-bottom:5px;">Price Range</h4>
                <h2 style="color:#1f77b4; margin:0;">{result['range']}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; text-align:center;">
                <h4 style="color:#555; margin-bottom:5px;">Approximate Price</h4>
                <h2 style="color:#2ca02c; margin:0;">{result['approx']}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("")

    st.markdown(
        f"""
        <div style="background-color:#e8f4fd; padding:18px; border-radius:10px; text-align:center; border:1px solid #b3d4f0;">
            <p style="margin:0; font-size:18px; color:#333333;">
                <b>Category:</b> 
                <span style="color:#d62728; font-size:20px; font-weight:bold;">{result['label']}</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
