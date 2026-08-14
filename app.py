import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(
    page_title="Mobile Price Predictor",
    page_icon="📱",
    layout="centered"
)

# Title
st.title("📱 Mobile Price Prediction")
st.markdown("Enter the specifications of a mobile phone and get the predicted price range (2026 standards).")


# Load model (cached so it loads only once)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobile_price_model.h5")


@st.cache_data
def load_and_fit_scaler():
    df = pd.read_csv(r"C:\Users\kusha\OneDrive\Desktop\New folder\mpp\dataset.csv")
    X = df.drop("price_range", axis=1)
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


try:
    model = load_model()
    scaler = load_and_fit_scaler()
except Exception as e:
    st.error(f"Error loading model or dataset: {e}")
    st.stop()

# Price mapping (2026 standards)
price_map = {
    0: {"range": "₹8,000 – ₹15,000", "approx": "₹11,500", "label": "Low / Entry-level"},
    1: {"range": "₹15,000 – ₹25,000", "approx": "₹20,000", "label": "Medium / Budget"},
    2: {"range": "₹25,000 – ₹40,000", "approx": "₹32,500", "label": "High / Mid-range"},
    3: {"range": "₹40,000 – ₹70,000+", "approx": "₹55,000", "label": "Very High / Premium"}
}

# Input Form
with st.form("phone_form"):
    st.subheader("Enter Phone Specifications")

    col1, col2 = st.columns(2)

    with col1:
        battery_power = st.number_input("Battery Power (mAh)", min_value=500, max_value=6000, value=1500, step=50)
        clock_speed = st.number_input("Clock Speed (GHz)", min_value=0.5, max_value=4.0, value=2.0, step=0.1)
        int_memory = st.number_input("Internal Memory (GB)", min_value=2, max_value=512, value=16, step=1)
        mobile_wt = st.number_input("Mobile Weight (grams)", min_value=80, max_value=300, value=130, step=1)
        n_cores = st.number_input("Number of Cores", min_value=1, max_value=16, value=4, step=1)
        ram = st.number_input("RAM (MB)", min_value=256, max_value=16384, value=2000, step=256)
        px_height = st.number_input("Pixel Height", min_value=0, max_value=3000, value=600, step=10)
        px_width = st.number_input("Pixel Width", min_value=0, max_value=3000, value=900, step=10)
        sc_h = st.number_input("Screen Height (cm)", min_value=5, max_value=25, value=14, step=1)
        sc_w = st.number_input("Screen Width (cm)", min_value=0, max_value=20, value=8, step=1)

    with col2:
        fc = st.number_input("Front Camera (MP)", min_value=0, max_value=50, value=5, step=1)
        pc = st.number_input("Primary Camera (MP)", min_value=0, max_value=200, value=13, step=1)
        m_dep = st.number_input("Mobile Depth (cm)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        talk_time = st.number_input("Talk Time (hours)", min_value=2, max_value=30, value=10, step=1)

        st.markdown("#### Connectivity & Features")
        blue = st.selectbox("Bluetooth", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        dual_sim = st.selectbox("Dual SIM", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        four_g = st.selectbox("4G Support", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        three_g = st.selectbox("3G Support", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        touch_screen = st.selectbox("Touch Screen", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        wifi = st.selectbox("WiFi", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

    submitted = st.form_submit_button("Predict Price", use_container_width=True)

# Result display
if submitted:
    new_phone = {
        "battery_power": battery_power,
        "blue": blue,
        "clock_speed": clock_speed,
        "dual_sim": dual_sim,
        "fc": fc,
        "four_g": four_g,
        "int_memory": int_memory,
        "m_dep": m_dep,
        "mobile_wt": mobile_wt,
        "n_cores": n_cores,
        "pc": pc,
        "px_height": px_height,
        "px_width": px_width,
        "ram": ram,
        "sc_h": sc_h,
        "sc_w": sc_w,
        "talk_time": talk_time,
        "three_g": three_g,
        "touch_screen": touch_screen,
        "wifi": wifi
    }

    new_data = pd.DataFrame([new_phone])
    new_data_scaled = scaler.transform(new_data)

    prediction = model.predict(new_data_scaled, verbose=0)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

    result = price_map[predicted_class]

    st.success("Prediction completed!")
    st.markdown("---")
    st.subheader("Prediction Result")

    # Better looking result cards (full text visible)
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

    st.markdown("")  # small space

    # Category + Confidence
    st.markdown(
        f"""
        <div style="background-color:#e8f4fd; padding:18px; border-radius:10px; text-align:center; border:1px solid #b3d4f0;">
            <p style="margin:0; font-size:16px; color:#333333;">
                <b>Category:</b> 
                <span style="color:#d62728; font-size:18px; font-weight:bold;">{result['label']}</span>
            </p>
            <p style="margin:8px 0 0 0; font-size:17px; color:#333333;">
                Confidence: <b style="color:#1f77b4;">{confidence:.1f}%</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(confidence / 100)