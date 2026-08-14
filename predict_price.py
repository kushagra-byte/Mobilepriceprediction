import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# -------------------------------------------------
# Load model and scaler
# -------------------------------------------------
model = tf.keras.models.load_model("mobile_price_model_2026.h5")
scaler = joblib.load("scaler_2026.pkl")

print("=" * 55)
print("     MOBILE PRICE PREDICTION SYSTEM (2026)")
print("=" * 55)
print("Enter the specifications of the mobile phone:\n")

# -------------------------------------------------
# Take inputs from user (new features)
# -------------------------------------------------
new_phone = {
    "ram_gb":              float(input("RAM (GB)                     : ")),
    "storage_gb":          float(input("Storage (GB)                 : ")),
    "battery_mah":         float(input("Battery (mAh)                : ")),
    "fast_charging_w":     float(input("Fast Charging (Watts)        : ")),
    "rear_camera_mp":      float(input("Rear Camera (MP)             : ")),
    "front_camera_mp":     float(input("Front Camera (MP)            : ")),
    "display_size_inch":   float(input("Display Size (inches)        : ")),
    "refresh_rate_hz":     float(input("Refresh Rate (Hz)            : ")),
    "weight_g":            float(input("Weight (grams)               : ")),
    "thickness_mm":        float(input("Thickness (mm)               : ")),
    "cpu_score":           float(input("CPU Score (approx)           : ")),
    "gpu_score":           float(input("GPU Score (approx)           : ")),
    "screen_to_body_ratio":float(input("Screen-to-Body Ratio (%)     : ")),
    "launch_year":         float(input("Launch Year (e.g. 2024)      : ")),
    "5g_support":          int(input("5G Support (1=Yes, 0=No)     : ")),
    "dual_sim":            int(input("Dual SIM (1=Yes, 0=No)       : ")),
    "wireless_charging":   int(input("Wireless Charging (1=Yes, 0=No): ")),
    "fingerprint_sensor":  int(input("Fingerprint Sensor (1=Yes, 0=No): ")),
    "face_unlock":         int(input("Face Unlock (1=Yes, 0=No)    : "))
}

# -------------------------------------------------
# Convert to DataFrame
# -------------------------------------------------
new_data = pd.DataFrame([new_phone])

# Important: Make sure columns are in the same order as training
# (The scaler expects the exact same feature order)
new_data = new_data[scaler.feature_names_in_] if hasattr(scaler, 'feature_names_in_') else new_data

# Scale the input
new_data_scaled = scaler.transform(new_data)

# -------------------------------------------------
# Predict
# -------------------------------------------------
prediction = model.predict(new_data_scaled, verbose=0)
predicted_class = int(np.argmax(prediction))
confidence = float(np.max(prediction) * 100)

# 2026 Price Map
price_map = {
    0: {"range": "₹8,000 – ₹15,000",   "approx": "₹11,500", "label": "Low / Entry-level"},
    1: {"range": "₹15,000 – ₹25,000",  "approx": "₹20,000", "label": "Medium / Budget"},
    2: {"range": "₹25,000 – ₹40,000",  "approx": "₹32,500", "label": "High / Mid-range"},
    3: {"range": "₹40,000 – ₹70,000+", "approx": "₹55,000", "label": "Very High / Premium"}
}

result = price_map[predicted_class]

# -------------------------------------------------
# Show Result
# -------------------------------------------------
print("\n" + "=" * 55)
print("              PREDICTION RESULT")
print("=" * 55)
print(f"Category              : {result['label']}")
print(f"Predicted Price Range : {result['range']}")
print(f"Approximate Price     : {result['approx']}")
print(f"Confidence            : {confidence:.1f}%")
print("=" * 55)
