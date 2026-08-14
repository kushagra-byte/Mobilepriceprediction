import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load model
model = tf.keras.models.load_model("mobile_price_model.h5")

print("=" * 50)
print("      MOBILE PRICE PREDICTION SYSTEM")
print("=" * 50)
print("Enter the specifications of the mobile phone:\n")

# Take inputs from user
new_phone = {
    "battery_power": float(input("Battery Power (mAh)          : ")),
    "blue":          int(input("Bluetooth (1 = Yes, 0 = No)  : ")),
    "clock_speed":   float(input("Clock Speed (GHz)            : ")),
    "dual_sim":      int(input("Dual SIM (1 = Yes, 0 = No)   : ")),
    "fc":            float(input("Front Camera (MP)            : ")),
    "four_g":        int(input("4G Support (1 = Yes, 0 = No) : ")),
    "int_memory":    float(input("Internal Memory (GB)         : ")),
    "m_dep":         float(input("Mobile Depth (cm)            : ")),
    "mobile_wt":     float(input("Mobile Weight (grams)        : ")),
    "n_cores":       float(input("Number of Cores              : ")),
    "pc":            float(input("Primary Camera (MP)          : ")),
    "px_height":     float(input("Pixel Height                 : ")),
    "px_width":      float(input("Pixel Width                  : ")),
    "ram":           float(input("RAM (MB)                     : ")),
    "sc_h":          float(input("Screen Height (cm)           : ")),
    "sc_w":          float(input("Screen Width (cm)            : ")),
    "talk_time":     float(input("Talk Time (hours)            : ")),
    "three_g":       int(input("3G Support (1 = Yes, 0 = No) : ")),
    "touch_screen":  int(input("Touch Screen (1 = Yes, 0 = No): ")),
    "wifi":          int(input("WiFi (1 = Yes, 0 = No)       : "))
}

# Convert to DataFrame
new_data = pd.DataFrame([new_phone])

# Refit scaler using training data
scaler = StandardScaler()
df = pd.read_csv(r"C:\Users\kusha\OneDrive\Desktop\New folder\mpp\dataset.csv")
X_train = df.drop("price_range", axis=1)
scaler.fit(X_train)

# Scale new input
new_data_scaled = scaler.transform(new_data)

# Predict
prediction = model.predict(new_data_scaled, verbose=0)
predicted_class = np.argmax(prediction)
confidence = np.max(prediction) * 100

# 2026 updated price map
price_map = {
    0: {"range": "₹8,000 – ₹15,000",   "approx": "₹11,500"},
    1: {"range": "₹15,000 – ₹25,000",  "approx": "₹20,000"},
    2: {"range": "₹25,000 – ₹40,000",  "approx": "₹32,500"},
    3: {"range": "₹40,000 – ₹70,000+", "approx": "₹55,000"}
}

result = price_map[predicted_class]

print("\n" + "=" * 50)
print("              PREDICTION RESULT")
print("=" * 50)
print(f"Predicted Price Range : {result['range']}")
print(f"Approximate Price     : {result['approx']}")
print(f"Confidence            : {confidence:.1f}%")
print("=" * 50)
