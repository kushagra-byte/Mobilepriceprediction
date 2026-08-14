import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

# Load model
model = tf.keras.models.load_model("mobile_price_model.h5")

# Define your new mobile phone specs (in the same order as training features)
new_phone = {
    "battery_power": 1500,
    "blue": 1,
    "clock_speed": 2.0,
    "dual_sim": 1,
    "fc": 5,
    "four_g": 1,
    "int_memory": 16,
    "m_dep": 0.5,
    "mobile_wt": 130,
    "n_cores": 4,
    "pc": 13,
    "px_height": 600,
    "px_width": 900,
    "ram": 2000,
    "sc_h": 14,
    "sc_w": 8,
    "talk_time": 10,
    "three_g": 1,
    "touch_screen": 1,
    "wifi": 1
}

# Convert to DataFrame
new_data = pd.DataFrame([new_phone])

# Use same scaler as training
scaler = StandardScaler()
df = pd.read_csv("dataset.csv")  # Load training data to refit scaler
X_train = df.drop("price_range", axis=1)
scaler.fit(X_train)

# Scale new input
new_data_scaled = scaler.transform(new_data)

# Predict
prediction = model.predict(new_data_scaled)
predicted_class = np.argmax(prediction)

# Map label to price range
price_map = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}
print(f"Predicted Price Range: {price_map[predicted_class]}")
