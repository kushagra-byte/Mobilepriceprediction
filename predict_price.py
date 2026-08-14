import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load model
model = tf.keras.models.load_model("mobile_price_model.h5")

# Define your new mobile phone specs (same order as training features)
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

# Refit scaler using training data
scaler = StandardScaler()
df = pd.read_csv(r"C:\Users\kusha\OneDrive\Desktop\New folder\mpp\dataset.csv")
X_train = df.drop("price_range", axis=1)
scaler.fit(X_train)

# Scale new input
new_data_scaled = scaler.transform(new_data)

# Predict
prediction = model.predict(new_data_scaled)
predicted_class = np.argmax(prediction)
confidence = np.max(prediction) * 100

# Map class → actual price range (you can adjust these values)
price_map = {
    0: {"range": "₹5,000 – ₹10,000",  "approx": "₹7,500"},
    1: {"range": "₹10,000 – ₹20,000", "approx": "₹15,000"},
    2: {"range": "₹20,000 – ₹35,000", "approx": "₹27,500"},
    3: {"range": "₹35,000 – ₹60,000+", "approx": "₹45,000"}
}

result = price_map[predicted_class]

print(f"Predicted Price Range : {result['range']}")
print(f"Approximate Price     : {result['approx']}")
print(f"Confidence            : {confidence:.1f}%")
