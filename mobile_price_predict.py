import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -------------------------------------------------
# 1. Load the new dataset
# -------------------------------------------------
df = pd.read_csv(r"C:\Users\kusha\OneDrive\Desktop\New folder\mpp\smartphone_dataset_1M.csv")  # ← change path

print("Original shape:", df.shape)

# -------------------------------------------------
# 2. Create price_range labels (2026 standards)
# -------------------------------------------------
def create_price_range(price):
    if price < 15000:
        return 0      # Low / Entry-level
    elif price < 25000:
        return 1      # Medium / Budget
    elif price < 40000:
        return 2      # High / Mid-range
    else:
        return 3      # Very High / Premium

df["price_range"] = df["price_inr"].apply(create_price_range)

print("\nPrice Range Distribution:")
print(df["price_range"].value_counts().sort_index())

# -------------------------------------------------
# 3. Select useful features
# -------------------------------------------------
numeric_features = [
    "ram_gb", "storage_gb", "battery_mah", "fast_charging_w",
    "rear_camera_mp", "front_camera_mp", "display_size_inch",
    "refresh_rate_hz", "weight_g", "thickness_mm",
    "cpu_score", "gpu_score", "screen_to_body_ratio",
    "launch_year", "5g_support", "dual_sim",
    "wireless_charging", "fingerprint_sensor", "face_unlock"
]

# Keep only columns that exist
numeric_features = [col for col in numeric_features if col in df.columns]

categorical_features = []
if "brand" in df.columns:
    categorical_features.append("brand")
if "os" in df.columns:
    categorical_features.append("os")
if "display_type" in df.columns:
    categorical_features.append("display_type")

print("\nUsing numeric features:", numeric_features)
print("Using categorical features:", categorical_features)

# -------------------------------------------------
# 4. Prepare final dataframe
# -------------------------------------------------
df_model = df[numeric_features + categorical_features + ["price_range"]].copy()
df_model = pd.get_dummies(df_model, columns=categorical_features, drop_first=True)
df_model = df_model.dropna()

print("\nFinal shape after cleaning:", df_model.shape)

# -------------------------------------------------
# 5. Separate features and target
# -------------------------------------------------
X = df_model.drop("price_range", axis=1)
y = df_model["price_range"]

print("Number of features:", X.shape[1])

# -------------------------------------------------
# 6. Scale features
# -------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, "scaler_2026.pkl")
print("Scaler saved as → scaler_2026.pkl")

# -------------------------------------------------
# 7. Train / Test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=4)
y_test_cat  = tf.keras.utils.to_categorical(y_test, num_classes=4)

# -------------------------------------------------
# 8. Build the Neural Network
# -------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------------------------
# 9. Train the model (with epochs + EarlyStopping)
# -------------------------------------------------
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "best_mobile_price_model_2026.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

print("\nStarting training...\n")

history = model.fit(
    X_train, y_train_cat,
    epochs=50,                          # ← You will see 1/50, 2/50 ...
    batch_size=32,
    validation_data=(X_test, y_test_cat),
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# -------------------------------------------------
# 10. Plot Training History
# -------------------------------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# -------------------------------------------------
# 11. Evaluate
# -------------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High", "Very High"]))

plt.figure(figsize=(7, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=["Low", "Medium", "High", "Very High"],
            yticklabels=["Low", "Medium", "High", "Very High"])
plt.title("Confusion Matrix - 2026 Dataset")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# -------------------------------------------------
# 12. Save final model
# -------------------------------------------------
model.save("mobile_price_model_2026.h5")
print("\nModel saved as → mobile_price_model_2026.h5")
print("Best model saved as → best_mobile_price_model_2026.h5")
print("Scaler saved as → scaler_2026.pkl")
