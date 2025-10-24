# train_and_save.py

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --------------------------
# Step 1: Load CSV
# --------------------------
csv_path = r"C:\Users\Sourodip\Desktop\predictive-maintenance-project\data\PdM_telemetry.csv"
data = pd.read_csv(csv_path)

# --------------------------
# Step 2: Create 'failure' column
# --------------------------
# Example: failure = 1 if vibration > 50 or volt < 150 else 0
data['failure'] = ((data['vibration'] > 50) | (data['volt'] < 150)).astype(int)

print("Sample data with failure column:")
print(data.head())

# --------------------------
# Step 3: Prepare features and labels
# --------------------------
features = ['volt', 'rotate', 'pressure', 'vibration']
X = data[features].values
y = data['failure'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Reshape for CNN-LSTM: (samples, timesteps=1, features)
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --------------------------
# Step 4: Build CNN + LSTM model
# --------------------------
model = Sequential([
    Conv1D(64, 2, activation='relu', input_shape=(1, 4)),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --------------------------
# Step 5: Train the model
# --------------------------
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

# --------------------------
# Step 6: Evaluate model
# --------------------------
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

# --------------------------
# Step 7: Save the model
# --------------------------
model.save("PrognosAI_model.h5")
print("Model saved as PrognosAI_model.h5")
