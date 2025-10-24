# app.py

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# --------------------------
# Step 1: Load the trained model
# --------------------------
model = load_model("PrognosAI_model.h5")
st.title("PrognosAI - Predictive Maintenance")
st.write("Upload your CSV file to predict machine failures.")

# --------------------------
# Step 2: Upload CSV
# --------------------------
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data preview:")
    st.dataframe(data.head())

    # --------------------------
    # Step 3: Check required columns
    # --------------------------
    required_cols = ['volt', 'rotate', 'pressure', 'vibration']
    if not all(col in data.columns for col in required_cols):
        st.error(f"CSV must contain columns: {required_cols}")
    else:
        # --------------------------
        # Step 4: Preprocess features
        # --------------------------
        X = data[required_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))  # (samples, 1, features)

        # --------------------------
        # Step 5: Predict failures
        # --------------------------
        preds = model.predict(X_scaled)
        data['failure_prediction'] = (preds > 0.5).astype(int)

        st.write("Prediction results:")
        st.dataframe(data)

        # Optional: show basic stats
        st.write("Failure summary:")
        st.write(data['failure_prediction'].value_counts())
