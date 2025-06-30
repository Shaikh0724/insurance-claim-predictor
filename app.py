# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from scripts.preprocess import load_and_prepare_data
from scripts.model import train_and_evaluate_model

# Train model only once when app starts
@st.cache_data
def train_model_once():
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()
    return X_train, X_test, y_train, y_test, feature_names

# UI Layout
st.set_page_config(page_title="Medical Insurance Cost Predictor", layout="centered")
st.title("💊 Medical Insurance Cost Predictor")
st.write("Enter your health details to predict insurance charges.")

# User Inputs
age = st.slider("Age", 18, 65, 30)
sex = st.selectbox("Gender", ["male", "female"])
bmi = st.slider("BMI", 15.0, 50.0, 25.0)
children = st.slider("Number of Children", 0, 5, 1)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Prepare single input row
input_dict = {
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region
}
input_df = pd.DataFrame([input_dict])

# Load training data
X_train, X_test, y_train, y_test, feature_names = train_model_once()

# Combine with full dataset for encoding
df = pd.read_csv("data/insurance.csv")
df = pd.concat([df.drop(columns="charges"), input_df], ignore_index=True)
df_encoded = pd.get_dummies(df, drop_first=True)
X_input = df_encoded.tail(1)

# Predict with new model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_input)[0]

st.markdown(f"### 💰 Predicted Insurance Charges: **₹ {prediction:,.2f}**")

# Show feature importance image
if os.path.exists("outputs/visuals/feature_importance.png"):
    st.image("outputs/visuals/feature_importance.png", caption="📊 Feature Importance", use_column_width=True)
else:
    st.warning("⚠️ Feature importance plot not found. Run `main.py` first.")
