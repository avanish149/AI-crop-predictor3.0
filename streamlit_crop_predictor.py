import random
import os
import pandas as pd
import streamlit as st
import numpy as np
import pickle

# -------------------------------------------------
# 1) Page setup
# -------------------------------------------------
st.title("AI Crop Predictor Dashboard")
st.write(
    "This dashboard uses a machine‑learning model trained on soil nutrients, "
    "weather conditions, and historical crop records to recommend the most "
    "suitable crop for your field. Enter values for nitrogen, phosphorus, "
    "potassium, temperature, humidity, pH, and rainfall to see the suggested "
    "crop along with an estimated market rate and expected yield."
)

DATAFILE = "crop_recommendation.csv"
MODELFILE = "crop_model.pkl"

# FIXED: Exact 7-feature columns (no rates/yield in model input)
FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# -------------------------------------------------
# 2) Load dataset
# -------------------------------------------------
if not os.path.isfile(DATAFILE):
    st.error(f"Dataset file '{DATAFILE}' not found in the current directory.")
    st.stop()

data = pd.read_csv(DATAFILE)

# Standardize column names
column_map = {
    "Nitrogen": "N",
    "Phosphorus": "P",
    "Potassium": "K",
    "Temperature": "temperature",
    "Humidity": "humidity",
    "pH_Value": "ph",
    "Rainfall": "rainfall",
    "Crop": "label",
}
data = data.rename(columns=column_map)

# FIXED: Only require model features + label (rates/yield optional for display)
required_columns = set(FEATURE_COLS) | {"label"}
missing = required_columns - set(data.columns)
if missing:
    st.error(f"Dataset is missing columns: {missing}")
    st.stop()

# -------------------------------------------------
# 3) Load pre-trained model (no training in app)
# -------------------------------------------------
if not os.path.isfile(MODELFILE):
    st.error(
        f"Model file '{MODELFILE}' not found. "
        f"Run your training script (crop_predictor_safe.py) first."
    )
    st.stop()


@st.cache_resource
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


model = load_model(MODELFILE)

st.subheader("Model status")
st.success("✅ RandomForest loaded from file.")
st.info("**Offline test accuracy: 99.32%**")
st.caption(f"Expected features: {model.feature_names_in_.tolist()}")

# -------------------------------------------------
# 4) Dataset preview
# -------------------------------------------------
st.subheader("Dataset Preview")
st.dataframe(data[FEATURE_COLS + ['label']].head(100), use_container_width=True)

# -------------------------------------------------
# 5) User input section
# -------------------------------------------------
st.subheader("Enter Values to Predict Crop")

if "rand_values" not in st.session_state:
    st.session_state["rand_values"] = {
        "N": 90,  # Rice defaults for testing
        "P": 42,
        "K": 43,
        "temperature": 20.88,
        "humidity": 82.02,
        "ph": 6.5,
        "rainfall": 202.93,
    }


def randomize_inputs():
    st.session_state["rand_values"] = {
        "N": random.randint(0, 200),
        "P": random.randint(0, 200),
        "K": random.randint(0, 200),
        "temperature": round(random.uniform(0, 60), 2),
        "humidity": round(random.uniform(0, 100), 2),
        "ph": round(random.uniform(0, 14), 2),
        "rainfall": round(random.uniform(0, 400), 2),
    }


if st.button("🎲 Randomize Inputs"):
    randomize_inputs()
    st.rerun()

with st.form("prediction_form"):
    N = st.number_input("Nitrogen (N)", 0, 200, st.session_state["rand_values"]["N"])
    P = st.number_input("Phosphorus (P)", 0, 200, st.session_state["rand_values"]["P"])
    K = st.number_input("Potassium (K)", 0, 200, st.session_state["rand_values"]["K"])
    temperature = st.number_input(
        "Temperature (°C)", 0.0, 60.0, st.session_state["rand_values"]["temperature"]
    )
    humidity = st.number_input(
        "Humidity (%)", 0.0, 100.0, st.session_state["rand_values"]["humidity"]
    )
    ph = st.number_input(
        "pH Value", 0.0, 14.0, st.session_state["rand_values"]["ph"]
    )
    rainfall = st.number_input(
        "Rainfall (mm)", 0.0, 400.0, st.session_state["rand_values"]["rainfall"]
    )
    submit = st.form_submit_button("🔮 Predict Crop", use_container_width=True)

# Your existing crop lookup (unchanged)
crop_data = {
    "rice": (25.5, 3850),
    "maize": (18.2, 4200),
    "chickpea": (65.0, 850),
    "kidneybeans": (45.0, 2800),
    "pigeonpeas": (70.0, 720),
    "mothbeans": (55.0, 450),
    "mungbean": (60.0, 500),
    "blackgram": (58.0, 480),
    "lentil": (62.0, 950),
    "pomegranate": (80.0, 22000),
    "banana": (35.0, 35000),
    "mango": (45.0, 8500),
    "grapes": (120.0, 22000),
    "watermelon": (12.0, 25000),
    "muskmelon": (15.0, 28000),
    "apple": (150.0, 20000),
    "orange": (40.0, 15000),
    "papaya": (25.0, 35000),
    "coconut": (30.0, 14000),
    "cotton": (120.0, 800),
    "jute": (35.0, 2500),
    "coffee": (200.0, 1200),
}


# -------------------------------------------------
# FIXED: Prediction using ONLY 7 model features
# -------------------------------------------------
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    # EXACT training format - 7 features only
    input_df = pd.DataFrame([
        [N, P, K, temperature, humidity, ph, rainfall]
    ], columns=FEATURE_COLS)

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]
    confidence = max(prob) * 100

    return pred, confidence


# -------------------------------------------------
# 6) Results display
# -------------------------------------------------
if submit:
    with st.spinner("Predicting..."):
        pred, confidence = predict_crop(N, P, K, temperature, humidity, ph, rainfall)

        # Debug info (remove after testing)
        with st.expander("🔍 Debug Info"):
            st.write("**Input sent to model:**")
            input_df = pd.DataFrame([
                [N, P, K, temperature, humidity, ph, rainfall]
            ], columns=FEATURE_COLS)
            st.dataframe(input_df)
            st.write(f"**Model expects:** {model.feature_names_in_.tolist()}")

        crop_key = str(pred).strip().lower()
        rate, yld = crop_data.get(crop_key, (25.5, 3850))

        # Results layout
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"""
            <div style="color:#10b981; font-size:2rem; font-weight:bold;">
                {pred}
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.metric("Confidence", f"{confidence:.1f}%")
        with col3:
            st.metric("Rate", f"₹{rate:.1f}/kg")

        st.markdown(f"""
        **Estimated yield:** {yld:.0f} kg/ha  
        **Optimal for:** N:{N:.0f}, P:{P:.0f}, K:{K:.0f} | {temperature:.1f}°C, {humidity:.1f}% humidity
        """)

# -------------------------------------------------
# 7) Basic statistics
# -------------------------------------------------
st.subheader("📊 Dataset Statistics")
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Samples", len(data))
    st.metric("Crop Types", data['label'].nunique())
with col2:
    st.metric("Avg N", f"{data['N'].mean():.1f}")
    st.metric("Avg Rainfall", f"{data['rainfall'].mean():.1f} mm")

st.write(data[FEATURE_COLS + ['label']].describe())

# Instructions
with st.expander("🚀 Quick Start"):
    st.markdown("""
    **1. Train model:** `python crop_predictor_safe.py`  
    **2. Run app:** `streamlit run streamlit_crop_predictor.py`  
    **3. Test with Rice:** N=90, P=42, K=43, temp=20.9, hum=82, pH=6.5, rain=203  
    **Expected:** "rice" with 99%+ confidence
    """)


