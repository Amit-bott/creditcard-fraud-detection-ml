# creditcard_dashboard.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# Default paths (update if you placed them elsewhere)
OUTPUT_DIR = "creditcard_model_output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "final_model.pkl")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")
IMPUTER_PATH = os.path.join(OUTPUT_DIR, "imputer.pkl")
SNAPSHOT = os.path.join(OUTPUT_DIR, "used_data_head.csv")
TARGET_COL = "Class"  # change if your dataset uses a different target name

@st.cache_resource
def load_objects():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}. Run training notebook first.")
        return None, None, None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(IMPUTER_PATH, "rb") as f:
        imputer = pickle.load(f)
    return model, scaler, imputer

model, scaler, imputer = load_objects()

st.title("Credit Card Prediction — Demo")
st.markdown("Enter feature values for a single transaction in the sidebar and click **Predict**.")

if model is None:
    st.stop()

snapshot = pd.read_csv(SNAPSHOT)
feature_cols = [c for c in snapshot.columns if c != TARGET_COL]

st.sidebar.header("Features")
input_vals = {}
for col in feature_cols:
    default = float(snapshot[col].iloc[0]) if col in snapshot.columns else 0.0
    input_vals[col] = st.sidebar.number_input(col, value=default)

if st.button("Predict"):
    X = pd.DataFrame([input_vals], columns=feature_cols)
    X = X.select_dtypes(include=[np.number])
    X_im = imputer.transform(X)
    X_sc = scaler.transform(X_im)
    pred = model.predict(X_sc)[0]
    prob = model.predict_proba(X_sc)[0][1] if hasattr(model, "predict_proba") else None
    st.write("Predicted class:", int(pred))
    if prob is not None:
        st.write("Probability (positive class):", float(prob))
