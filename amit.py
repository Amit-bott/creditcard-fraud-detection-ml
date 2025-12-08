# # creditcard_dashboard.py
# import streamlit as st
# import pandas as pd
# import pickle
# import numpy as np
# import os

# # Default paths (update if you placed them elsewhere)
# OUTPUT_DIR = "creditcard_model_output"
# MODEL_PATH = os.path.join(OUTPUT_DIR, "final_model.pkl")
# SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")
# IMPUTER_PATH = os.path.join(OUTPUT_DIR, "imputer.pkl")
# SNAPSHOT = os.path.join(OUTPUT_DIR, "used_data_head.csv")
# TARGET_COL = "Class"  # change if your dataset uses a different target name

# @st.cache_resource
# def load_objects():
#     if not os.path.exists(MODEL_PATH):
#         st.error(f"Model not found at {MODEL_PATH}. Run training notebook first.")
#         return None, None, None
#     with open(MODEL_PATH, "rb") as f:
#         model = pickle.load(f)
#     with open(SCALER_PATH, "rb") as f:
#         scaler = pickle.load(f)
#     with open(IMPUTER_PATH, "rb") as f:
#         imputer = pickle.load(f)
#     return model, scaler, imputer

# model, scaler, imputer = load_objects()

# st.title("Credit Card Prediction — Demo")
# st.markdown("Enter feature values for a single transaction in the sidebar and click **Predict**.")

# if model is None:
#     st.stop()

# snapshot = pd.read_csv(SNAPSHOT)
# feature_cols = [c for c in snapshot.columns if c != TARGET_COL]

# st.sidebar.header("Features")
# input_vals = {}
# for col in feature_cols:
#     default = float(snapshot[col].iloc[0]) if col in snapshot.columns else 0.0
#     input_vals[col] = st.sidebar.number_input(col, value=default)

# if st.button("Predict"):
#     X = pd.DataFrame([input_vals], columns=feature_cols)
#     X = X.select_dtypes(include=[np.number])
#     X_im = imputer.transform(X)
#     X_sc = scaler.transform(X_im)
#     pred = model.predict(X_sc)[0]
#     prob = model.predict_proba(X_sc)[0][1] if hasattr(model, "predict_proba") else None
#     st.write("Predicted class:", int(pred))
#     if prob is not None:
#         st.write("Probability (positive class):", float(prob))





# creditcard_dashboard.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

OUTPUT_DIR = "creditcard_model_output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "final_model.pkl")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")
IMPUTER_PATH = os.path.join(OUTPUT_DIR, "imputer.pkl")
SNAPSHOT = os.path.join(OUTPUT_DIR, "used_data_head.csv")
TARGET_COL = "Class"

# -----------------------------
# LOAD OBJECTS
# -----------------------------
@st.cache_resource
def load_objects():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        with open(IMPUTER_PATH, "rb") as f:
            imputer = pickle.load(f)
        return model, scaler, imputer
    except Exception as e:
        st.error(f"❌ Could not load model files. Error:\n{e}")
        return None, None, None

model, scaler, imputer = load_objects()

if model is None:
    st.stop()


# -----------------------------
# LOAD SAMPLE DATA
# -----------------------------
snapshot = pd.read_csv(SNAPSHOT)
feature_cols = [c for c in snapshot.columns if c != TARGET_COL]


# -----------------------------
# NAVIGATION MENU
# -----------------------------
menu = st.sidebar.radio(
    "📌 Navigation",
    ["🔮 Prediction", "📊 Data Preview", "📉 Model Info"],
)


# ======================================================================================
# 1️⃣ PREDICTION PAGE
# ======================================================================================
if menu == "🔮 Prediction":
    st.title("🔮 Credit Card Fraud Prediction")

    st.markdown(
        """
        Enter the transaction details on the left.  
        Click **Predict** to check whether it is **Fraud** or **Not Fraud**.
        """
    )

    # ---- Input UI ---------------------
    st.sidebar.header("📝 Input Features")
    input_vals = {}
    for col in feature_cols:
        default = float(snapshot[col].iloc[0])
        input_vals[col] = st.sidebar.number_input(col, value=default)

    # ---- Prediction Button ------------
    if st.button("🚀 Predict"):
        X = pd.DataFrame([input_vals], columns=feature_cols)
        X = X.select_dtypes(include=[np.number])

        X_im = imputer.transform(X)
        X_sc = scaler.transform(X_im)

        pred = int(model.predict(X_sc)[0])
        proba = float(model.predict_proba(X_sc)[0][1])

        # ---- Metric Cards ----------------
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Prediction",
                "Fraud ❗" if pred == 1 else "Not Fraud ✔️",
                delta=None
            )

        with col2:
            st.metric(
                "Fraud Probability",
                f"{proba*100:.2f}%"
            )

        with col3:
            st.metric(
                "Safe Probability",
                f"{(1 - proba) * 100:.2f}%"
            )

        # ---- Gauge Chart -------------------
        st.markdown("### 📈 Fraud Risk Indicator")
        st.progress(proba)

        # ---- Raw Output -------------------
        with st.expander("🔍 Detailed Prediction Output"):
            st.write("Input Data", X)
            st.write("Scaled Data", X_sc)
            st.write("Predicted Class:", pred)
            st.write("Probability (Fraud):", proba)


# ======================================================================================
# 2️⃣ DATA PREVIEW PAGE
# ======================================================================================
elif menu == "📊 Data Preview":
    st.title("📊 Dataset Preview")

    st.write(
        """
        This is a snapshot of the dataset used during training.
        Useful for understanding feature ranges and values.
        """
    )

    st.dataframe(snapshot)

    st.markdown("### 📌 Feature Summary")
    st.write(snapshot.describe())


# ======================================================================================
# 3️⃣ MODEL INFO PAGE
# ======================================================================================
elif menu == "📉 Model Info":
    st.title("📉 Model Information")

    st.write("Below are details of the loaded model and preprocessing pipeline.")

    st.markdown("### 🔧 Model Type")
    st.write(type(model))

    st.markdown("### 🔧 Scaler")
    st.write(type(scaler))

    st.markdown("### 🔧 Imputer")
    st.write(type(imputer))

    st.markdown("### 📁 Loaded Files")
    st.json(
        {
            "Model": MODEL_PATH,
            "Scaler": SCALER_PATH,
            "Imputer": IMPUTER_PATH,
            "Data Snapshot": SNAPSHOT,
        }
    )

    st.markdown("### 📄 Feature List")
    st.write(feature_cols)
