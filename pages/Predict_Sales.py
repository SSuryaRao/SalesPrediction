import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import gdown

# --- Page Config and Title ---
st.set_page_config(page_title="Sales Prediction", layout="centered")
st.title("🛍️ Super Store Sales Prediction")
st.markdown("Fill in the details below to predict the **Sales Amount**.")

# --- Google Drive Model File ID ---
GDRIVE_MODEL_FILE_ID = "1z65zbCMMNms-xrpuuePG_BEXDt6MVROX"  # 🔁 Replace this with your actual file ID from Google Drive

# --- Directory Paths ---
current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, "..")
model_dir = os.path.join(root_dir, "models")
data_dir = os.path.join(root_dir, "data")

model_path = os.path.join(model_dir, "sales_model.pkl")
features_path = os.path.join(model_dir, "feature_columns.pkl")
test_data_path = os.path.join(data_dir, "test_data.csv")

# --- Function to download model from Google Drive ---
def download_model_from_gdrive(file_id, dest_path):
    if not os.path.exists(dest_path):
        st.info("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, dest_path, quiet=False)

# --- Download and Load Model ---
try:
    download_model_from_gdrive(GDRIVE_MODEL_FILE_ID, model_path)
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Failed to download or load model: {e}")
    st.stop()

# --- Load Feature Columns from Local ---
if not os.path.exists(features_path):
    st.error(f"❌ Feature columns file not found at: {features_path}")
    st.stop()

try:
    feature_columns = joblib.load(features_path)
except Exception as e:
    st.error(f"❌ Failed to load feature columns: {e}")
    st.stop()

# --- Dropdown Options from Test Data ---
dropdown_options = {}
if os.path.exists(test_data_path):
    try:
        test_df_for_options = pd.read_csv(test_data_path)
        for col in feature_columns:
            if col in test_df_for_options.columns and test_df_for_options[col].dtype == 'object':
                dropdown_options[col] = sorted(test_df_for_options[col].dropna().unique().tolist())
    except Exception as e:
        st.warning(f"Could not infer dropdown options from test data: {e}")

# --- Input Form for Prediction ---
st.subheader("📝 Input Features")
with st.form("prediction_form"):
    user_input = {}
    if isinstance(feature_columns, list):
        for col in feature_columns:
            if col in ["Age", "Qty", "day", "weekday", "year"]:
                user_input[col] = st.number_input(f"Enter {col}", min_value=0, step=1, key=f"single_{col}")
            elif col in dropdown_options:
                user_input[col] = st.selectbox(f"Select {col}", options=dropdown_options[col], key=f"single_{col}")
            else:
                user_input[col] = st.text_input(f"Enter {col}", key=f"single_{col}")

    submitted = st.form_submit_button("🚀 Predict Sales")

# --- Perform Prediction ---
if submitted:
    try:
        input_df = pd.DataFrame([user_input])[feature_columns]
        prediction = model.predict(input_df)
        predicted_amount = prediction[0]
        st.success(f"💰 **Predicted Sales Amount:** ₹{predicted_amount:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
