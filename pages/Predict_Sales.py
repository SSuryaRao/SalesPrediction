import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Page Config and Title ---
st.set_page_config(page_title="Sales Prediction", layout="centered")
st.title("🛍️ Super Store Sales Prediction")
st.markdown("Fill in the details below to predict the **Sales Amount**.")

# --- Paths ---
# Adjust paths to correctly reference files from the 'pages' directory
# This assumes 'models' and 'data' folders are in the root directory of your Streamlit app
current_dir = os.path.dirname(__file__) # Get the directory of the current script (pages)
root_dir = os.path.join(current_dir, "..") # Go up one level to the root app directory

model_path = os.path.join(root_dir, "models", "sales_model.pkl")
features_path = os.path.join(root_dir, "models", "feature_columns.pkl")
test_data_path = os.path.join(root_dir, "data", "test_data.csv")

# --- File Checks ---
if not os.path.exists(model_path):
    st.error(f"❌ Model file not found at: {model_path}. Ensure 'sales_model.pkl' exists in the 'models/' folder.")
    st.stop()
if not os.path.exists(features_path):
    st.error(f"❌ Feature columns file not found at: {features_path}. Ensure 'feature_columns.pkl' exists in the 'models/' folder.")
    st.stop()


# --- Load Model and Features ---
try:
    model = joblib.load(model_path)
    feature_columns = joblib.load(features_path)
except Exception as e:
    st.error(f"Error loading model files: {e}")
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


# --- Input Form for Single Prediction ---
st.subheader("📝 Input Features")
with st.form("prediction_form"):
    user_input = {}
    # Ensure feature_columns is a list
    if isinstance(feature_columns, list):
        for col in feature_columns:
            # Check for numeric or categorical to create appropriate input field
            if col in ["Age", "Qty", "day", "weekday", "year"]:
                user_input[col] = st.number_input(f"Enter {col}", min_value=0, step=1, key=f"single_{col}")
            elif col in dropdown_options:
                user_input[col] = st.selectbox(f"Select {col}", options=dropdown_options[col], key=f"single_{col}")
            else:
                # Fallback for other columns, assumes text. Adjust if needed.
                user_input[col] = st.text_input(f"Enter {col}", key=f"single_{col}")
    
    submitted = st.form_submit_button("🚀 Predict Sales")

# --- Perform Single Prediction ---
if submitted:
    try:
        # Create DataFrame and ensure columns are in the correct order
        input_df = pd.DataFrame([user_input])[feature_columns]
        
        # The model pipeline handles encoding, so we pass raw data
        prediction = model.predict(input_df)
        
        # The output of a scikit-learn regressor is a numpy array
        predicted_amount = prediction[0]
        
        st.success(f"💰 **Predicted Sales Amount:** ₹{predicted_amount:,.2f}")
    except Exception as e:
        st.error(f"❌ **Prediction failed:** {e}")
        st.info("Please ensure all input fields are filled correctly. The model may not have been trained on the combination of inputs provided.")


