import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(page_title="Customer Segmentation App", layout="centered")

# Load the saved models
@st.cache_resource
def load_models():
    scaler = joblib.load('scaler_model.pkl')
    kmeans = joblib.load('kmeans_model.pkl')
    return scaler, kmeans

try:
    scaler, kmeans = load_models()

    st.title("🛍️ Mall Customer Segmentation")
    st.write("""
    This app uses a **K-Means Clustering** model to categorize customers based on their 
    Annual Income and Spending Score.
    """)

    st.divider()

    # User Input Sidebar or Columns
    st.subheader("Input Customer Details")
    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input("Annual Income (k$)", min_value=1, max_value=200, value=50)
    
    with col2:
        spending_score = st.slider("Spending Score (1-100)", 1, 100, 50)

    if st.button("Predict Cluster"):
        # 1. Prepare the input data
        input_data = pd.DataFrame([[income, spending_score]], 
                                 columns=['Annual Income (k$)', 'Spending Score (1-100)'])

        # 2. Scale the data using the loaded scaler
        input_scaled = scaler.transform(input_data)

        # 3. Predict the cluster
        cluster_id = kmeans.predict(input_scaled)[0]

        # 4. Display results
        st.success(f"The customer belongs to **Cluster {cluster_id}**")

        # Optional: Add descriptions based on the clusters found in your notebook
        cluster_descriptions = {
            0: "Standard: Average income and average spending.",
            1: "Target: High income and high spending score.",
            2: "Sensible: Low income and high spending score.",
            3: "Careful: High income but low spending score.",
            4: "Miser: Low income and low spending score."
        }
        
        # Note: Match these descriptions to your specific cluster visualization results
        st.info(f"**Cluster Insight:** {cluster_descriptions.get(cluster_id, 'No description available.')}")

except FileNotFoundError:
    st.error("Error: Model files not found. Please ensure 'scaler_model.pkl' and 'kmeans_model.pkl' are in the same directory.")