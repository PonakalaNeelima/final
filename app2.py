import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the models and scaler
try:
    meta_model = joblib.load('Stacked_Model.joblib')
    # Removed logistic_regression loading
    svm = joblib.load('SVM.joblib')
    tree = joblib.load('DecisionTree.joblib')
    forest = joblib.load('RandomForest.joblib')
    scaler = joblib.load('scaler.joblib')  # Load the scaler
    st.write("Models Loaded Successfully")
except FileNotFoundError:
    st.error("One or more model files not found. Please ensure they are in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Define the input parameters (Corrected and Consistent)
input_parameters = ['ph', 'hardness', 'turbidity', 'arsenic', 'chloramine', 'bacteria', 'lead', 'nitrates', 'mercury']

# Create a function to predict the water quality
def predict_water_quality(input_data):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure correct column order and names
        input_df = input_df[input_parameters]

        # Scale the input data
        X_scaled = scaler.transform(input_df)

        # Get predictions from base models (probabilities)
        svm_pred = svm.predict_proba(X_scaled)[:, 1]
        tree_pred = tree.predict_proba(X_scaled)[:, 1]
        forest_pred = forest.predict_proba(X_scaled)[:, 1]

        # Create a stacked input for the meta-model with 3 base models
        stacked_input = np.array([svm_pred[0], tree_pred[0], forest_pred[0]]).reshape(1, -1)

        # Get the final prediction from the meta-model
        final_prediction = meta_model.predict(stacked_input)[0]

        st.subheader("Model Predictions:")
        st.write(f"SVM Prediction(accuracy-93.03%): {int(np.round(svm_pred[0]))}")
        st.write(f"Decision Tree Prediction(accuracy-90.96%): {int(np.round(tree_pred[0]))}")
        st.write(f"Random Forest Prediction(accuracy-93.21%): {int(np.round(forest_pred[0]))}")
        st.write(f"Meta Model(LR) Prediction(accuracy-93.25%): {final_prediction}")

        return final_prediction
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


# Create the Streamlit web app
st.title('Safeguarding Public Health Through Comprehensive Water Purity Analysis')
input_parameters1 = ['ph (6.5-8.5)', 'hardness (60-180 mg/L)', 'turbidity (0-1 NTU)', 'arsenic (0-0.01 mg/L)', 'chloramine (0.5-2 mg/L)', 'bacteria (0-1 CFU/ml)', 'lead (0-0.01 mg/L)', 'nitrates (0-10 mg/L)', 'mercury (0-0.001 mg/L)']
param_mapping = {
    'ph (6.5-8.5)': 'ph',
    'hardness (60-180 mg/L)': 'hardness',
    'turbidity (0-1 NTU)': 'turbidity',
    'arsenic (0-0.01 mg/L)': 'arsenic',
    'chloramine (0.5-2 mg/L)': 'chloramine',
    'bacteria (0-1 CFU/ml)': 'bacteria',
    'lead (0-0.01 mg/L)': 'lead',
    'nitrates (0-10 mg/L)': 'nitrates',
    'mercury (0-0.001 mg/L)': 'mercury'
}

input_data = {}
for display_name in input_parameters1:
    actual_key = param_mapping[display_name]
    input_data[actual_key] = st.number_input(f'Enter {display_name} value:', value=0.0)

# Predict water quality when the 'Predict' button is clicked
if st.button('Predict'):
    prediction = predict_water_quality(input_data)

    if prediction is not None:
        if prediction == 0:
            st.subheader('The water is predicted to be: Unsafe')
        else:
            st.subheader('The water is predicted to be: Safe')
