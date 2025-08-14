import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
import time

# Set page configuration for a wider layout and custom title
st.set_page_config(layout="centered", page_title="Heart Health Predictor 💖")

# Apply custom CSS for a unique and beautiful look
st.markdown(
    """
    <style>
    /* General body styling */
    body {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f0f2f6 0%, #e0e5ec 100%); /* Soft, light gradient */
    }

    /* Main container styling */
    .st-emotion-cache-z5fcl4 { /* This targets the main content area */
        background-color: #ffffff;
        border-radius: 1.5rem; /* More rounded corners */
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1); /* Softer, larger shadow */
        padding: 2.5rem; /* Increased padding */
        border: 1px solid #e2e8f0; /* Subtle border */
    }

    /* Main header styling */
    .main-header {
        font-size: 3.5em; /* Larger font size */
        color: #E91E63; /* Vibrant pink/red */
        text-align: center;
        font-weight: 800; /* Extra bold */
        margin-bottom: 0.5em;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.15); /* More pronounced shadow */
        letter-spacing: -0.03em; /* Tighter letter spacing */
    }

    /* Subheader styling */
    .subheader {
        font-size: 1.4em; /* Slightly larger */
        color: #4CAF50; /* Green for health-related subheader */
        text-align: center;
        margin-bottom: 2.5em;
        font-weight: 500;
    }

    /* Styling for sidebar headers */
    .st-emotion-cache-1c7y2kd { /* Targets h2 in sidebar */
        color: #1976D2; /* Blue for sidebar header */
        font-size: 1.8em; /* Larger sidebar header */
        font-weight: 700;
        margin-bottom: 1em;
        text-align: center;
    }

    /* Styling for sidebar image container */
    .st-emotion-cache-1r6y40z { /* Targets image container in sidebar */
        border-radius: 1rem; /* Rounded corners */
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.15); /* Soft shadow */
        margin-bottom: 2em;
    }

    /* Styling for the prediction button */
    .stButton>button {
        background-color: #4CAF50; /* Green button */
        color: white;
        font-weight: bold;
        padding: 0.8em 1.5em; /* Adjusted padding */
        border-radius: 0.75rem; /* More rounded */
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(76, 175, 80, 0.3); /* Green shadow */
        font-size: 1.1em;
        width: 100%; /* Make button full width */
    }
    .stButton>button:hover {
        background-color: #388E3C; /* Darker green on hover */
        transform: translateY(-3px); /* Lift effect */
        box-shadow: 0 8px 20px rgba(76, 175, 80, 0.4); /* Larger shadow on hover */
    }

    /* Styling for success and warning messages */
    .stSuccess, .stWarning {
        border-radius: 1rem; /* Rounded corners */
        padding: 1.5em; /* Increased padding */
        font-size: 1.2em;
        font-weight: 600; /* Semi-bold */
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin-top: 1.5em; /* Space above messages */
    }
    .stSuccess {
        background-color: #e8f5e9; /* Light green background */
        color: #2e7d32; /* Dark green text */
        border: 1px solid #a5d6a7;
    }
    .stWarning {
        background-color: #ffebee; /* Light red background */
        color: #c62828; /* Dark red text */
        border: 1px solid #ef9a9a;
    }

    /* Styling for input labels */
    .st-emotion-cache-vk330f { /* Targets labels for input widgets */
        font-weight: 600;
        color: #333;
        margin-bottom: 0.5em;
    }

    /* Styling for selectboxes and radio buttons */
    .st-emotion-cache-1kyx5e9, .st-emotion-cache-1v0mbdj { /* Targets selectbox and radio button containers */
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 0.5em;
        border: 1px solid #ddd;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main title and subheader
st.markdown('<p class="main-header">❤️ Heart Disease Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Using Machine Learning for Early Diagnosis</p>', unsafe_allow_html=True)

# Project description and algorithms used
st.write("""
Heart disease prevention is critical, and data-driven prediction systems can significantly aid in early diagnosis and treatment. Machine Learning offers accurate prediction capabilities, enhancing healthcare outcomes. This project analyzes a heart disease dataset with appropriate preprocessing. Multiple classification algorithms were implemented in Python using Scikit-learn and Keras to predict the presence of heart disease.

**Algorithms Used:**
* Logistic Regression
* Naive Bayes
* Support Vector Machine (Linear)
* K-Nearest Neighbors
* Decision Tree
* Random Forest
* XGBoost
* Artificial Neural Network (1 Hidden Layer, Keras)
""")

# Load the trained model (heart_disease_pred.pkl)
try:
    with open('heart_disease_pred.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: Model file 'heart_disease_pred.pkl' not found. Please ensure it's in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load data to get min/max values for sliders (from your GitHub raw link)
url = '''https://github.com/ankitmisk/Heart_Disease_Prediction_ML_Model/blob/main/heart.csv?raw=true'''
try:
    df = pd.read_csv(url)
except Exception as e:
    st.error(f"Error loading dataset from URL: {e}")
    st.stop()

# Sidebar for patient feature selection
st.sidebar.header('Patient Features')
# Updated image parameter
st.sidebar.image('https://upload.wikimedia.org/wikipedia/ps/1/13/HeartBeat.gif', caption='Heartbeat Monitor', use_container_width=True)

# Dictionary to store input values
input_values = {}

# Configuration for each feature's input widget
features_config = {
    'age': {'type': 'slider', 'min': 29, 'max': 77, 'default': 50, 'step': 1},
    'sex': {'type': 'radio', 'options': {0: 'Female', 1: 'Male'}, 'default': 1},
    'cp': {'type': 'selectbox', 'options': {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-Anginal Pain', 3: 'Asymptomatic'}, 'default': 0},
    'trestbps': {'type': 'slider', 'min': 94, 'max': 200, 'default': 120, 'step': 1},
    'chol': {'type': 'slider', 'min': 126, 'max': 564, 'default': 240, 'step': 1},
    'fbs': {'type': 'radio', 'options': {0: 'False (<120 mg/dl)', 1: 'True (>120 mg/dl)'}, 'default': 0},
    'restecg': {'type': 'selectbox', 'options': {0: 'Normal', 1: 'ST-T wave abnormality', 2: 'Left ventricular hypertrophy'}, 'default': 0},
    'thalach': {'type': 'slider', 'min': 71, 'max': 202, 'default': 150, 'step': 1},
    'exang': {'type': 'radio', 'options': {0: 'No', 1: 'Yes'}, 'default': 0},
    'oldpeak': {'type': 'slider', 'min': 0.0, 'max': 6.2, 'default': 1.0, 'step': 0.1},
    'slope': {'type': 'selectbox', 'options': {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}, 'default': 0},
    'ca': {'type': 'slider', 'min': 0, 'max': 4, 'default': 0, 'step': 1},
    'thal': {'type': 'selectbox', 'options': {0: 'Normal', 1: 'Fixed Defect', 2: 'Reversible Defect'}, 'default': 0},
}

# Display input widgets in two columns in the sidebar for a compact layout
col1, col2 = st.columns(2)
sidebar_cols = [col1, col2]
current_col_idx = 0

for feature, config in features_config.items():
    with sidebar_cols[current_col_idx]:
        if config['type'] == 'slider':
            # Use dataframe's min/max if not explicitly set in config, otherwise use config values
            min_val = float(df[feature].min()) if 'min' not in config else float(config['min'])
            max_val = float(df[feature].max()) if 'max' not in config else float(config['max'])
            input_values[feature] = st.slider(
                f'Select **{feature.replace("_", " ").title()}** value',
                min_val,
                max_val,
                float(config['default']),
                float(config['step'])
            )
        elif config['type'] == 'radio':
            input_values[feature] = st.radio(
                f'Select **{feature.replace("_", " ").title()}**',
                list(config['options'].keys()),
                format_func=lambda x: config['options'][x],
                index=list(config['options'].keys()).index(config['default']) # Set index based on default value
            )
        elif config['type'] == 'selectbox':
            input_values[feature] = st.selectbox(
                f'Select **{feature.replace("_", " ").title()}**',
                list(config['options'].keys()),
                format_func=lambda x: config['options'][x],
                index=list(config['options'].keys()).index(config['default']) # Set index based on default value
            )
    current_col_idx = (current_col_idx + 1) % len(sidebar_cols)

# Convert input values to a numpy array for prediction
final_input_array = np.array([list(input_values.values())])

# Separator before the prediction button
st.markdown("---")

# Prediction button
if st.button('Predict Heart Disease Likelihood'):
    with st.spinner('Predicting...'):
        # Simulate a small delay for better user experience
        time.sleep(1.5)
        # Perform prediction using the loaded model
        prediction = model.predict(final_input_array)[0]

    # Display prediction result
    if prediction == 0:
        st.success('✅ Prediction: Low Likelihood of Heart Disease')
    else:
        st.warning('⚠️ Prediction: High Likelihood of Heart Disease')

# Separator and footer
st.markdown("---")
st.markdown("Developed with ❤️ by **Dhruv Sharma**")
