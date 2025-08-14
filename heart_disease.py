import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
import time

# Set page configuration for a wider layout and custom title
st.set_page_config(layout="wide", page_title="Heart Health Predictor 💖") # Changed to wide layout

# Apply custom CSS for a unique and beautiful look
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="st-emotion-cache"] {
        font-family: 'Inter', sans-serif;
        color: #333;
    }

    /* Main container styling */
    .st-emotion-cache-z5fcl4 { /* Targets the main content area */
        background: linear-gradient(135deg, #ffffff 0%, #f0f8ff 100%); /* Soft blue-white gradient */
        border-radius: 1.5rem; /* More rounded corners */
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.1); /* Softer, larger shadow */
        padding: 3rem; /* Increased padding */
        border: 1px solid #e0eaf6; /* Subtle border */
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

    /* Main header styling */
    .main-header {
        font-size: 3.8em; /* Even larger font size */
        color: #D32F2F; /* Deep red for emphasis */
        text-align: center;
        font-weight: 800; /* Extra bold */
        margin-bottom: 0.2em;
        text-shadow: 4px 4px 8px rgba(0,0,0,0.18); /* More pronounced shadow */
        letter-spacing: -0.04em; /* Tighter letter spacing */
    }

    /* Subheader styling */
    .subheader {
        font-size: 1.6em; /* Larger subheader */
        color: #5C6BC0; /* Muted blue for subheader */
        text-align: center;
        margin-bottom: 2.5em;
        font-weight: 600;
    }

    /* Styling for sidebar headers */
    .st-emotion-cache-1c7y2kd { /* Targets h2 in sidebar */
        color: #0D47A1; /* Darker blue for sidebar header */
        font-size: 2em; /* Larger sidebar header */
        font-weight: 700;
        margin-bottom: 1.5em;
        text-align: center;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
    }

    /* Styling for sidebar image container */
    .st-emotion-cache-1r6y40z { /* Targets image container in sidebar */
        border-radius: 1.2rem; /* Rounded corners */
        overflow: hidden;
        box-shadow: 0 8px 20px rgba(0,0,0,0.18); /* Soft shadow */
        margin-bottom: 2.5em;
        border: 2px solid #CFD8DC; /* Light grey border */
    }

    /* Styling for the prediction button */
    .stButton>button {
        background: linear-gradient(45deg, #4CAF50 0%, #66BB6A 100%); /* Green gradient */
        color: white;
        font-weight: bold;
        padding: 1em 2em; /* Larger padding */
        border-radius: 1rem; /* More rounded */
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.4); /* Green shadow */
        font-size: 1.2em;
        width: 100%; /* Make button full width */
        letter-spacing: 0.05em;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #388E3C 0%, #4CAF50 100%); /* Darker green gradient on hover */
        transform: translateY(-4px); /* Lift effect */
        box-shadow: 0 12px 30px rgba(76, 175, 80, 0.5); /* Larger shadow on hover */
    }

    /* Styling for success and warning messages */
    .stSuccess, .stWarning {
        border-radius: 1.2rem; /* Rounded corners */
        padding: 1.8em; /* Increased padding */
        font-size: 1.3em;
        font-weight: 700; /* Bold */
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        margin-top: 2em; /* Space above messages */
        border-left: 8px solid; /* Bold left border */
    }
    .stSuccess {
        background-color: #e8f5e9; /* Light green background */
        color: #2E7D32; /* Dark green text */
        border-color: #4CAF50;
    }
    .stWarning {
        background-color: #ffebee; /* Light red background */
        color: #C62828; /* Dark red text */
        border-color: #D32F2F;
    }

    /* Styling for input labels */
    .st-emotion-cache-vk330f { /* Targets labels for input widgets */
        font-weight: 600;
        color: #444;
        margin-bottom: 0.6em;
        font-size: 1.05em;
    }

    /* Styling for input widgets (sliders, selectboxes, radio buttons) */
    .st-emotion-cache-1kyx5e9, .st-emotion-cache-1v0mbdj, .st-emotion-cache-1q1n004 {
        background-color: #fcfdff;
        border-radius: 0.8rem;
        padding: 0.7em;
        border: 1px solid #c0d0e0;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.05); /* Inner shadow for depth */
    }

    /* Styling for the main project description text */
    .st-emotion-cache-10qg059 {
        font-size: 1.1em;
        line-height: 1.6;
        color: #555;
    }

    /* Image styling within the main content */
    .main-image {
        border-radius: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-bottom: 2em;
        border: 1px solid #e2e8f0;
    }

    /* Footer styling */
    .st-emotion-cache-ch5fnp { /* Targets markdown for footer */
        text-align: center;
        color: #777;
        font-size: 0.9em;
        margin-top: 3em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main title and subheader
st.markdown('<p class="main-header">❤️ Heart Disease Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Leveraging Machine Learning for Proactive Health Insights</p>', unsafe_allow_html=True)

# Main project image
st.image('https://itdesigners.org/wp-content/uploads/2024/02/heart-1024x576.jpg', caption='Predicting Heart Health', use_container_width=True, classes="main-image")

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
    # Explicitly convert relevant columns to numeric to prevent TypeError
    numeric_cols = [
        'age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Fill any NaN values that might result from coercion (e.g., if there were non-numeric strings)
        # For this specific dataset, original EDA showed no NaNs, but this is good practice.
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)

except Exception as e:
    st.error(f"Error loading dataset from URL: {e}")
    st.stop()

# Sidebar for patient feature selection
st.sidebar.header('Patient Features')
st.sidebar.image('https://upload.wikimedia.org/wikipedia/ps/1/13/HeartBeat.gif', caption='Heartbeat Monitor', use_container_width=True)

# Dictionary to store input values
input_values = {}

# Configuration for each feature's input widget
features_config = {
    'age': {'type': 'slider', 'min': 29, 'max': 77, 'default': 50, 'step': 1, 'label': 'Age'},
    'sex': {'type': 'radio', 'options': {0: 'Female', 1: 'Male'}, 'default': 1, 'label': 'Sex'},
    'cp': {'type': 'selectbox', 'options': {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-Anginal Pain', 3: 'Asymptomatic'}, 'default': 0, 'label': 'Chest Pain Type'},
    'trestbps': {'type': 'slider', 'min': 94, 'max': 200, 'default': 120, 'step': 1, 'label': 'Resting Blood Pressure (trestbps)'},
    'chol': {'type': 'slider', 'min': 126, 'max': 564, 'default': 240, 'step': 1, 'label': 'Cholesterol (chol)'},
    'fbs': {'type': 'radio', 'options': {0: 'False (<120 mg/dl)', 1: 'True (>120 mg/dl)'}, 'default': 0, 'label': 'Fasting Blood Sugar > 120 mg/dl'},
    'restecg': {'type': 'selectbox', 'options': {0: 'Normal', 1: 'ST-T wave abnormality', 2: 'Left ventricular hypertrophy'}, 'default': 0, 'label': 'Resting ECG Results'},
    'thalach': {'type': 'slider', 'min': 71, 'max': 202, 'default': 150, 'step': 1, 'label': 'Max Heart Rate Achieved (thalach)'},
    'exang': {'type': 'radio', 'options': {0: 'No', 1: 'Yes'}, 'default': 0, 'label': 'Exercise Induced Angina'},
    'oldpeak': {'type': 'slider', 'min': 0.0, 'max': 6.2, 'default': 1.0, 'step': 0.1, 'label': 'ST Depression (oldpeak)'},
    'slope': {'type': 'selectbox', 'options': {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}, 'default': 0, 'label': 'Slope of Peak ST Segment'},
    'ca': {'type': 'slider', 'min': 0, 'max': 4, 'default': 0, 'step': 1, 'label': 'Number of Major Vessels (0-3)'},
    'thal': {'type': 'selectbox', 'options': {0: 'Normal', 1: 'Fixed Defect', 2: 'Reversible Defect'}, 'default': 0, 'label': 'Thalassemia'},
}

# Display input widgets in two columns in the sidebar for a compact layout
col1, col2 = st.columns(2)
sidebar_cols = [col1, col2]
current_col_idx = 0

for feature, config in features_config.items():
    with sidebar_cols[current_col_idx]:
        label = config.get('label', feature.replace("_", " ").title())
        if config['type'] == 'slider':
            min_val = float(df[feature].min()) if 'min' not in config else float(config['min'])
            max_val = float(df[feature].max()) if 'max' not in config else float(config['max'])
            input_values[feature] = st.slider(
                f'**{label}**', # Use bold for labels
                min_val,
                max_val,
                float(config['default']),
                float(config['step'])
            )
        elif config['type'] == 'radio':
            input_values[feature] = st.radio(
                f'**{label}**', # Use bold for labels
                list(config['options'].keys()),
                format_func=lambda x: config['options'][x],
                index=list(config['options'].keys()).index(config['default'])
            )
        elif config['type'] == 'selectbox':
            input_values[feature] = st.selectbox(
                f'**{label}**', # Use bold for labels
                list(config['options'].keys()),
                format_func=lambda x: config['options'][x],
                index=list(config['options'].keys()).index(config['default'])
            )
    current_col_idx = (current_col_idx + 1) % len(sidebar_cols)

# Convert input values to a numpy array for prediction
final_input_array = np.array([list(input_values.values())])

# Separator before the prediction button
st.markdown("---")

# Prediction button
if st.button('Predict Heart Disease Likelihood'):
    with st.spinner('Analyzing data...'):
        time.sleep(1.5) # Simulate processing time
        prediction = model.predict(final_input_array)[0]

    # Display prediction result
    if prediction == 0:
        st.success('✅ Prediction: Low Likelihood of Heart Disease. Keep up the good work!')
    else:
        st.warning('⚠️ Prediction: High Likelihood of Heart Disease. Consider consulting a healthcare professional.')

# Separator and footer
st.markdown("---")
st.markdown("Developed with ❤️ by **Dhruv Sharma**")
