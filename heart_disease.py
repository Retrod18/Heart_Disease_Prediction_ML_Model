# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
import time

# Set page configuration for a wider layout and custom title
st.set_page_config(
    layout="wide",
    page_title="Heart Health Predictor 💖",
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': 'https://www.streamlit.io/help',
        'Report a bug': "https://github.com/streamlit/streamlit/issues",
        'About': "# Heart Health Predictor"
    }
)

# --- Custom CSS for a unique and beautiful look ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="st-emotion-cache"] {
        font-family: 'Inter', sans-serif;
        color: #1a1a1a; /* Darker text color for better visibility */
    }

    /* Ensure the entire app background is light */
    .stApp {
        background: linear-gradient(135deg, #f0f2f6 0%, #e0e5ec 100%) !important; /* Force light gradient */
    }

    /* Main content area styling */
    .st-emotion-cache-z5fcl4 { /* Targets the main content area */
        background: linear-gradient(135deg, #ffffff 0%, #f0f8ff 100%) !important; /* Soft blue-white gradient */
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
        color: #4A55A2; /* Darker muted blue for subheader */
        text-align: center;
        margin-bottom: 2.5em;
        font-weight: 600;
    }

    /* --- Sidebar Styling --- */
    /* Target the main sidebar container */
    section[data-testid="stSidebar"] {
        background: #ffffff !important; /* Pure white background for sidebar */
        border-right: 1px solid #e0e0e0; /* Light grey border */
        box-shadow: 5px 0 15px rgba(0,0,0,0.08); /* Soft shadow to differentiate */
        padding-top: 2rem;
        color: #1a1a1a !important; /* Ensure all text within sidebar is dark */
    }
    /* Target the sidebar content wrapper for all text and elements */
    .st-emotion-cache-10o4u2p, .st-emotion-cache-10o4u2p * { /* Select all children too */
        color: #1a1a1a !important; /* Dark text color for all sidebar content */
        background-color: transparent !important; /* Ensure no conflicting backgrounds */
    }

    /* Sidebar header styling */
    .st-emotion-cache-1c7y2kd { /* Targets h2 in sidebar */
        color: #0D47A1 !important; /* Darker blue for sidebar header */
        font-size: 2em; /* Larger sidebar header */
        font-weight: 700;
        margin-bottom: 1.5em;
        text-align: center;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
    }

    /* Sidebar image container */
    .st-emotion-cache-1r6y40z { /* Targets image container in sidebar */
        border-radius: 1.2rem; /* Rounded corners */
        overflow: hidden;
        box-shadow: 0 8px 20px rgba(0,0,0,0.18); /* Soft shadow */
        margin-bottom: 2.5em;
        border: 2px solid #CFD8DC; /* Light grey border */
    }

    /* Sidebar expander styling */
    .st-emotion-cache-p5m6cs { /* Targets expander container */
        background-color: #f8f8f8 !important; /* Slightly off-white for expanders */
        border-radius: 0.8rem;
        border: 1px solid #e0e0e0 !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 1em;
        padding: 0.5em 1em;
    }
    .st-emotion-cache-p5m6cs p { /* Text inside expander */
        color: #1a1a1a !important; /* Ensure expander text is dark */
        font-weight: 500;
    }
    .st-emotion-cache-p5m6cs div[data-testid="stExpanderToggleIcon"] { /* Expander arrow icon */
        color: #0D47A1 !important; /* Ensure expander icon is dark blue */
    }
    .st-emotion-cache-p5m6cs div[data-testid="stExpanderDetails"] { /* Content inside expander details */
        color: #1a1a1a !important; /* Ensure expander details text is dark */
    }
    .st-emotion-cache-p5m6cs button { /* Expander header button */
        color: #1a1a1a !important; /* Ensure expander button text is dark */
        font-weight: 600;
        font-size: 1.1em;
    }

    /* Styling for sidebar input labels */
    .st-emotion-cache-vk330f label {
        color: #1a1a1a !important; /* Force dark color for labels */
        margin-bottom: 0.2em; /* Reduce space below label */
        padding-top: 1em; /* Add padding above label for spacing */
    }
    /* Styling for sidebar input elements (sliders, selectboxes, radio buttons) */
    .st-emotion-cache-1kyx5e9, .st-emotion-cache-1v0mbdj, .st-emotion-cache-1q1n004 {
        background-color: #f0f0f0 !important; /* Light grey background for inputs */
        border-radius: 0.8rem;
        padding: 0.7em;
        border: 1px solid #d0d0d0 !important;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.05); /* Inner shadow for depth */
        color: #1a1a1a !important; /* Ensure input text is dark */
        margin-bottom: 1em; /* Add space below input field */
    }
    /* For slider numbers */
    .st-emotion-cache-1q1n004 div[data-testid="stSlider"] div[data-baseweb="slider"] div[data-testid="stTickBar"] div {
        color: #1a1a1a !important; /* Make slider tick numbers visible */
    }
    .st-emotion-cache-1q1n004 div[data-testid="stSlider"] div[data-baseweb="slider"] div[data-testid="stTickBar"] div[data-testid="stTickBarValue"] {
        color: #1a1a1a !important; /* Make slider value numbers visible */
    }

    /* --- Radio Button Icon Specific Styling --- */
    /* This targets the actual radio button circle */
    .st-emotion-cache-10o4u2p .st-emotion-cache-1v0mbdj > label > div:first-child {
        border-color: #0D47A1 !important; /* Dark blue border for the radio button circle */
        background-color: transparent !important; /* Ensure background is transparent */
    }
    /* This targets the inner dot of the selected radio button */
    .st-emotion-cache-10o4u2p .st-emotion-cache-1v0mbdj > label > div:first-child > div {
        background-color: #0D47A1 !important; /* Dark blue color for the selected dot */
    }
    /* This targets the text next to the radio button */
    .st-emotion-cache-10o4u2p .st-emotion-cache-1v0mbdj > label > div:last-child {
        color: #1a1a1a !important; /* Ensure the radio button text is dark */
    }


    /* --- End Sidebar Styling --- */


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
    /* This targets labels outside the sidebar */
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
        color: #333; /* Darker text for readability */
    }

    /* Image styling within the main content */
    .main-image {
        border-radius: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-bottom: 2em;
        border: 1px solid #e2e8f0;
        width: 100%; /* Ensure image takes full width of its container */
        height: auto; /* Maintain aspect ratio */
        display: block; /* Remove extra space below image */
    }

    /* Footer styling */
    .st-emotion-cache-ch5fnp { /* Targets markdown for footer */
        text-align: center;
        color: #444; /* Darker color for visibility */
        font-size: 1em; /* Slightly larger font size */
        font-weight: 500; /* Medium weight */
        margin-top: 3em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main title and subheader
st.markdown('<p class="main-header">❤️ Heart Disease Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Leveraging Machine Learning for Proactive Health Insights</p>', unsafe_allow_html=True)

# Main project image - now wrapped in markdown to apply custom class
st.markdown(
    f'<div class="main-image-container">'
    f'<img src="https://itdesigners.org/wp-content/uploads/2024/02/heart-1024x576.jpg" class="main-image" alt="Predicting Heart Health">'
    f'<p style="text-align: center; color: #777; font-size: 0.9em; margin-top: 0.5em;">Predicting Heart Health</p>'
    f'</div>',
    unsafe_allow_html=True
)

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

# --- Sidebar Content ---
st.sidebar.header('Patient Features')
st.sidebar.image('https://upload.wikimedia.org/wikipedia/ps/1/13/HeartBeat.gif', caption='Heartbeat Monitor', use_container_width=True)

# Dummy Quick Stats section
st.sidebar.markdown("---")
st.sidebar.subheader("Quick Stats (Dummy Data)")
with st.sidebar.expander("Patient Overview"):
    st.write("Total Patients: **1,250**")
    st.write("Avg. Age: **54.5 years**")
    st.write("Gender Ratio (M/F): **65% / 35%**")

with st.sidebar.expander("Risk Factors"):
    st.write("Avg. Cholesterol: **245 mg/dL**")
    st.write("Avg. Blood Pressure: **128 mmHg**")
    st.write("Patients with Angina: **30%**")

st.sidebar.markdown("---") # Separator before actual inputs

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

# Display input widgets in each st.sidebar.container() for better spacing control
for feature, config in features_config.items():
    with st.sidebar.container(): # Each input gets its own container for more vertical space
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
    st.sidebar.markdown("---") # Add a separator after each input for clear distinction

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
st.markdown("Developed by **Dhruv Sharma**")
