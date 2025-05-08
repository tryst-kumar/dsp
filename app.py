import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import json # Added for loading scaler config

# Add this to control Streamlit's caching behavior which can sometimes cause issues with repeated value changes
st.cache_data.clear()

# Load the ML model
model = None
scaler_config = None
expected_features_order = ['id', 'Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
scaled_features_list = []

try:
    # Try loading the model without its compilation state
    model = load_model('calories_model.h5', compile=False)
    st.success("Model loaded successfully!")
    try:
        with open('scaler_config.json', 'r') as f:
            scaler_config = json.load(f)
        scaled_features_list = scaler_config.get('columns', [])
        st.success("Scaler configuration loaded successfully!")
    except FileNotFoundError:
        st.error("Error: scaler_config.json not found. Please ensure it's in the same directory.")
        scaler_config = None # Ensure app knows config is missing
    except json.JSONDecodeError:
        st.error("Error: Could not decode scaler_config.json. File might be corrupted.")
        scaler_config = None # Ensure app knows config is missing
    except Exception as e:
        st.error(f"Error loading scaler_config.json: {e}")
        scaler_config = None

except Exception as e:
    st.error(f"Error loading the Keras model: {e}")
    st.warning("Please ensure 'calories_model.h5' is a valid Keras model file and there are no TensorFlow/Keras version mismatches. Trying to load with 'compile=False' might help if the issue is optimizer-related.")

# Streamlit app
st.title("Calories Prediction")

if model is not None and scaler_config is not None:
    # Input fields - Use session state to maintain values across reruns
    if 'age' not in st.session_state:
        st.session_state.age = 25
    if 'weight' not in st.session_state:
        st.session_state.weight = 70.0
    if 'height' not in st.session_state:
        st.session_state.height = 170.0
    if 'gender' not in st.session_state:
        st.session_state.gender = "Male"
    if 'duration' not in st.session_state:
        st.session_state.duration = 30.0
    if 'heart_rate' not in st.session_state:
        st.session_state.heart_rate = 100.0
    if 'body_temp' not in st.session_state:
        st.session_state.body_temp = 37.0

    # Input fields with session state to maintain values
    age = st.number_input("Age:", min_value=0, max_value=120, value=st.session_state.age, step=1, key='age_input', format="%d")
    weight = st.number_input("Weight (kg):", min_value=0.0, max_value=300.0, value=st.session_state.weight, step=0.1, key='weight_input', format="%.1f")
    height = st.number_input("Height (cm):", min_value=0.0, max_value=250.0, value=st.session_state.height, step=0.1, key='height_input', format="%.1f")
    gender = st.radio("Gender:", options=["Male", "Female"], index=0 if st.session_state.gender == "Male" else 1, key='gender_input')
    duration = st.number_input("Duration (minutes):", min_value=0.0, value=st.session_state.duration, step=0.1, key='duration_input', format="%.1f")
    heart_rate = st.number_input("Heart Rate (bpm):", min_value=0.0, value=st.session_state.heart_rate, step=0.1, key='heart_rate_input', format="%.1f")
    body_temp = st.number_input("Body Temperature (°C):", min_value=30.0, max_value=45.0, value=st.session_state.body_temp, step=0.1, key='body_temp_input', format="%.1f")

    # Update session state
    st.session_state.age = age
    st.session_state.weight = weight
    st.session_state.height = height
    st.session_state.gender = gender
    st.session_state.duration = duration
    st.session_state.heart_rate = heart_rate
    st.session_state.body_temp = body_temp

    # Predict button
    if st.button("Predict"):
        try:
            # Encode gender (Male: 1, Female: 0 - matching notebook)
            gender_encoded = 1 if gender == "Male" else 0
            
            # Placeholder for 'id' feature (as it was in X_train columns)
            id_feature = 0 

            # Create a dictionary for easy access and ordering
            raw_features = {
                'id': id_feature,
                'Sex': gender_encoded,
                'Age': age,
                'Height': height,
                'Weight': weight,
                'Duration': duration,
                'Heart_Rate': heart_rate,
                'Body_Temp': body_temp
            }

            # Prepare input for the model - scale features
            input_features_scaled = []
            for feature_name in expected_features_order:
                value = raw_features[feature_name]
                if feature_name in scaled_features_list:
                    # Apply scaling: (value - mean) / scale
                    try:
                        idx = scaler_config['columns'].index(feature_name)
                        mean = scaler_config['mean'][idx]
                        scale = scaler_config['scale'][idx]
                        if scale == 0: # Avoid division by zero
                            st.warning(f"Scale for feature '{feature_name}' is zero. Using unscaled value.")
                            input_features_scaled.append(value)
                        else:
                            input_features_scaled.append((value - mean) / scale)
                    except (ValueError, IndexError) as e:
                        st.error(f"Error applying scaling for feature '{feature_name}': {e}. Using raw value.")
                        input_features_scaled.append(value) # Fallback to raw value
                else:
                    # Feature is not in the scaled list (e.g., 'id', 'Sex')
                    input_features_scaled.append(value)
            
            # Debug information - sometimes helps with troubleshooting
            st.write("Debug - Scaled features:", input_features_scaled)
            
            if len(input_features_scaled) != 8:
                st.error(f"Internal error: Prepared feature vector has {len(input_features_scaled)} elements, expected 8.")
            else:
                input_data = np.array([input_features_scaled], dtype=np.float32) # Ensure float32 for TF
                with st.spinner("Calculating..."):
                    prediction = model.predict(input_data, verbose=0)  # Set verbose=0 to suppress TF output
                st.success(f"Predicted Calories: {prediction[0][0]:.2f}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
else:
    st.info("Model or scaler configuration not loaded. Please check the errors above.")
