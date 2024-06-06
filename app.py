import streamlit as st
import pandas as pd
import pickle

# Load the model and preprocessor
with open('random_forest_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# with open('preprocessor.pkl', 'rb') as f:
    # loaded_preprocessor = pickle.load(f)

# Set page title and icon
st.set_page_config(page_title="Medical insurance Charges Prediction App", page_icon="💵")

# Title of the web app
st.title("Medical Charges Prediction App")
st.markdown("""
    <style>
    .main {
        background-color: #f0f0f5;
    }
    </style>
    """, unsafe_allow_html=True)

# Add a sidebar for better organization
st.sidebar.header("User Input Parameters")

# Input fields in the sidebar
age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=25)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=100.0, value=30.5)
children = st.sidebar.number_input("Number of children", min_value=0, max_value=10, value=2)
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])

# Create a dictionary with the input data
input_data = {
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker]
}

# Convert input data to DataFrame
input_df = pd.DataFrame(input_data)

# Display user inputs in the main page
st.subheader("Input Data")
st.write(input_df)

# Button for prediction
if st.button("Predict"):
    # Make prediction using the loaded model
    prediction = loaded_model.predict(input_df)
    st.subheader("Prediction Result")
    st.success(f"The predicted medical insurance charges are: ${prediction[0]:.2f}")
    st.balloons()

# Footer
st.markdown("""
    ---
    **Note**: This prediction is based on a machine learning model trained on online data and may not be accurate for every individual. For precise assessments tailored to your specific context, please consult a professional and consider building a similar app using your local data.
""")
