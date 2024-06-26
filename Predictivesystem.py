import pandas as pd
import pickle

# Load the model and preprocessor
with open('trained_model.sav', 'rb') as f:
    loaded_model = pickle.load(f)


# Define input data for prediction
input_data = {
    'age': [25],
    'sex': ['female'],
    'bmi': [30.5],
    'children': [2],
    'smoker': ['yes']
}

# Convert input data to DataFrame
input_df = pd.DataFrame(input_data)

# Make prediction using the loaded model
prediction = loaded_model.predict(input_df)

print("Predicted charges:", prediction[0])
