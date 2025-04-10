from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the CSV file to extract migraine types
csv_path = r"C:\Users\VICTUS\OneDrive\Documents\migraine_data1.csv"  # Replace with your CSV file path
if not os.path.exists(csv_path):
    print(f"Error: CSV file '{csv_path}' not found. Please ensure the file exists.")
    migraine_type_mapping = {}
else:
    try:
        data = pd.read_csv(csv_path)
        if 'Type' in data.columns:
            # Use LabelEncoder to map migraine types to numerical values
            label_encoder = LabelEncoder()
            data['Type_encoded'] = label_encoder.fit_transform(data['Type'])
            
            # Create a mapping from numerical values to migraine types
            migraine_type_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
            print("Migraine types extracted successfully:")
            print(migraine_type_mapping)
        else:
            print("Error: 'Type' column not found in the CSV file.")
            migraine_type_mapping = {}
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        migraine_type_mapping = {}

# Check if the model file exists
model_path = r"C:\Users\VICTUS\Downloads\random_forest_migration_model (9).pkl"
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found. Please ensure the file exists.")
    model = None
else:
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get user inputs
            Age = float(request.form['age'])
            Duration = float(request.form['Duration'])
            Frequency = float(request.form['Frequency'])
            Location = float(request.form['Location'])
            Character = float(request.form['Character'])
            Intensity = float(request.form['Intensity'])

            # Debug: Print input values
            print(f"Input Values - Age: {Age}, Duration: {Duration}, Frequency: {Frequency}, "
                  f"Location: {Location}, Character: {Character}, Intensity: {Intensity}")

            # Create a feature array
            features = np.array([[Age, Duration, Frequency, Location, Character, Intensity]])

            # Debug: Print feature array
            print("Feature Array:", features)

            # Make prediction
            if model:
                prediction = model.predict(features)[0]  # Get the numerical prediction
                print(f"Numerical Prediction: {prediction}")

                # Map the numerical prediction to a migraine type
                predicted_label = migraine_type_mapping.get(prediction, "Unknown Migraine Type")
                print(f"Predicted Label: {predicted_label}")
            else:
                predicted_label = "Error: Model not loaded."

            return render_template('predict.html', prediction=predicted_label)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('predict.html', prediction=f"Error: {str(e)}")
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)