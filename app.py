# from flask import Flask, request, jsonify
# import pickle
# import pandas as pd
# import numpy as np

# app = Flask(__name__)

# # Load model and scaler
# try:
#     with open('knn_model.pkl', 'rb') as f:
#         knn_model = pickle.load(f)
#     with open('feature_scaler.pkl', 'rb') as f:
#         scaler = pickle.load(f)
    
#     # Get the complete feature list from the scaler
#     if hasattr(scaler, 'feature_names_in_'):
#         all_features = list(scaler.feature_names_in_)
#         print("Scaler expects these features:", all_features)
#     else:
#         # Fallback if feature_names_in_ is not available
#         all_features = [
#             'nr. sessions', 'total km', 'km Z3-4', 'km Z5-T1-T2', 'km sprinting', 'strength training', 
#             'hours alternative', 'perceived exertion', 'perceived trainingSuccess', 'perceived recovery',
#             'perceived recovery.6'
#         ]
#         print("Warning: Using hardcoded feature list. Ensure it matches the scaler's expectations.")

#     # Define the 14 features the model expects (adjust this list based on your model's training data)
#     model_features = [
#         'km sprinting', 'perceived exertion', 'strength training.1', 'perceived trainingSuccess.1',
#         'km Z3-4.2', 'hours alternative.2', 'perceived recovery.2', 'perceived exertion.3',
#         'strength training.3', 'perceived exertion.4', 'perceived trainingSuccess.4', 'total km.5',
#         'perceived exertion.5', 'perceived recovery.6'
#     ]
#     print("Model expects these features:", model_features)

# except Exception as e:
#     print(f"Error loading files: {e}")
#     exit(1)

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
        
#         # Create a DataFrame with all expected features, filling missing ones with 0
#         input_data = {}
#         for feature in all_features:
#             input_data[feature] = data.get(feature, 0)  # Use 0 as default if feature missing
            
#         # Convert to DataFrame and ensure correct order
#         input_df = pd.DataFrame([input_data])[all_features]
        
#         # Scale the data
#         scaled_data = scaler.transform(input_df)
        
#         # Select only the features the model expects
#         scaled_data_model = pd.DataFrame(scaled_data, columns=all_features)[model_features].values
        
#         # Predict
#         prediction = knn_model.predict(scaled_data_model)
#         prediction_proba = knn_model.predict_proba(scaled_data_model).tolist()
        
#         return jsonify({
#             'prediction': int(prediction[0]),
#             'probability': prediction_proba[0]
#         })
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# ✅ Enable CORS for all routes and origins (you can restrict later if needed)
CORS(app)

# Load model and scaler
try:
    with open('knn_model.pkl', 'rb') as f:
        knn_model = pickle.load(f)
    with open('feature_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    if hasattr(scaler, 'feature_names_in_'):
        all_features = list(scaler.feature_names_in_)
        print("Scaler expects these features:", all_features)
    else:
        all_features = [
            'nr. sessions', 'total km', 'km Z3-4', 'km Z5-T1-T2', 'km sprinting', 'strength training',
            'hours alternative', 'perceived exertion', 'perceived trainingSuccess', 'perceived recovery',
            'perceived recovery.6'
        ]
        print("Warning: Using hardcoded feature list. Ensure it matches the scaler's expectations.")

    model_features = [
        'km sprinting', 'perceived exertion', 'strength training.1', 'perceived trainingSuccess.1',
        'km Z3-4.2', 'hours alternative.2', 'perceived recovery.2', 'perceived exertion.3',
        'strength training.3', 'perceived exertion.4', 'perceived trainingSuccess.4', 'total km.5',
        'perceived exertion.5', 'perceived recovery.6'
    ]
    print("Model expects these features:", model_features)

except Exception as e:
    print(f"Error loading files: {e}")
    exit(1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Create DataFrame with all expected features, filling missing with 0
        input_data = {}
        for feature in all_features:
            input_data[feature] = data.get(feature, 0)
        
        input_df = pd.DataFrame([input_data])[all_features]
        
        # Scale the data
        scaled_data = scaler.transform(input_df)
        
        # Select only the model features
        scaled_data_model = pd.DataFrame(scaled_data, columns=all_features)[model_features].values
        
        # Predict
        prediction = knn_model.predict(scaled_data_model)
        prediction_proba = knn_model.predict_proba(scaled_data_model).tolist()

        return jsonify({
            'prediction': int(prediction[0]),
            'probability': prediction_proba[0]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
