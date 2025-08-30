from flask import Flask, request, jsonify
import pandas as pd
import joblib
from src.churn_pipeline import ChurnModelPipeline

# Initialize the Flask application
app = Flask(__name__)

# --- MODEL LOADING ---
# Load the trained pipeline object once when the application starts
MODEL_PATH = 'artifacts/churn_model_pipeline.joblib'
pipeline = ChurnModelPipeline.load(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives customer data in JSON format, makes a prediction,
    and returns the result.
    """
    try:
        # Get data from the POST request
        json_data = request.get_json()
        
        # Convert JSON to pandas DataFrames
        crm_df = pd.DataFrame(json_data['crm'])
        tickets_df = pd.DataFrame(json_data['tickets'])

        # Use the pipeline to make a prediction
        predictions, probabilities = pipeline.predict(crm_df, tickets_df)

        # Prepare the response
        result = {
            'churn_prediction': predictions.tolist(),
            'churn_probability': probabilities.tolist()
        }
        
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # This allows you to run the app locally for testing
    app.run(host='0.0.0.0', port=5000, debug=True)
