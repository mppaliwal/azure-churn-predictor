# FILE: src/score.py

import os
import joblib
import pandas as pd
import json

def init():
    """
    This function is called when the service is started.
    It loads the model from the file system and makes it global
    so it can be reused for each prediction.
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created by Azure ML.
    # It points to the directory where the model artifact was downloaded.
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "churn_model_pipeline.joblib")
    model = joblib.load(model_path)
    print("Model loaded successfully.")

def run(raw_data):
    """
    This function is called for each incoming API request.
    """
    try:
        data = json.loads(raw_data)
        
        # Convert the incoming JSON to pandas DataFrames
        crm_df = pd.DataFrame(data['crm'])
        tickets_df = pd.DataFrame(data['tickets'])

        # Use the global model object to make a prediction
        predictions, probabilities = model.predict(crm_df, tickets_df)

        # Format the response as a JSON dictionary
        result = {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist()
        }
        return result
        
    except Exception as e:
        error = str(e)
        return {"error": error}