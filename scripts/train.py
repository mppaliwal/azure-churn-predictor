import pandas as pd
from src.churn_pipeline import ChurnModelPipeline
import os

# --- Configuration ---
CRM_DATA_PATH = 'data/synthetic_crm_data_5_percent_churn.csv'
TICKETS_DATA_PATH = 'data/synthetic_support_tickets_5_percent_churn.csv'
MODEL_OUTPUT_PATH = 'artifacts/churn_model_pipeline.joblib'

def run_training():
    """
    Executes the model training and saves the pipeline artifact.
    """
    print("Loading training data...")
    crm_df = pd.read_csv(CRM_DATA_PATH)
    tickets_df = pd.read_csv(TICKETS_DATA_PATH)

    # Initialize and train the pipeline
    pipeline = ChurnModelPipeline(model_type='xgb')
    pipeline.train(crm_df, tickets_df)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    
    # Save the trained pipeline
    pipeline.save(MODEL_OUTPUT_PATH)
    print(f"Pipeline saved successfully to {MODEL_OUTPUT_PATH}")

if __name__ == '__main__':
    run_training()
