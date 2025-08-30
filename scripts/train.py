# FILE: scripts/train.py

import pandas as pd
import argparse
import os
from io import StringIO
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from src.churn_pipeline import ChurnModelPipeline

def download_blob_to_dataframe(storage_account, container, blob_name):
    """Connects to Azure Blob Storage and downloads a blob into a pandas DataFrame."""
    account_url = f"https://{storage_account}.blob.core.windows.net"
    # DefaultAzureCredential will use your 'az login' credentials
    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(account_url, credential=credential)
    blob_client = blob_service_client.get_blob_client(container=container, blob=blob_name)
    
    print(f"Downloading data from blob: {blob_name}...")
    downloader = blob_client.download_blob(encoding='utf-8')
    blob_content = downloader.readall()
    
    return pd.read_csv(StringIO(blob_content))

def run_training(storage_account, container, crm_blob, tickets_blob, model_output_path):
    """Main function to run the model training pipeline using data from Azure."""
    print("--- Starting Training Process ---")
    
    crm_df = download_blob_to_dataframe(storage_account, container, crm_blob)
    tickets_df = download_blob_to_dataframe(storage_account, container, tickets_blob)

    pipeline = ChurnModelPipeline(model_type='xgb')
    pipeline.train(crm_df, tickets_df)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    pipeline.save(model_output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage-account', type=str, required=True, help='Your Azure Storage account name.')
    parser.add_argument('--container', type=str, required=True, help='Your container name in Blob Storage.')
    parser.add_argument('--crm-blob', type=str, required=True, help='The name of the CRM data blob.')
    parser.add_argument('--tickets-blob', type=str, required=True, help='The name of the tickets data blob.')
    parser.add_argument('--model-output', type=str, required=True, help='Local path to save the trained model artifact.')
    
    args = parser.parse_args()

    run_training(
        storage_account=args.storage_account,
        container=args.container,
        crm_blob=args.crm_blob,
        tickets_blob=args.tickets_blob,
        model_output_path=args.model_output
    )