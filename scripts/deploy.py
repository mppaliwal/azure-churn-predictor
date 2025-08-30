# FILE: scripts/deploy.py

import sys
import os
# --- THIS IS THE FIX ---
# Add the project's root directory to the Python path
# This allows the script to find the 'src' module if needed in the future
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --------------------

import argparse
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model, ManagedOnlineEndpoint, ManagedOnlineDeployment, CodeConfiguration, Environment
from azure.identity import DefaultAzureCredential

def run_deployment(workspace, resource_group, model_name, endpoint_name, model_path):
    """
    Connects to Azure ML and deploys the trained churn model.
    """
    print("--- Starting Manual Deployment Process ---")
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        workspace_name=workspace,
        resource_group_name=resource_group,
        subscription_id=None
    )

    # 1. Register the Model
    print(f"Registering model '{model_name}' from path '{model_path}'...")
    model = Model(
        name=model_name,
        path=model_path,
        type="custom_model",
        description="Churn prediction model pipeline."
    )
    registered_model = ml_client.models.create_or_update(model)
    print(f"✅ Model registered. Version: {registered_model.version}")

    # 2. Create or Update the Online Endpoint
    print(f"Creating or updating online endpoint '{endpoint_name}'...")
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Online endpoint for churn prediction.",
        auth_mode="key"
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).wait()
    print("✅ Endpoint is ready.")

    # 3. Create the Deployment
    print("Creating a new deployment for the endpoint...")
    env = Environment(
        name="churn-prod-env",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file=None,
        pip_requirements_file="requirements.txt"
    )

    blue_deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=endpoint_name,
        model=registered_model,
        code_configuration=CodeConfiguration(
            code="./src",
            scoring_script="score.py"
        ),
        environment=env,
        instance_type="Standard_DS2_v2",
        instance_count=1
    )
    ml_client.online_deployments.begin_create_or_update(blue_deployment).wait()
    print("✅ Deployment created.")

    # 4. Allocate Traffic to the New Deployment
    print("Allocating 100% of traffic to the new 'blue' deployment...")
    endpoint.traffic = {"blue": 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).wait()
    print("✅ Traffic allocated successfully.")
    print("\n--- Deployment Complete! ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, required=True, help='Your Azure ML workspace name.')
    parser.add_argument('--resource-group', type=str, required=True, help='Your Azure resource group name.')
    parser.add_argument('--model-name', type=str, default='churn-predictor', help='A name for the registered model.')
    parser.add_argument('--endpoint-name', type=str, default='churn-api-endpoint', help='The name of the deployment endpoint.')
    parser.add_argument('--model-path', type=str, default='./artifacts', help='The local directory containing your trained model artifact.')

    args = parser.parse_args()

    run_deployment(
        workspace=args.workspace,
        resource_group=args.resource_group,
        model_name=args.model_name,
        endpoint_name=args.endpoint_name,
        model_path=args.model_path
    )