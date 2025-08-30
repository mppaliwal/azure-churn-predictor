# FILE: scripts/deploy.py

import argparse
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model, ManagedOnlineEndpoint, ManagedOnlineDeployment, CodeConfiguration, Environment
from azure.identity import DefaultAzureCredential

def run_deployment(workspace, resource_group, model_name, endpoint_name, model_path):
    """
    Connects to Azure ML and deploys the trained churn model.
    """
    print("--- Starting Manual Deployment Process ---")

    # Use DefaultAzureCredential which automatically uses your 'az login' credentials
    credential = DefaultAzureCredential()

    # Create a client to connect to your Azure ML workspace
    ml_client = MLClient(
        credential=credential,
        workspace_name=workspace,
        resource_group_name=resource_group,
        subscription_id=None # Reads subscription from your az login context
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
    # Define the environment from your requirements file
    env = Environment(
        name="churn-prod-env",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file=None,
        pip_requirements_file="requirements.txt"
    )

    blue_deployment = ManagedOnlineDeployment(
        name="blue", # Name of the deployment, e.g., for blue-green deployment
        endpoint_name=endpoint_name,
        model=registered_model,
        code_configuration=CodeConfiguration(
            code="./src", # The folder containing your score.py
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
    parser.add_...