# FILE: scripts/deploy.py

import sys
import os
import subprocess
import argparse

# Add the project's root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_command(command, raise_on_error=True):
    """Runs a command-line command and checks for errors."""
    print(f"Executing command: {' '.join(command)}")
    # Using shell=True for simplicity, safe as commands are constructed internally
    result = subprocess.run(' '.join(command), shell=True, capture_output=True, text=True, check=False)
    
    if result.returncode != 0 and raise_on_error:
        print("--- ERROR ---")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise Exception(f"Command failed with exit code {result.returncode}")
    
    print("--- SUCCESS ---")
    if result.stdout:
        print(result.stdout)
    return result

def run_deployment(subscription_id, resource_group, workspace, model_name, endpoint_name, model_path):
    """
    Deploys the trained churn model by calling the Azure CLI.
    """
    print("--- Starting Manual Deployment Process using Azure CLI ---")

    # Set the default subscription, resource group, and workspace for the CLI
    print("Setting Azure ML CLI defaults...")
    run_command([
        "az", "configure", "--defaults", 
        f"workspace={workspace}", 
        f"group={resource_group}",
        f"subscription={subscription_id}"
    ])

    # 1. Register the Model using the CLI
    print(f"Registering model '{model_name}' from path '{model_path}'...")
    run_command([
        "az", "ml", "model", "create",
        "--name", model_name,
        "--path", model_path,
        "--type", "custom_model"
    ])
    print("✅ Model registered.")

    # 2. Create or Update the Online Endpoint using the CLI
    print(f"Creating or updating online endpoint '{endpoint_name}'...")
    run_command([
        "az", "ml", "online-endpoint", "update",
        "--name", endpoint_name,
        "--file", ".azure/endpoint.yml"
    ])
    print("✅ Endpoint is ready.")

    # 3. Create the Environment as a standalone asset
    print("Creating or updating the environment in Azure ML...")
    run_command([
        "az", "ml", "environment", "create",
        "--file", ".azure/environment.yml"
    ])
    print("✅ Environment created/updated.")
    
    # 4. Delete old deployment to ensure a clean slate
    print("Attempting to delete existing deployment...")
    run_command([
        "az", "ml", "online-deployment", "delete",
        "--name", "blue",
        "--endpoint-name", endpoint_name,
        "--yes"
    ], raise_on_error=False)
    print("✅ Cleanup step complete.")
    
    # --- THIS IS THE FINAL FIX ---
    # 5. Create a fresh deployment using explicit flags to bypass the CLI bug
    print("Creating a new deployment for the endpoint with explicit flags...")
    run_command([
        "az", "ml", "online-deployment", "create",
        "--name", "blue",
        "--endpoint-name", endpoint_name,
        "--model", f"azureml:{model_name}:latest",
        "--environment", f"azureml:churn-prod-env:latest",
        "--code-path", "./src",
        "--scoring-script", "score.py",
        "--instance-type", "Standard_DS2_v2",
        "--instance-count", "1",
        "--all-traffic"
    ])
    print("✅ Deployment created and traffic allocated.")
    # ---------------------------
    
    print("\n--- 🎉 DEPLOYMENT COMPLETE! 🎉 ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subscription-id', type=str, required=True)
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--resource-group', type=str, required=True)
    parser.add_argument('--model-name', type=str, default='churn-predictor')
    parser.add_argument('--endpoint-name', type=str, default='churn-api-endpoint')
    parser.add_argument('--model-path', type=str, default='./artifacts')

    args = parser.parse_args()

    run_deployment(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace=args.workspace,
        model_name=args.model_name,
        endpoint_name=args.endpoint_name,
        model_path=args.model_path
    )