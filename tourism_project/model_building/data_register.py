from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
repo_id = "shivaniPandey/tourism-purchase-predictor"  # Dataset repo
repo_type = "dataset"

# Fetch HF token from GitHub Secrets during workflow
HF_TOKEN = os.environ.get("HF_TOKEN")

api = HfApi(token=HF_TOKEN)

# ----------------------------------------------------
# Step 1: Create dataset repo if not exists
# ----------------------------------------------------
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset Repository '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Dataset Repository '{repo_id}' not found. Creating...")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,
        token=HF_TOKEN
    )
    print(f"Dataset Repository '{repo_id}' created successfully.")

# Step 2: Upload dataset folder

print("Uploading tourism dataset folder to HuggingFace...")

api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Dataset uploaded successfully!")
