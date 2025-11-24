from huggingface_hub import HfApi
import os

# Use environment variable for HF token
import os
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id="shivaniPandey/tourism-purchase-predictor",
    repo_type="space",
    path_in_repo=""
)


api = HfApi(token=os.getenv("HF_TOKEN"))

# Upload deployment folder to HF Space
api.upload_folder(
    folder_path="tourism_project/deployment",    # local folder containing your files
    repo_id="shivaniPandey/tourism-purchase-predictor",  # your HF Space repo
    repo_type="space",                            # dataset, model, or space
    path_in_repo=""                               # root of the Space repo
)
