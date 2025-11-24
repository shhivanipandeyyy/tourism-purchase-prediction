from huggingface_hub import HfApi

import os


# Make sure the token is set in GitHub Actions secrets or locally
token = os.getenv("HF_TOKEN")


api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path = "tourism_project/deployment"      # local folder containing your files
repo_id     = "shivaniPandey/tourism-purchase-predictor"  # your HF Space repo
repo_type   = "space"                           # dataset, model, or space
path_in_repo = ""                               # root of the Space repo
                          # optional: subfolder path inside the repo
)
