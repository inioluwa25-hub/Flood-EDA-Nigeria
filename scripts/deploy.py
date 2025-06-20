import boto3
import os
from pathlib import Path


def deploy_models():
    s3 = boto3.client("s3")
    bucket = "nigeria-flood-model"

    # Get project root directory
    project_root = Path(__file__).resolve().parent.parent

    # Upload models
    models_dir = project_root / "models"
    for model_file in os.listdir(models_dir):
        if model_file.endswith(".h5") or model_file.endswith(".pkl"):
            s3.upload_file(str(models_dir / model_file), bucket, f"models/{model_file}")
            print(f"✅ Uploaded {model_file} to S3")

    # Upload notebooks
    notebook_path = project_root / "notebooks" / "dl_models.ipynb"
    s3.upload_file(str(notebook_path), bucket, "notebooks/dl_models.ipynb")

    print("🚀 All deployment assets uploaded to AWS S3")


if __name__ == "__main__":
    deploy_models()
