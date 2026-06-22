import os
import sys
import shutil
import argparse
from huggingface_hub import HfApi

def deploy():
    parser = argparse.ArgumentParser(description="Deploy MURA Bone Classification App to Hugging Face Spaces")
    parser.add_argument("--username", type=str, default="Saeed0296", help="Your Hugging Face username")
    parser.add_argument("--token", type=str, help="Your Hugging Face Access Token with WRITE permissions")
    args = parser.parse_args()

    username = args.username
    token = args.token

    if not token:
        print("Error: Hugging Face Access Token is required. Use --token flag.")
        sys.exit(1)

    # Paths configuration
    base_dir = "/Users/saeedanwar/Desktop/saeed"
    project_dir = os.path.join(base_dir, "mura_classification")
    weights_dir = os.path.join(base_dir, "MURA", "checkpoints")
    staging_dir = os.path.join(project_dir, "temp_hf_deploy")

    print(f"Staging files in: {staging_dir}")

    # Remove staging if exists
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir)

    # 1. Copy Dockerfile
    shutil.copy2(os.path.join(project_dir, "Dockerfile"), os.path.join(staging_dir, "Dockerfile"))

    # 2. Copy requirements_hf.txt
    shutil.copy2(os.path.join(project_dir, "requirements_hf.txt"), os.path.join(staging_dir, "requirements_hf.txt"))

    # 3. Copy README_hf.md as README.md (HF Space config)
    shutil.copy2(os.path.join(project_dir, "README_hf.md"), os.path.join(staging_dir, "README.md"))

    # 4. Copy backend and static directories
    shutil.copytree(os.path.join(project_dir, "backend"), os.path.join(staging_dir, "backend"))
    shutil.copytree(os.path.join(project_dir, "static"), os.path.join(staging_dir, "static"))

    # 5. Copy weights (only the 3 generic 'all' checkpoints to fit under the 1 GB HF Space free tier limit)
    target_weights_dir = os.path.join(staging_dir, "MURA", "checkpoints")
    os.makedirs(target_weights_dir)
    print(f"Copying generic 'all' model weights from {weights_dir} to staging (to fit within 1 GB limit)...")
    
    files_to_copy = [
        "mura_all_best_model.pth",
        "mura_densenet_all_best_model.pth",
        "mura_vit_all_best_model.pth",
        "overall_training_statistics.json",
        "densenet_training_statistics.json",
        "vit_training_statistics.json"
    ]
    for file_name in files_to_copy:
        src_path = os.path.join(weights_dir, file_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(target_weights_dir, file_name))
        else:
            print(f"Warning: {file_name} not found in {weights_dir}")

    repo_id = f"{username}/mura-classification"
    print(f"Connecting to Hugging Face to check/create repository: {repo_id}...")
    api = HfApi()

    try:
        # Create Space repo if not exists
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            exist_ok=True,
            token=token
        )
        print(f"Repository {repo_id} ready on Hugging Face Spaces.")
    except Exception as e:
        print(f"Error checking/creating repository: {e}")
        # Clean up staging directory before exiting
        shutil.rmtree(staging_dir)
        sys.exit(1)

    print("Uploading folder contents to Hugging Face Spaces (including weights)... This will take a few minutes depending on your network upload speed.")
    try:
        # Upload folder
        api.upload_folder(
            folder_path=staging_dir,
            repo_id=repo_id,
            repo_type="space",
            token=token
        )
        print("Upload completed successfully!")
        print(f"Your app is building. You can view the status and access the dashboard at:")
        print(f"https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"Upload failed: {e}")
    finally:
        # Clean up staging directory to save local space
        print("Cleaning up local staging files...")
        shutil.rmtree(staging_dir)

if __name__ == "__main__":
    deploy()
