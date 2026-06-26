import os
import concurrent.futures
from tqdm import tqdm

src_dir = "/Users/saeedanwar/Library/CloudStorage/GoogleDrive-saeedanwar166167@gmail.com/My Drive/Extra_work/MURA/MURA-v1.1"

# Gather all file paths
all_files = []
for root, dirs, files in os.walk(src_dir):
    for f in files:
        if not f.startswith("."):
            all_files.append(os.path.join(root, f))

total_files = len(all_files)
print(f"Total files to cache: {total_files}")

def download_file(file_path):
    try:
        # Opening and reading a small chunk forces the OS FileProvider to download the file in the background
        with open(file_path, "rb") as f:
            f.read(1024)
    except Exception:
        pass

# Use 64 parallel threads to download multiple files at the same time
print("Starting parallel download/caching from Google Drive...")
with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
    # Use tqdm to show a nice text progress bar in the logs
    for _ in tqdm(executor.map(download_file, all_files), total=total_files):
        pass

print("All MURA files cached locally on your Mac!")
