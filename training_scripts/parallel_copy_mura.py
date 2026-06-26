import os
import shutil
import concurrent.futures
from tqdm import tqdm

src_dir = "/Users/saeedanwar/Library/CloudStorage/GoogleDrive-saeedanwar166167@gmail.com/My Drive/Extra_work/MURA"
dst_dir = "/Users/saeedanwar/Desktop/saeed/MURA"

# Gather all files recursively, including the dataset and the trained .h5 models
all_files = []
for root, dirs, files in os.walk(src_dir):
    for f in files:
        if not f.startswith("."):
            src_file = os.path.join(root, f)
            rel_path = os.path.relpath(src_file, src_dir)
            dst_file = os.path.join(dst_dir, rel_path)
            all_files.append((src_file, dst_file))

total_files = len(all_files)
print(f"Total files to sync locally: {total_files}")

def copy_file(paths):
    src, dst = paths
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    except Exception as e:
        print(f"Error copying {src} -> {dst}: {e}")

# Use 64 parallel threads to download and copy everything in parallel
print("Starting parallel local sync from Google Drive...")
with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
    for _ in tqdm(executor.map(copy_file, all_files), total=total_files):
        pass

print("Parallel sync of dataset and models completed successfully!")
