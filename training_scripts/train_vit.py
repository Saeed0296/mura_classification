import os
import gc
import copy
import time
import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import cohen_kappa_score, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = "/Users/saeedanwar/Desktop/saeed/MURA"
DATASET_PATH = os.path.join(BASE_DIR, "MURA-v1.1")
TRAIN_CSV = os.path.join(DATASET_PATH, "train_image_paths.csv")
VALID_CSV = os.path.join(DATASET_PATH, "valid_image_paths.csv")

MODEL_ARCH = "vit_b_16"
BATCH_SIZE = 16            # Safe batch size of 16 to avoid VRAM / unified memory spikes on ViT attention blocks
EPOCHS = 5                 # 5 epochs per model
LEARNING_RATE = 5e-5       # Slower learning rate for transformer fine-tuning
NUM_WORKERS = 0            # Set to 0 on macOS to prevent process deadlocks
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Device Configuration
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Device Selected: {DEVICE}")
print(f"Model Architecture: {MODEL_ARCH}")
print(f"Batch Size: {BATCH_SIZE} (Safe transformer VRAM limit)")
print(f"Dataset Path: {DATASET_PATH}")
print(f"Checkpoints Directory: {CHECKPOINT_DIR}\n")

# ==========================================
# DATASET DEFINITION
# ==========================================
class MURADataset(Dataset):
    def __init__(self, df, base_dir, transform=None):
        self.df = df
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.df.iloc[idx]['path']
        img_path = os.path.join(self.base_dir, rel_path)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"\n[Warning] Corrupted image skipped: {img_path}. Error: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        label = self.df.iloc[idx]['label']
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

# Normalization stats (ViT uses ImageNet mean and std as well)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# Load master CSV dataframes
df_train_master = pd.read_csv(TRAIN_CSV, header=None, names=['path'])
df_valid_master = pd.read_csv(VALID_CSV, header=None, names=['path'])

def get_label(path):
    return 1 if 'positive' in path.lower() else 0

def get_joint_type(path):
    parts = path.split('/')
    return parts[2] if len(parts) > 2 else 'UNKNOWN'

df_train_master['label'] = df_train_master['path'].apply(get_label)
df_train_master['joint'] = df_train_master['path'].apply(get_joint_type)

df_valid_master['label'] = df_valid_master['path'].apply(get_label)
df_valid_master['joint'] = df_valid_master['path'].apply(get_joint_type)

# Define categories to train in series
JOINT_CATEGORIES = [
    "XR_ELBOW",
    "XR_FINGER",
    "XR_FOREARM",
    "XR_HAND",
    "XR_HUMERUS",
    "XR_SHOULDER",
    "XR_WRIST",
    "ALL"
]

overall_stats = {}

# ==========================================
# TRAINING PIPELINE
# ==========================================
def run_training(category):
    print(f"\n==========================================")
    print(f"STARTING ViT-B-16 TRAINING: {category}")
    print(f"==========================================")
    
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()

    # Filter dataframes
    if category == "ALL":
        train_df = df_train_master.copy()
        valid_df = df_valid_master.copy()
    else:
        train_df = df_train_master[df_train_master['joint'] == category].reset_index(drop=True)
        valid_df = df_valid_master[df_valid_master['joint'] == category].reset_index(drop=True)

    print(f"Training images: {len(train_df)} | Validation images: {len(valid_df)}")
    
    train_dataset = MURADataset(train_df, base_dir=BASE_DIR, transform=train_transforms)
    val_dataset = MURADataset(valid_df, base_dir=BASE_DIR, transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Initialize model
    model = models.vit_b_16(pretrained=True)
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, 1)

    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    best_kappa = -1.0
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    model_save_path = os.path.join(CHECKPOINT_DIR, f"mura_vit_{category.lower()}_best_model.pth")
    
    history = {
        "train_loss": [], "train_acc": [], "train_kappa": [],
        "val_loss": [], "val_acc": [], "val_kappa": []
    }

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        
        # 1. Training Epoch
        model.train()
        running_loss = 0.0
        train_labels = []
        train_preds = []
        
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).cpu().numpy()
            
            train_labels.extend(labels.cpu().numpy())
            train_preds.extend(preds)

        train_loss = running_loss / len(train_dataset)
        train_acc = accuracy_score(train_labels, train_preds)
        train_kappa = cohen_kappa_score(train_labels, train_preds)

        # 2. Validation Epoch
        model.eval()
        val_running_loss = 0.0
        val_labels = []
        val_preds = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) >= 0.5).cpu().numpy()
                
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds)

        val_loss = val_running_loss / len(val_dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        val_kappa = cohen_kappa_score(val_labels, val_preds)
        
        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_kappa"].append(train_kappa)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_kappa"].append(val_kappa)
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - Kappa: {train_kappa:.4f} | "
              f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.4f} - Kappa: {val_kappa:.4f} | "
              f"Time: {epoch_time:.1f}s")
              
        # Checkpoint Saving
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({
                'model_state_dict': best_model_wts,
                'val_kappa': best_kappa,
                'val_acc': best_acc,
                'history': history
            }, model_save_path)
            print(f"--> Saved new best ViT checkpoint (Kappa: {best_kappa:.4f})")

    print(f"ViT Training for {category} completed. Best Val Kappa: {best_kappa:.4f} | Best Val Acc: {best_acc:.4f}")
    
    overall_stats[category] = {
        "dataset_size": {"train": len(train_df), "val": len(valid_df)},
        "best_val_kappa": best_kappa,
        "best_val_accuracy": best_acc,
        "history": history
    }

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    start_all = time.time()
    
    for category in JOINT_CATEGORIES:
        try:
            run_training(category)
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Failed to train ViT model for {category}. Skipping to next category.")
            print(f"Error logs: {e}\n")
            overall_stats[category] = {
                "status": "failed",
                "error": str(e)
            }
            
    stats_save_path = os.path.join(CHECKPOINT_DIR, "vit_training_statistics.json")
    with open(stats_save_path, "w", encoding="utf-8") as f:
        json.dump(overall_stats, f, indent=4)
        
    total_duration = time.time() - start_all
    print(f"\n==========================================")
    print(f"ALL ViT-B-16 JOINT MODELS TRAINED IN SERIES SUCCESSFULLY!")
    print(f"Total time elapsed: {total_duration / 3600:.2f} hours")
    print(f"Summary JSON saved to: {stats_save_path}")
    print(f"==========================================")
