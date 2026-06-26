import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, cohen_kappa_score

# Device Configuration
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

BASE_DIR = "/Users/saeedanwar/Desktop/saeed/MURA"
DATASET_PATH = os.path.join(BASE_DIR, "MURA-v1.1")
VALID_CSV = os.path.join(DATASET_PATH, "valid_image_paths.csv")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

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
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        label = self.df.iloc[idx]['label']
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

def get_label(path):
    return 1 if 'positive' in path.lower() else 0

# Load models
def load_resnet50():
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 1)
    cp = torch.load(os.path.join(CHECKPOINT_DIR, "mura_all_best_model.pth"), map_location=DEVICE, weights_only=False)
    model.load_state_dict(cp['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    return model

def load_densenet169():
    model = models.densenet169()
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    cp = torch.load(os.path.join(CHECKPOINT_DIR, "mura_densenet_all_best_model.pth"), map_location=DEVICE, weights_only=False)
    model.load_state_dict(cp['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    return model

def load_vit():
    model = models.vit_b_16()
    model.heads.head = nn.Linear(model.heads.head.in_features, 1)
    cp = torch.load(os.path.join(CHECKPOINT_DIR, "mura_vit_all_best_model.pth"), map_location=DEVICE, weights_only=False)
    model.load_state_dict(cp['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    return model

def main():
    print(f"Device Selected: {DEVICE}")
    df_valid = pd.read_csv(VALID_CSV, header=None, names=['path'])
    df_valid['label'] = df_valid['path'].apply(get_label)
    
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    val_dataset = MURADataset(df_valid, base_dir=BASE_DIR, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print("Loading models...")
    resnet = load_resnet50()
    densenet = load_densenet169()
    vit = load_vit()
    
    resnet_probs = []
    densenet_probs = []
    vit_probs = []
    true_labels = []
    
    print(f"Evaluating {len(df_valid)} validation images in batches...")
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(DEVICE)
            
            # Predict
            res_out = torch.sigmoid(resnet(inputs)).cpu().numpy().flatten()
            den_out = torch.sigmoid(densenet(inputs)).cpu().numpy().flatten()
            vit_out = torch.sigmoid(vit(inputs)).cpu().numpy().flatten()
            
            resnet_probs.extend(res_out)
            densenet_probs.extend(den_out)
            vit_probs.extend(vit_out)
            true_labels.extend(labels.numpy())
            
            if (i+1) % 20 == 0:
                print(f"Processed batch {i+1}/{len(val_loader)}")

    resnet_probs = np.array(resnet_probs)
    densenet_probs = np.array(densenet_probs)
    vit_probs = np.array(vit_probs)
    true_labels = np.array(true_labels)
    
    # Calculate single model metrics
    def get_metrics(probs, labels):
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(labels, preds)
        kappa = cohen_kappa_score(labels, preds)
        return acc, kappa
        
    res_acc, res_kappa = get_metrics(resnet_probs, true_labels)
    den_acc, den_kappa = get_metrics(densenet_probs, true_labels)
    vit_acc, vit_kappa = get_metrics(vit_probs, true_labels)
    
    # Hybrid consensus
    hybrid_probs = 0.15 * resnet_probs + 0.35 * densenet_probs + 0.50 * vit_probs
    hybrid_acc, hybrid_kappa = get_metrics(hybrid_probs, true_labels)
    
    print("\n==========================================")
    print("EVALUATION RESULTS (ALL GENERIC MODEL)")
    print("==========================================")
    print(f"ResNet50:      Accuracy = {res_acc*100:.2f}% | Cohen's Kappa = {res_kappa:.4f}")
    print(f"DenseNet169:   Accuracy = {den_acc*100:.2f}% | Cohen's Kappa = {den_kappa:.4f}")
    print(f"ViT-B-16:      Accuracy = {vit_acc*100:.2f}% | Cohen's Kappa = {vit_kappa:.4f}")
    print(f"Hybrid Ensemble: Accuracy = {hybrid_acc*100:.2f}% | Cohen's Kappa = {hybrid_kappa:.4f}")
    print("==========================================")

if __name__ == "__main__":
    main()
