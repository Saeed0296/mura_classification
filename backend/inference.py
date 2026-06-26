import os
import time
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Device Configuration
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Path to checkpoints
CHECKPOINT_DIR = os.getenv("MURA_CHECKPOINT_DIR", "/Users/saeedanwar/Desktop/saeed/project/mura/MURA/checkpoints")

# Normalization stats
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Validation transforms for inference
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# Initialize model instances
def load_resnet50_model(category):
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 1)
    
    checkpoint_name = f"mura_{category.lower()}_best_model.pth"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    
    # Dynamic fallback to generic 'all' model if specific checkpoint is missing
    if not os.path.exists(checkpoint_path) and category.lower() != "all":
        fallback_name = "mura_all_best_model.pth"
        fallback_path = os.path.join(CHECKPOINT_DIR, fallback_name)
        if os.path.exists(fallback_path):
            checkpoint_path = fallback_path
            print(f"[Inference] ResNet50 checkpoint for {category} not found. Falling back to generic model.")
            
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[Inference] Loaded ResNet50 weights from {os.path.basename(checkpoint_path)}")
        except Exception as e:
            print(f"[Inference] Error loading ResNet50 checkpoint: {e}. Using untrained baseline.")
    else:
        print(f"[Inference] ResNet50 checkpoint not found. Using baseline weights.")
        
    model = model.to(DEVICE)
    model.eval()
    return model

def load_densenet169_model(category):
    model = models.densenet169()
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    
    checkpoint_name = f"mura_densenet_{category.lower()}_best_model.pth"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    
    # Dynamic fallback to generic 'all' model if specific checkpoint is missing
    if not os.path.exists(checkpoint_path) and category.lower() != "all":
        fallback_name = "mura_densenet_all_best_model.pth"
        fallback_path = os.path.join(CHECKPOINT_DIR, fallback_name)
        if os.path.exists(fallback_path):
            checkpoint_path = fallback_path
            print(f"[Inference] DenseNet169 checkpoint for {category} not found. Falling back to generic model.")
            
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[Inference] Loaded DenseNet169 weights from {os.path.basename(checkpoint_path)}")
        except Exception as e:
            print(f"[Inference] Error loading DenseNet169 checkpoint: {e}. Using untrained baseline.")
    else:
        print(f"[Inference] DenseNet169 checkpoint not found. Using baseline weights.")
        
    model = model.to(DEVICE)
    model.eval()
    return model

def load_vit_model(category):
    model = models.vit_b_16()
    model.heads.head = nn.Linear(model.heads.head.in_features, 1)
    
    checkpoint_name = f"mura_vit_{category.lower()}_best_model.pth"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    
    # Dynamic fallback to generic 'all' model if specific checkpoint is missing
    if not os.path.exists(checkpoint_path) and category.lower() != "all":
        fallback_name = "mura_vit_all_best_model.pth"
        fallback_path = os.path.join(CHECKPOINT_DIR, fallback_name)
        if os.path.exists(fallback_path):
            checkpoint_path = fallback_path
            print(f"[Inference] ViT checkpoint for {category} not found. Falling back to generic model.")
            
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[Inference] Loaded ViT weights from {os.path.basename(checkpoint_path)}")
        except Exception as e:
            print(f"[Inference] Error loading ViT checkpoint: {e}. Using untrained baseline.")
    else:
        print(f"[Inference] ViT checkpoint not found. Using baseline weights.")
        
    model = model.to(DEVICE)
    model.eval()
    return model

def predict_single_image(image_path, category):
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = val_transforms(image).unsqueeze(0).to(DEVICE)
    except Exception as e:
        return {"error": f"Failed to load or process image: {str(e)}"}
        
    results = {}
    
    # 1. ResNet50 Inference
    try:
        resnet_model = load_resnet50_model(category)
        t0 = time.time()
        with torch.no_grad():
            output = resnet_model(input_tensor)
            prob = torch.sigmoid(output).item()
        results["resnet50"] = {
            "prediction": "Abnormal (Positive)" if prob >= 0.5 else "Normal (Negative)",
            "probability": prob,
            "latency_ms": round((time.time() - t0) * 1000, 2)
        }
    except Exception as e:
        results["resnet50"] = {"error": str(e)}
        
    # 2. DenseNet169 Inference
    try:
        densenet_model = load_densenet169_model(category)
        t0 = time.time()
        with torch.no_grad():
            output = densenet_model(input_tensor)
            prob = torch.sigmoid(output).item()
        results["densenet169"] = {
            "prediction": "Abnormal (Positive)" if prob >= 0.5 else "Normal (Negative)",
            "probability": prob,
            "latency_ms": round((time.time() - t0) * 1000, 2)
        }
    except Exception as e:
        results["densenet169"] = {"error": str(e)}
        
    # 3. ViT-B-16 Inference
    try:
        vit_model = load_vit_model(category)
        t0 = time.time()
        with torch.no_grad():
            output = vit_model(input_tensor)
            prob = torch.sigmoid(output).item()
        results["vit_b_16"] = {
            "prediction": "Abnormal (Positive)" if prob >= 0.5 else "Normal (Negative)",
            "probability": prob,
            "latency_ms": round((time.time() - t0) * 1000, 2)
        }
    except Exception as e:
        results["vit_b_16"] = {"error": str(e)}
        
    # 4. Hybrid SOTA Ensemble Inference
    try:
        valid_probs = []
        valid_latencies = []
        weights = []
        
        if "resnet50" in results and "error" not in results["resnet50"]:
            valid_probs.append(results["resnet50"]["probability"])
            valid_latencies.append(results["resnet50"]["latency_ms"])
            weights.append(0.15)
            
        if "densenet169" in results and "error" not in results["densenet169"]:
            valid_probs.append(results["densenet169"]["probability"])
            valid_latencies.append(results["densenet169"]["latency_ms"])
            weights.append(0.35)
            
        if "vit_b_16" in results and "error" not in results["vit_b_16"]:
            valid_probs.append(results["vit_b_16"]["probability"])
            valid_latencies.append(results["vit_b_16"]["latency_ms"])
            weights.append(0.50)
            
        if len(valid_probs) > 0:
            total_w = sum(weights)
            norm_weights = [w / total_w for w in weights]
            hybrid_prob = sum(p * w for p, w in zip(valid_probs, norm_weights))
            hybrid_latency = sum(valid_latencies)
            
            results["hybrid_ensemble"] = {
                "prediction": "Abnormal (Positive)" if hybrid_prob >= 0.5 else "Normal (Negative)",
                "probability": hybrid_prob,
                "latency_ms": round(hybrid_latency, 2)
            }
        else:
            results["hybrid_ensemble"] = {"error": "All underlying models failed."}
    except Exception as e:
        results["hybrid_ensemble"] = {"error": f"Failed to compute ensemble: {str(e)}"}
        
    return results
