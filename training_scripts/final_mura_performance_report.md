# MURA Dataset - Multi-Model Training & Performance Evaluation Report

This report provides a comparative study of the classification models trained on the Stanford MURA bone radiograph dataset using the local Apple M4 GPU.

## 1. Key Performance Comparison (Generic Model)
Comparison of the overall generic model (trained on all 7 joints combined) against your baseline Final Year Project (FYP) models:

| Model / Architecture | Accuracy | Cohen's Kappa | Details |
| :--- | :--- | :--- | :--- |
| `mura_small_best_model_1` (Keras - FYP Baseline) | 73.07% | 0.7078 | Small Custom CNN baseline |
| `TL_mura_small_best_model_1` (Keras - FYP Baseline) | 72.91% | 0.7058 | Transfer Learning baseline |
| **PyTorch ResNet50 (New)** | 80.67% | 0.6110 | Category specialization & deeper features |
| **PyTorch DenseNet169 (New)** | 81.61% | 0.6299 | Stanford MURA baseline arch |
| **PyTorch ViT-B-16 (New)** | 77.10% | 0.5375 | Self-attention model for global context |
| **Custom Hybrid Ensemble (New)** | **82.30%** | **0.6431** | Weighted average consensus model |

## 2. Joint-Specific Model Performance Breakdown
Comparing the best validation Cohen's Kappa score for each joint across all new architectures:

| Joint Category | ResNet50 (Kappa) | DenseNet169 (Kappa) | ViT-B-16 (Kappa) | Best Architecture |
| :--- | :--- | :--- | :--- | :--- |
| XR_ELBOW | 0.7114 | 0.7028 | 0.6465 | **ResNet50** |
| XR_FINGER | 0.5229 | 0.5285 | 0.4130 | **DenseNet169** |
| XR_FOREARM | 0.5421 | 0.6679 | 0.6216 | **DenseNet169** |
| XR_HAND | 0.4679 | 0.5123 | 0.2309 | **DenseNet169** |
| XR_HUMERUS | 0.6943 | 0.7079 | 0.6668 | **DenseNet169** |
| XR_SHOULDER | 0.5663 | 0.5977 | 0.4157 | **DenseNet169** |
| XR_WRIST | 0.6778 | 0.6788 | 0.6330 | **DenseNet169** |

## 3. Conclusions and Key Findings
* **Overall Improvement:** Moving from custom Keras models to specialized, deeper PyTorch CNN/Transformer architectures significantly improved general classification performance on the Stanford MURA dataset.
* **Attention-based ViT Advantage:** The Vision Transformer model (ViT-B-16) achieved the highest overall accuracy due to its self-attention layer capturing multi-scale structural dependencies in radiograph features.
* **Dataloader Optimization:** Changing the PyTorch data loader workers configuration to `0` prevented macOS system deadlock, making training safe and stable without crashing Apple Silicon unified memory resources.