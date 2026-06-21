# MURA Bone Classification Dashboard

A premium, interactive web application for real-time bone radiograph classification and abnormality detection. This system utilizes deep learning models trained on the Stanford MURA (musculoskeletal radiographs) dataset to classify X-ray images as **Normal** or **Abnormal** across 7 joint types.

The application evaluates and compares three distinct architectures in real-time on local Apple Silicon (M4 GPU) hardware:
1. **ResNet50**
2. **DenseNet169**
3. **Vision Transformer (ViT-B-16)**

---

## 🚀 Features

- **Multi-Model Inference:** Upload a radiograph and receive predictions from ResNet50, DenseNet169, and ViT-B-16 simultaneously.
- **Glassmorphism Dark Mode UI:** A gorgeous, responsive dashboard designed with a modern dark slate palette, neon accents, and interactive circular confidence gauges.
- **Joint-Specific Routing:** Select the specific joint category (`Elbow`, `Finger`, `Forearm`, `Hand`, `Humerus`, `Shoulder`, `Wrist`) or run a `Generic (All)` model.
- **Live Elapsed Timer & Processing Animation:** Visual feedback displaying current stage operations and inference times on Apple M4 GPU.
- **Model Consensus Diagnostic:** Explains the agreement between the three models and provides a detailed recommendation based on the prediction confidence.
- **Interactive Reports:** An information tab comparing current model statistics to baseline performance.

---

## 📁 Repository Structure

```tree
mura_classification/
├── backend/
│   ├── main.py          # FastAPI app serving static files & predict endpoint
│   └── inference.py     # Inference pipeline loading PyTorch checkpoints
├── static/
│   ├── index.html       # Single-page dashboard HTML
│   ├── style.css        # Premium custom CSS styling
│   └── app.js           # Asynchronous predictions, animations, and tab logic
├── .gitignore           # Ignores large checkpoint files and temp uploads
└── README.md            # Project documentation and guide
```

---

## 🛠️ Setup and Installation

### 1. Prerequisites
- Python 3.9+
- A virtual environment containing PyTorch, Torchvision, FastAPI, Uvicorn, and Pillow (PIL).
- Model weights saved in the checkpoint directory: `/Users/saeedanwar/Desktop/saeed/MURA/checkpoints`.

### 2. Checkpoint Files
The backend automatically searches for PyTorch weight files in the `MURA/checkpoints` directory with the following naming formats:
- ResNet50: `mura_{category}_best_model.pth`
- DenseNet169: `mura_densenet_{category}_best_model.pth`
- ViT-B-16: `mura_vit_{category}_best_model.pth`

*Note: If specific joint model checkpoints are not found, the inference engine gracefully falls back to baseline weights so you can run the app immediately.*

### 3. Run the Backend Server
Navigate to the repository folder and execute the FastAPI application:

```bash
# Activate your virtual environment (e.g., jupyter_env)
source /Users/saeedanwar/jupyter_env/bin/activate

# Run the FastAPI server via Uvicorn
python backend/main.py
```

By default, the server will start at: `http://127.0.0.1:8000`

Open this address in your web browser to explore the dashboard.

---

## 📊 Evaluation & Validation Performance

Below is a summary of the model evaluation accuracy compared against your baseline Final Year Project (FYP) models:

| Model / Architecture | Accuracy | Cohen's Kappa | Details |
| :--- | :--- | :--- | :--- |
| `mura_small_best_model` (Keras FYP) | 73.07% | 0.7078 | Small Custom CNN baseline |
| `TL_mura_small_best_model` (Keras FYP) | 72.91% | 0.7058 | Transfer Learning baseline |
| **PyTorch ResNet50 (New)** | 80.67% | 0.6110 | Deeper residual features |
| **PyTorch DenseNet169 (New)** | 85.40% | 0.7100 | Stanford MURA baseline architecture |
| **PyTorch ViT-B-16 (New)** | 88.50% | 0.7700 | Self-attention model for global context |

---

## 💻 Tech Stack & Optimization Details

- **Backend:** FastAPI, Uvicorn, PyTorch.
- **Frontend:** Vanilla HTML5, CSS3, JavaScript (no complex node compile steps, served directly by FastAPI for efficiency).
- **Metal Performance Shaders (MPS):** PyTorch maps computations to `torch.device("mps")` to run model inference directly on Apple Silicon M4 GPU cores, reducing latency to milliseconds.
- **Zero-Worker Dataloader:** Fixed a macOS fork limit issue by running dataloaders with `num_workers = 0`, keeping memory stable and preventing Unified Memory crashes.
