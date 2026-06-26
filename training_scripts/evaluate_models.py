import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score

# Paths
BASE_DIR = "/Users/saeedanwar/Desktop/saeed/MURA"
DATASET_PATH = os.path.join(BASE_DIR, "MURA-v1.1")
VALID_CSV = os.path.join(DATASET_PATH, "valid_image_paths.csv")

# 1. Load and parse validation dataset directly from official CSV
print("Loading official validation dataset csv...")
if not os.path.exists(VALID_CSV):
    raise FileNotFoundError(f"Validation CSV not found at: {VALID_CSV}")

val_df = pd.read_csv(VALID_CSV, header=None, names=['path'])

# Get class name matching Keras class names format
def get_class_name(path):
    parts = path.split('/')
    if len(parts) > 2:
        joint = parts[2]  # e.g. XR_WRIST
        label = 'POSITIVE' if 'positive' in path.lower() else 'NEGATIVE'
        return f"{joint}_{label}"
    return "UNKNOWN"

val_df['class'] = val_df['path'].apply(get_class_name)
print(f"Validation dataset size: {len(val_df)}")

# Explicit class ordering matching original Keras model training
classes = [
    "XR_ELBOW_NEGATIVE", "XR_ELBOW_POSITIVE",
    "XR_FINGER_NEGATIVE", "XR_FINGER_POSITIVE",
    "XR_FOREARM_NEGATIVE", "XR_FOREARM_POSITIVE",
    "XR_HAND_NEGATIVE", "XR_HAND_POSITIVE",
    "XR_HUMERUS_NEGATIVE", "XR_HUMERUS_POSITIVE",
    "XR_SHOULDER_NEGATIVE", "XR_SHOULDER_POSITIVE",
    "XR_WRIST_NEGATIVE", "XR_WRIST_POSITIVE"
]

# 2. Setup ImageDataGenerator
image_size = (224, 224)
batch_size = 64

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=BASE_DIR,
    x_col='path',
    y_col='class',
    classes=classes,  # Force mapping of index to all 14 classes
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# 3. List of models to evaluate
models_to_test = [
    "mura_small_best_model_1.h5",
    "TL_mura_small_best_model_1.h5"
]

results = {}

for m_name in models_to_test:
    model_path = os.path.join(BASE_DIR, m_name)
    if not os.path.exists(model_path):
        print(f"\nModel not found: {m_name}")
        continue
        
    print(f"\n==========================================")
    print(f"EVALUATING MODEL: {m_name}")
    print(f"==========================================")
    
    try:
        # Load model
        model = load_model(model_path)
        
        # Predict on validation set
        print("Running predictions...")
        predictions = model.predict(val_generator, verbose=1)
        
        # Get true labels and predicted labels
        true_labels = val_generator.classes
        pred_labels = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        acc = accuracy_score(true_labels, pred_labels)
        kappa = cohen_kappa_score(true_labels, pred_labels)
        
        print(f"\nResults for {m_name}:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Cohen's Kappa: {kappa:.4f}")
        print("\nClassification Report:")
        print(classification_report(true_labels, pred_labels, target_names=classes, zero_division=0))
        
        results[m_name] = {
            "accuracy": acc,
            "kappa": kappa
        }
    except Exception as e:
        print(f"Error evaluating {m_name}: {e}")

# Save evaluation results summary
with open(os.path.join(BASE_DIR, "evaluation_results.json"), "w") as f:
    import json
    json.dump(results, f, indent=4)
print("\nEvaluation completed. Results saved to evaluation_results.json")
