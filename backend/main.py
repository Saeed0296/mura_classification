import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from inference import predict_single_image

app = FastAPI(title="MURA Classification API", description="FastAPI backend to run bone radiograph predictions using ResNet50, DenseNet169, and ViT models on macOS MPS.")

# Configure CORS so that frontend can communicate with backend easily
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temp directory to save uploaded images
TEMP_DIR = "/Users/saeedanwar/Desktop/saeed/mura_classification/temp"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    category: str = Form(...)
):
    valid_categories = ["XR_ELBOW", "XR_FINGER", "XR_FOREARM", "XR_HAND", "XR_HUMERUS", "XR_SHOULDER", "XR_WRIST", "ALL"]
    if category not in valid_categories:
        raise HTTPException(status_code=400, detail=f"Invalid joint category: {category}. Must be one of {valid_categories}")
        
    # Save the file temporarily
    temp_file_path = os.path.join(TEMP_DIR, image.filename)
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save uploaded image: {str(e)}")
        
    # Run predictions
    try:
        results = predict_single_image(temp_file_path, category)
    except Exception as e:
        results = {"error": f"Error running inference pipeline: {str(e)}"}
    finally:
        # Clean up temporary file to prevent disk fill-up
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
    if "error" in results:
        raise HTTPException(status_code=500, detail=results["error"])
        
    return results

# Mount static frontend directory
STATIC_DIR = "/Users/saeedanwar/Desktop/saeed/mura_classification/static"
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount the root directory to serve static HTML, CSS, JS
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
