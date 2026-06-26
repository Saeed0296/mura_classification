FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    MURA_CHECKPOINT_DIR=/app/MURA/checkpoints \
    MURA_TEMP_DIR=/app/temp \
    MURA_STATIC_DIR=/app/static \
    HOST=0.0.0.0 \
    PORT=7860

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
# We use the CPU-only version of PyTorch to reduce image size and RAM usage on Hugging Face Spaces free CPU tier.
COPY requirements_hf.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY backend/ /app/backend/
COPY static/ /app/static/
COPY MURA/checkpoints/ /app/MURA/checkpoints/

# Expose default port for Hugging Face Spaces
EXPOSE 7860

# Command to run uvicorn
CMD ["python", "backend/main.py"]
