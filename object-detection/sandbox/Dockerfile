# Replicates the AINM competition sandbox environment
# Python 3.11, PyTorch 2.6.0+cu124, ultralytics 8.1.0, onnxruntime-gpu 1.20.0
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Match sandbox Python packages
RUN pip install --no-cache-dir \
    ultralytics==8.1.0 \
    onnxruntime-gpu==1.20.0 \
    opencv-python-headless==4.9.0.80 \
    albumentations==1.3.1 \
    Pillow==10.2.0 \
    numpy==1.26.4 \
    scipy==1.12.0 \
    scikit-learn==1.4.0 \
    pycocotools==2.0.7 \
    ensemble-boxes==1.0.9 \
    timm==0.9.12 \
    supervision==0.18.0 \
    safetensors==0.4.2

# Working directory for submission code
WORKDIR /submission

# Data and output directories (mounted at runtime)
RUN mkdir -p /data/images /output

# Default entrypoint: run.py with sandbox contract args
ENTRYPOINT ["python", "run.py", "--input", "/data/images", "--output", "/output/predictions.json"]
