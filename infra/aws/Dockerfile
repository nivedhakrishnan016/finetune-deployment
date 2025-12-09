# 1. Switch to NVIDIA's PyTorch base image with CUDA
FROM nvcr.io/nvidia/pytorch:22.12-py3

# Set environment variables
ENV PYTHONUNBUFFERED 1
# Correct format for Gunicorn to find the app instance
ENV FLASK_APP=service.app:app
ENV PORT 5000
ENV MODEL_NAME_OR_PATH mistralai/Mistral-7B-Instruct-v0.2
# This path is where the adapter weights will be copied inside the container
ENV ADAPTER_PATH /app/lora_adapter  # Simplified path

# Set the working directory
WORKDIR /app

# The base image already has python, pip, and CUDA/PyTorch installed.
# We only need to install the project-specific dependencies.

# Copy requirements and install Python dependencies
COPY infra/requirements.txt /app/
# Note: You can remove '--no-cache-dir' as it's less critical with a staged build
RUN pip install -r requirements.txt

# --- COPY APPLICATION AND MODEL WEIGHTS ---
# 1. Copy the application code (service/app.py)
# Note: Ensure the copy destination matches the FLASK_APP path (service/app.py)
COPY service/app.py /app/service/app.py

# 2. Copy the LORA adapter weights into the simplified path
# Source path: relative to the build context (~/)
# Destination path: matches the new ADAPTER_PATH /app/lora_adapter
COPY fine_tune_output/lora_adapter /app/lora_adapter

# Expose the Flask port
EXPOSE 5000

# Start the application using Gunicorn
CMD ["gunicorn", "--workers", "1", "--threads", "1", "--timeout", "120", "--bind", "0.0.0.0:5000", "service.app:app"]