# Use official Python slim image
# significantly smaller than nvidia/cuda base
# PyTorch wheels will provide necessary CUDA runtime libraries
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies
# ffmpeg is crucial for audio processing
# git is needed for pip installs from git
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first to leverage cache
COPY requirements.txt .

# Install Python dependencies
# Use --no-cache-dir to keep image size small
# Increase timeout for large downloads
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose the Flask port
EXPOSE 7860

# Define the run command
# Ensure run.sh is executable
RUN chmod +x run.sh

# Run the application
CMD ["./run.sh"]
