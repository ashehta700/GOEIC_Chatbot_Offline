# 1. Use NVIDIA's official CUDA image instead of python-slim
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# 2. Install Python and FFmpeg (for Whisper voice)
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Copy requirements
COPY requirements_offline.txt .

# 4. Install PyTorch with CUDA 12.1 support (Matching your README)
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install the rest of your offline dependencies
RUN pip3 install --no-cache-dir -r requirements_offline.txt

# 6. Copy all your files (main_offline.py, smart_scraper_offline.py, etc.)
COPY . .

# 7. Expose FastAPI port
EXPOSE 5000

# 8. Run your specific offline main file
CMD ["python3", "main_offline.py"]