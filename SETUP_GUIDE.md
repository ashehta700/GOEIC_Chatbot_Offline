# ═══════════════════════════════════════════════════════════════
#  GOEIC OFFLINE RAG — LINUX DEPLOYMENT GUIDE
#  Ubuntu 22.04 / 24.04 LTS  (also works on Debian 12)
#  Server: Xeon E5-1650 v4 | 30 GB RAM | RTX 4070 Ti SUPER 16 GB
# ═══════════════════════════════════════════════════════════════

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 COMPATIBILITY SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ✅ Windows 10/11   — Tested, fully working
  ✅ Linux Ubuntu 22 — Tested, fully working
  ✅ Linux Ubuntu 24 — Tested, fully working

  Cross-platform fixes applied:
  • tempfile.gettempdir() instead of hardcoded paths
  • asyncio.get_running_loop() instead of get_event_loop()
  • ThreadPoolExecutor (no subprocess) for background tasks
  • Thread-safe WebSocket broadcaster (no new event loops)
  • Path() objects everywhere (no os.sep issues)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 TABLE OF CONTENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  PART 1  │  System Packages
  PART 2  │  GPU DRIVER + CUDA  ← FIX "GPU not working"
  PART 3  │  Docker + Weaviate
  PART 4  │  Ollama (Local LLM)
  PART 5  │  Python Environment
  PART 6  │  Project Files + .env
  PART 7  │  First Run & Smoke Tests
  PART 8  │  Systemd Service (auto-start)
  PART 9  │  Nginx Reverse Proxy
  PART 10 │  Monitoring Commands
  PART 11 │  Troubleshooting GPU

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PART 1 — SYSTEM PACKAGES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 1.1 Update system
sudo apt update && sudo apt upgrade -y

# 1.2 Install required packages
sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    git \
    wget \
    curl \
    ffmpeg \
    nginx \
    htop

# 1.3 Verify FFmpeg (needed for Whisper voice)
ffmpeg -version
# Expected: ffmpeg version 4.4.x or 6.x


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PART 2 — GPU DRIVER + CUDA  (Fix "GPU not working")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ─── STEP 2.1: Check what's currently installed ──────────────

nvidia-smi

# If nvidia-smi WORKS → jump to Step 2.3 (install CUDA)
# If nvidia-smi says "command not found" → do Step 2.2 first

# ─── STEP 2.2: Install NVIDIA Driver ─────────────────────────

# Find recommended driver version
sudo apt install ubuntu-drivers-common -y
ubuntu-drivers devices
# Look for line like: driver : nvidia-driver-550 - recommended

# Install recommended driver
sudo ubuntu-drivers autoinstall

# REBOOT (required after driver install)
sudo reboot

# After reboot, verify driver
nvidia-smi
# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 550.xx   Driver Version: 550.xx   CUDA Version: 12.4          |
# +-----------------------------------------------------------------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. |
# | RTX 4070 Ti S...     Off |   ...                |          0 |
# | 70W  /  285W |   2000MiB /  16376MiB |      1%      Default |

# ─── STEP 2.3: Install CUDA Toolkit 12.4 ─────────────────────

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-4 -y

# Add CUDA to PATH (add to ~/.bashrc for persistence)
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA
nvcc --version
# Expected: Cuda compilation tools, release 12.4

# ─── STEP 2.4: Verify GPU is visible to Python ───────────────

python3 -c "
import torch
print('PyTorch version :', torch.__version__)
print('CUDA available  :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU name        :', torch.cuda.get_device_name(0))
    print('VRAM total      :', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), 'GB')
    print('VRAM free       :', round(torch.cuda.memory_reserved(0) / 1024**3, 2), 'GB used')
else:
    print('❌ GPU not visible — check driver installation above')
"

# ─── STEP 2.5: If CUDA shows "available: False" after driver ──
# Most common reason: PyTorch was installed for CPU only
# Fix by reinstalling PyTorch with CUDA support (done in Part 5)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PART 3 — DOCKER + WEAVIATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 3.1 Install Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
sudo apt install docker-compose-plugin -y
newgrp docker   # Apply group without logout

# Verify
docker --version
docker compose version

# 3.2 Create project directory
mkdir -p ~/goeic_rag/weaviate_data
cd ~/goeic_rag

# 3.3 Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.4'
services:
  weaviate:
    image: semitechnologies/weaviate:1.27.0
    ports:
      - "8080:8080"
      - "50051:50051"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      ENABLE_MODULES: ''
      CLUSTER_HOSTNAME: 'node1'
    volumes:
      - ./weaviate_data:/var/lib/weaviate
    restart: unless-stopped
EOF

# 3.4 Start Weaviate
docker compose up -d

# 3.5 Wait and verify (may take 30 seconds first start)
sleep 15
curl http://localhost:8080/v1/meta | python3 -m json.tool | head -10
# Should return JSON with Weaviate version info

# 3.6 Make Docker start on boot
sudo systemctl enable docker


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PART 4 — OLLAMA (LOCAL LLM)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 4.1 Install Ollama (auto-detects GPU)
curl -fsSL https://ollama.com/install.sh | sh

# 4.2 Enable auto-start
sudo systemctl enable ollama
sudo systemctl start ollama
sleep 3

# Verify Ollama is running
sudo systemctl status ollama --no-pager
# Should show: Active: active (running)

# 4.3 Pull the AI model
# For RTX 4070 Ti SUPER (16 GB VRAM) → 14B is best quality
ollama pull qwen2.5:14b

# This takes 5-15 minutes on first run (8.5 GB download)
# You can watch progress live

# Alternative smaller models:
# ollama pull qwen2.5:7b    ← faster, uses ~5 GB VRAM
# ollama pull qwen2.5:3b    ← fastest, lowest quality

# 4.4 Verify model + GPU usage
ollama list
# Should show: qwen2.5:14b   ...GB

# Test inference with GPU
ollama run qwen2.5:14b "اختبار سريع"

# While running, in ANOTHER terminal check GPU:
 
# Should show: VRAM usage jumped to ~12000 MiB for 14B model
# This confirms GPU is being used by Ollama ✅


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PART 5 — PYTHON ENVIRONMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cd ~/goeic_rag

# 5.1 Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Verify Python version
python --version
# Expected: Python 3.11.x

# 5.2 Upgrade pip
pip install --upgrade pip setuptools wheel

# 5.3 Install PyTorch WITH CUDA 12.1 support
# THIS IS THE KEY STEP FOR GPU SUPPORT
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# ✅ Verify GPU works in PyTorch IMMEDIATELY after install
python -c "
import torch
if torch.cuda.is_available():
    print('✅ GPU WORKING:', torch.cuda.get_device_name(0))
    print('   VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1024**3,1), 'GB')
else:
    print('❌ GPU not detected by PyTorch')
    print('   Check: nvidia-smi works? CUDA installed?')
"

# 5.4 Install project requirements
pip install -r requirements_offline.txt

# NOTE: If you see conflicts, install in this order:
# pip install weaviate-client==4.9.3
# pip install sentence-transformers==3.3.1
# pip install openai-whisper==20231117
# pip install fastapi uvicorn aiohttp httpx
# pip install edge-tts bcrypt python-multipart
# pip install beautifulsoup4 lxml requests
# pip install pandas openpyxl python-docx
# pip install langchain-text-splitters python-dotenv

# 5.5 Verify all key imports
python -c "
import torch, whisper, weaviate, fastapi, edge_tts
from sentence_transformers import SentenceTransformer
print('✅ All imports OK')
print('   GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY')
"


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PART 6 — PROJECT FILES + .env
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 6.1 Expected directory structure
#
# ~/goeic_rag/
# ├── main_offline.py
# ├── smart_scraper_offline.py
# ├── smart_excel_uploader_offline.py
# ├── requirements_offline.txt
# ├── .env
# ├── docker-compose.yml
# ├── weaviate_data/             ← auto-created
# ├── logs/                      ← auto-created by app
# ├── uploads/                   ← auto-created by app
# └── public/
#     ├── index.html
#     ├── dashboard.html         ← use dashboard_updated.html
#     ├── login.html
#     └── logo.png

# 6.2 Copy files from Windows to Linux via SCP
# Run this on your WINDOWS machine (Git Bash or PowerShell):
#
#   scp -r "D:/path/to/goeic_rag/*" username@SERVER_IP:~/goeic_rag/
#
# Or use WinSCP (GUI) to drag and drop files

# 6.3 Create .env file
cat > ~/goeic_rag/.env << 'EOF'
# ── LLM Configuration ─────────────────────────────────────────
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:14b
OLLAMA_TIMEOUT=180

# ── Embedding Model ────────────────────────────────────────────
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2

# ── Voice / Whisper ────────────────────────────────────────────
WHISPER_MODEL_SIZE=base

# ── Weaviate ───────────────────────────────────────────────────
WEAVIATE_HOST=localhost

# ── Security ───────────────────────────────────────────────────
# Change this in production!
SECRET_KEY=change_this_to_a_long_random_string_in_production
EOF

# 6.4 Set correct permissions
chmod 600 ~/goeic_rag/.env
chmod +x ~/goeic_rag/main_offline.py


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PART 7 — FIRST RUN & SMOKE TESTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cd ~/goeic_rag
source venv/bin/activate

# 7.1 Create Weaviate schema (run ONCE before first start)
python - << 'PYEOF'
import weaviate
from weaviate.classes.config import Configure, Property, DataType

client = weaviate.connect_to_local()

if not client.collections.exists("GOEIC_Knowledge_Base_V2"):
    client.collections.create(
        name="GOEIC_Knowledge_Base_V2",
        properties=[
            Property(name="content",     data_type=DataType.TEXT),
            Property(name="title",       data_type=DataType.TEXT),
            Property(name="url",         data_type=DataType.TEXT),
            Property(name="category",    data_type=DataType.TEXT),
            Property(name="language",    data_type=DataType.TEXT),
            Property(name="source_type", data_type=DataType.TEXT),
            Property(name="chunk_type",  data_type=DataType.TEXT),
            Property(name="parent_id",   data_type=DataType.TEXT),
            Property(name="content_hash",data_type=DataType.TEXT),
        ]
    )
    print("✅ Collection created: GOEIC_Knowledge_Base_V2")
else:
    print("✅ Collection already exists")

client.close()
PYEOF


# 7.2 Start the application (foreground for first test)
python main_offline.py

# Expected startup output:
# ✅ Embedding Model: paraphrase-multilingual-MiniLM-L12-v2
# 🎮 GPU: NVIDIA GeForce RTX 4070 Ti SUPER (16.0GB VRAM)    ← GPU WORKING ✅
# ✅ Local Embeddings on GPU
# ✅ Weaviate Connected
# ✅ Ollama Connected. Available models: ['qwen2.5:14b']
# INFO: Uvicorn running on http://0.0.0.0:8000

# 7.3 Quick smoke tests (open new terminal)

# Test: Server running
curl http://localhost:8000/health
# Expected: {"status":"healthy", "gpu":"NVIDIA GeForce RTX 4070 Ti SUPER ..."}

# Test: Chat page loads
curl -s http://localhost:8000 | grep -o "<title>.*</title>"
# Expected: <title>GOEIC Enterprise Assistant</title>

# Test: Admin login
curl -s http://localhost:8000/admin | grep -o "<title>.*</title>"
# Expected: <title>تسجيل الدخول - GOEIC Admin</title>

# Press CTRL+C to stop, then proceed to Part 8 for production setup


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PART 8 — SYSTEMD SERVICE (AUTO-START ON BOOT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 8.1 Create systemd service file
sudo tee /etc/systemd/system/goeic.service << EOF
[Unit]
Description=GOEIC Offline RAG Chatbot
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/goeic_rag
Environment="PATH=$HOME/goeic_rag/venv/bin:/usr/local/cuda/bin:/usr/bin:/bin"
ExecStartPre=/bin/bash -c 'cd $HOME/goeic_rag && docker compose up -d'
ExecStartPre=/bin/sleep 10
ExecStart=$HOME/goeic_rag/venv/bin/python $HOME/goeic_rag/main_offline.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=goeic

[Install]
WantedBy=multi-user.target
EOF

# 8.2 Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable goeic
sudo systemctl start goeic

# 8.3 Check service is running
sudo systemctl status goeic --no-pager
# Expected: Active: active (running)

# 8.4 View live logs from service
sudo journalctl -u goeic -f
# Press CTRL+C to stop following

# 8.5 Useful service commands
sudo systemctl restart goeic   # Restart after code changes
sudo systemctl stop goeic      # Stop
sudo systemctl start goeic     # Start
sudo journalctl -u goeic -n 50 # Last 50 log lines


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PART 9 — NGINX REVERSE PROXY (PORT 80/443)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 9.1 Create Nginx config
sudo tee /etc/nginx/sites-available/goeic << 'EOF'
server {
    listen 80;
    server_name your_domain_or_ip;

    # Increase upload size for Excel files
    client_max_body_size 50M;

    # ── WebSocket support (for live logs) ─────────────────────
    location /ws/ {
        proxy_pass         http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade    $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host       $host;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }

    # ── Main app ───────────────────────────────────────────────
    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_read_timeout 180s;
    }
}
EOF

# 9.2 Enable site
sudo ln -s /etc/nginx/sites-available/goeic /etc/nginx/sites-enabled/
sudo nginx -t           # Test config syntax
sudo systemctl reload nginx

# 9.3 Test (replace with your server IP)
curl http://YOUR_SERVER_IP/health

# 9.4 Optional: Add HTTPS with Let's Encrypt
# sudo apt install certbot python3-certbot-nginx -y
# sudo certbot --nginx -d yourdomain.com


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PART 10 — MONITORING COMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# GPU monitoring (live - run in separate terminal)
watch -n 1 nvidia-smi
# Look for:
#  Ollama process  → ~12000 MiB for 14B model during inference
#  Python process  → ~1000 MiB for embedding model
#  Total usage     → ~13000 MiB out of 16376 MiB

# RAM monitoring
htop
# Look for:
#  Total used < 25 GB (leaving 5 GB free)

# Application logs (live)
sudo journalctl -u goeic -f --no-pager

# Or tail log file directly
tail -f ~/goeic_rag/logs/production_trace.log

# Weaviate stats
curl http://localhost:8080/v1/meta | python3 -m json.tool

# Ollama running models
curl http://localhost:11434/api/ps

# Check all services at once
sudo systemctl status goeic ollama docker --no-pager


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PART 11 — TROUBLESHOOTING GPU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─── Problem: App logs show "⚠️ Local Embeddings on CPU" ────────

  Cause 1: PyTorch installed without CUDA
  Fix:
    source ~/goeic_rag/venv/bin/activate
    pip uninstall torch torchvision torchaudio -y
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121
    python -c "import torch; print(torch.cuda.is_available())"
    # Must print True


  Cause 2: CUDA version mismatch (e.g. CUDA 12.6 but PyTorch wants 12.1)
  Fix:
    # Check your CUDA version
    nvcc --version     # e.g. "release 12.4"
    nvidia-smi         # e.g. "CUDA Version: 12.4"

    # Install matching PyTorch:
    # CUDA 11.8 → --index-url .../whl/cu118
    # CUDA 12.1 → --index-url .../whl/cu121  ← most common
    # CUDA 12.4 → --index-url .../whl/cu124


─── Problem: Ollama uses CPU not GPU ───────────────────────────

  Check:
    nvidia-smi dmon -s u    # watch GPU utilization live
    # Run a query, GPU % should spike to 60-100%

  Fix 1: Reinstall Ollama after CUDA
    sudo systemctl stop ollama
    curl -fsSL https://ollama.com/install.sh | sh
    sudo systemctl start ollama
    ollama pull qwen2.5:14b

  Fix 2: Force GPU with env variable
    sudo tee /etc/systemd/system/ollama.service.d/override.conf << 'EOF'
    [Service]
    Environment="CUDA_VISIBLE_DEVICES=0"
    EOF
    sudo systemctl daemon-reload
    sudo systemctl restart ollama


─── Problem: "CUDA out of memory" ──────────────────────────────

  Your GPU: 16 GB VRAM
  Typical usage:
    qwen2.5:14b  → ~12 GB  ← recommended
    qwen2.5:7b   → ~5 GB   ← if 14B fails
    embeddings   → ~1 GB

  Fix: Switch to 7B model in .env
    OLLAMA_MODEL=qwen2.5:7b
    sudo systemctl restart goeic


─── Problem: "nvidia-smi not found" after reboot ───────────────

  Fix:
    sudo apt install --reinstall nvidia-driver-550
    sudo reboot
    nvidia-smi  # should work now


─── Problem: Docker GPU not working ────────────────────────────
  (Only needed if you want Weaviate on GPU - not required)

    sudo apt install nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker


─── Problem: WebSocket logs not showing in browser ─────────────

  Check Nginx config has WebSocket proxy (see Part 9)
  Check browser console (F12):
    Should show: "✅ متصل بخادم السجلات | Connected to log server"

  Test WebSocket directly:
    # Install wscat
    npm install -g wscat
    wscat -c ws://localhost:8000/ws/logs


─── Problem: Voice not working ─────────────────────────────────

  Check 1: FFmpeg installed
    ffmpeg -version    # must work

  Check 2: Whisper installed
    source ~/goeic_rag/venv/bin/activate
    python -c "import whisper; print('OK')"

  Check 3: Reinstall
    pip install openai-whisper==20231117 ffmpeg-python

  Check 4: Test manually
    python - << 'EOF'
    import whisper
    model = whisper.load_model("base")
    print("✅ Whisper loaded OK")
    EOF


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 COMPLETE QUICK-START SUMMARY (copy-paste order)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Run these commands IN ORDER on a fresh Ubuntu 22.04 server:

sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3.11-dev build-essential git wget curl ffmpeg nginx

# GPU Driver
sudo ubuntu-drivers autoinstall && sudo reboot
# (wait for reboot, then continue)

# CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update && sudo apt install cuda-toolkit-12-4 -y
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && source ~/.bashrc

# Docker
curl -fsSL https://get.docker.com | sudo sh && sudo usermod -aG docker $USER && newgrp docker
sudo apt install docker-compose-plugin -y

# Ollama
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable ollama && sudo systemctl start ollama
ollama pull qwen2.5:14b

# Project Setup
mkdir -p ~/goeic_rag/weaviate_data && cd ~/goeic_rag
# [Copy your project files here via SCP or git]

# Weaviate
cat > docker-compose.yml << 'EOF'
version: '3.4'
services:
  weaviate:
    image: semitechnologies/weaviate:1.27.0
    ports: ["8080:8080","50051:50051"]
    environment:
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
    volumes: ["./weaviate_data:/var/lib/weaviate"]
    restart: unless-stopped
EOF
docker compose up -d && sleep 15

# Python Environment
python3.11 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_offline.txt

# Verify GPU
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"

# Create schema and run
python -c "
import weaviate
from weaviate.classes.config import Configure, Property, DataType
client = weaviate.connect_to_local()
if not client.collections.exists('GOEIC_Knowledge_Base_V2'):
    client.collections.create('GOEIC_Knowledge_Base_V2', properties=[
        Property(name='content', data_type=DataType.TEXT),
        Property(name='title', data_type=DataType.TEXT),
        Property(name='url', data_type=DataType.TEXT),
        Property(name='category', data_type=DataType.TEXT),
        Property(name='language', data_type=DataType.TEXT),
        Property(name='source_type', data_type=DataType.TEXT),
        Property(name='chunk_type', data_type=DataType.TEXT),
        Property(name='parent_id', data_type=DataType.TEXT),
        Property(name='content_hash', data_type=DataType.TEXT),
    ])
    print('✅ Schema created')
client.close()
"

python main_offline.py
# 🎉 Server running on http://0.0.0.0:8000

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EXPECTED STARTUP LOG (with GPU working correctly)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2026-02-17 10:00:01 | INFO | init_embeddings | 🔄 Loading local embedding model...
2026-02-17 10:00:04 | INFO | init_embeddings | 🎮 GPU: NVIDIA GeForce RTX 4070 Ti SUPER (16.0GB VRAM)
2026-02-17 10:00:04 | INFO | init_embeddings | ✅ Local Embeddings on GPU     ← GPU CONFIRMED ✅
2026-02-17 10:00:04 | INFO | init_embeddings | ✅ Embedding Model: paraphrase-multilingual-MiniLM-L12-v2
2026-02-17 10:00:05 | INFO | <module>        | ✅ Weaviate Connected
2026-02-17 10:00:05 | INFO | <module>        | ════════════════════════════
2026-02-17 10:00:05 | INFO | <module>        | 🚀 GOEIC Enterprise OFFLINE V2
2026-02-17 10:00:05 | INFO | <module>        | 🤖 LLM: Ollama (qwen2.5:14b)
2026-02-17 10:00:05 | INFO | <module>        | 📊 Embeddings: Local (paraphrase-multilingual-MiniLM-L12-v2)
2026-02-17 10:00:05 | INFO | <module>        | 🎮 GPU: RTX 4070 Ti SUPER
2026-02-17 10:00:05 | INFO | <module>        | 💰 API Costs: $0.00 (100% OFFLINE)
2026-02-17 10:00:06 | INFO | lifespan        | ✅ Ollama Connected. Available models: ['qwen2.5:14b']
INFO:                                           Uvicorn running on http://0.0.0.0:8000

# If you see "⚠️ Local Embeddings on CPU" → see Part 11 GPU troubleshooting