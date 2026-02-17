#!/bin/bash

# ═══════════════════════════════════════════════════════════════
# GOEIC OFFLINE RAG - ONE-CLICK DEPLOYMENT SCRIPT
# ═══════════════════════════════════════════════════════════════

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════"
echo "  GOEIC OFFLINE RAG SYSTEM - DEPLOYMENT"
echo "════════════════════════════════════════════════════════════"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    echo -e "${RED}⚠️  Please do not run as root${NC}"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════
# 1. CHECK PREREQUISITES
# ═══════════════════════════════════════════════════════════════

echo -e "\n${YELLOW}📋 Checking prerequisites...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.10+${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python found: $(python3 --version)${NC}"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ NVIDIA GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}⚠️  No NVIDIA GPU detected. CPU mode will be slower.${NC}"
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Installing...${NC}"
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    sudo usermod -aG docker $USER
    echo -e "${GREEN}✅ Docker installed. Please logout and login again.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker found${NC}"

# ═══════════════════════════════════════════════════════════════
# 2. INSTALL OLLAMA
# ═══════════════════════════════════════════════════════════════

echo -e "\n${YELLOW}🤖 Setting up Ollama...${NC}"

if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo -e "${GREEN}✅ Ollama installed${NC}"
else
    echo -e "${GREEN}✅ Ollama already installed${NC}"
fi

# Start Ollama service
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama server..."
    ollama serve > /dev/null 2>&1 &
    sleep 3
fi

# Pull model
echo "Pulling Qwen 2.5 14B model (this may take a while)..."
ollama pull qwen2.5:14b

echo -e "${GREEN}✅ Ollama ready with Qwen 2.5 14B${NC}"

# ═══════════════════════════════════════════════════════════════
# 3. SETUP WEAVIATE
# ═══════════════════════════════════════════════════════════════

echo -e "\n${YELLOW}🗄️  Setting up Weaviate database...${NC}"

# Create docker-compose if doesn't exist
if [ ! -f "docker-compose.yml" ]; then
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
fi

# Start Weaviate
echo "Starting Weaviate..."
docker-compose up -d

# Wait for Weaviate to be ready
echo "Waiting for Weaviate to start..."
for i in {1..30}; do
    if curl -s http://localhost:8080/v1/meta > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Weaviate ready${NC}"
        break
    fi
    sleep 1
done

# ═══════════════════════════════════════════════════════════════
# 4. SETUP PYTHON ENVIRONMENT
# ═══════════════════════════════════════════════════════════════

echo -e "\n${YELLOW}🐍 Setting up Python environment...${NC}"

# Create venv if doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "Installing Python packages..."
pip install -r requirements_offline.txt

# Install PyTorch with CUDA if GPU available
if command -v nvidia-smi &> /dev/null; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

echo -e "${GREEN}✅ Python environment ready${NC}"

# ═══════════════════════════════════════════════════════════════
# 5. CONFIGURATION
# ═══════════════════════════════════════════════════════════════

echo -e "\n${YELLOW}⚙️  Creating configuration...${NC}"

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${GREEN}✅ Created .env file${NC}"
    echo -e "${YELLOW}⚠️  Please edit .env if needed${NC}"
else
    echo -e "${GREEN}✅ .env already exists${NC}"
fi

# Create necessary directories
mkdir -p logs uploads public

# ═══════════════════════════════════════════════════════════════
# 6. VERIFY INSTALLATION
# ═══════════════════════════════════════════════════════════════

echo -e "\n${YELLOW}🔍 Verifying installation...${NC}"

# Check Python packages
python3 << 'EOF'
import sys
try:
    import torch
    import sentence_transformers
    import weaviate
    import fastapi
    print(f"✅ All packages installed")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"❌ Missing package: {e}")
    sys.exit(1)
EOF

echo -e "${GREEN}✅ Installation complete!${NC}"

# ═══════════════════════════════════════════════════════════════
# 7. NEXT STEPS
# ═══════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  🎉 INSTALLATION COMPLETE!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "📌 Next steps:"
echo ""
echo "1️⃣  Index your data:"
echo "   ${YELLOW}python3 smart_scraper_offline.py${NC}     # Scrape website"
echo "   ${YELLOW}OR${NC}"
echo "   ${YELLOW}python3 smart_excel_uploader_offline.py${NC}  # Upload from Excel"
echo ""
echo "2️⃣  Start the application:"
echo "   ${YELLOW}python3 main_offline.py${NC}"
echo ""
echo "3️⃣  Access the interface:"
echo "   ${GREEN}http://localhost:8000${NC} (User interface)"
echo "   ${GREEN}http://localhost:8000/admin${NC} (Admin panel)"
echo ""
echo "4️⃣  Check health:"
echo "   ${YELLOW}curl http://localhost:8000/health${NC}"
echo ""
echo "📖 For detailed documentation, see: SETUP_GUIDE.md"
echo "════════════════════════════════════════════════════════════"
