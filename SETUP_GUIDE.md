# GOEIC Enterprise Assistant — Windows Setup Guide

**Version:** 2.0 — Fully Offline  
**Requirements:** Windows 10/11 (64-bit), 16 GB RAM minimum, NVIDIA GPU recommended

---

## Overview

The system runs three Docker containers:

| Container | Purpose | Port |
|-----------|---------|------|
| `goeic-ollama` | Local LLM (Qwen 2.5) | 11434 |
| `goeic-weaviate` | Vector database | 8080 / 50051 |
| `goeic-app` | FastAPI application | 8000 |

All data is stored in named Docker volumes — your database persists across restarts and upgrades.

---

## Step 1 — Install Prerequisites

### 1.1 Docker Desktop

1. Download **Docker Desktop for Windows** from https://www.docker.com/products/docker-desktop/
2. Run the installer and follow the prompts (enable WSL 2 when asked).
3. Restart your computer.
4. Open Docker Desktop and wait for the whale icon in the taskbar to stop animating.
5. Verify the installation — open **PowerShell** and run:

```powershell
docker --version
docker compose version
```

Both commands should print version numbers. If they don't, restart Docker Desktop and try again.

### 1.2 NVIDIA GPU Support (Optional but Recommended)

Skip this section if you do not have an NVIDIA GPU. The application will still run on CPU.

1. Install the latest **NVIDIA Game Ready or Studio Driver** from https://www.nvidia.com/drivers
2. Install the **NVIDIA Container Toolkit for Windows (WSL 2)**:
   - Open PowerShell **as Administrator** and run:
   ```powershell
   wsl --install
   wsl --set-default-version 2
   ```
   - In Docker Desktop → Settings → Resources → WSL Integration, enable your WSL 2 distro.
3. Verify GPU access inside Docker:
   ```powershell
   docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
   ```
   If you see your GPU listed, GPU passthrough is working.

> **No GPU?** Open `docker-compose.yml` and delete or comment out the entire `deploy:` block inside the `ollama` service. Ollama will run on CPU only (slower generation speed).

---

## Step 2 — Download the Project

### Option A — Clone from GitHub (recommended)

```powershell
git clone https://github.com/YOUR-ORG/YOUR-REPO.git goeic
cd goeic
```

### Option B — Download ZIP

1. Go to the GitHub repository page.
2. Click **Code → Download ZIP**.
3. Extract the ZIP to a folder, e.g. `C:\goeic`.
4. Open PowerShell and `cd` into that folder:
   ```powershell
   cd C:\goeic
   ```

---

## Step 3 — Create the `.env` File

Copy the example file:

```powershell
copy .env.example .env
```

Open `.env` in Notepad (or VS Code) and fill in the values:

```env
# ── Ollama ────────────────────────────────────────────
OLLAMA_HOST=http://ollama:11434
OLLAMA_MODEL=qwen2.5:14b
OLLAMA_TIMEOUT=180

# ── Weaviate ──────────────────────────────────────────
WEAVIATE_HOST=weaviate

# ── Embeddings ────────────────────────────────────────
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2

# ── Whisper (voice) ───────────────────────────────────
# Options: tiny, base, small, medium, large
WHISPER_MODEL_SIZE=base
```

> **Tip:** If you don't have a `.env.example` file, just create a new `.env` file with the content above.

---

## Step 4 — Start All Services

Run this command from the project folder:

```powershell
docker compose up -d
```

Docker will:
1. Pull the Ollama, Weaviate, and app images (~4 GB total on first run).
2. Create the named volumes for persistent storage.
3. Start all three containers.

Check that all containers are running:

```powershell
docker compose ps
```

You should see `Up` (healthy) for all three services. The app takes about 60 seconds to become healthy on first start while it downloads the embedding model.

---

## Step 5 — Download the LLM Model

This is a **one-time step**. The model is saved to a Docker volume and survives container restarts.

### For NVIDIA GPU (recommended — 14B parameter model):

```powershell
docker exec -it goeic-ollama ollama pull qwen2.5:14b
```

Download size: ~9 GB. This may take 10–30 minutes depending on your internet connection.

### For CPU only (smaller, faster model):

```powershell
docker exec -it goeic-ollama ollama pull qwen2.5:7b
```

Download size: ~5 GB.

After downloading, update `OLLAMA_MODEL` in your `.env` file to match the model name you pulled, then restart the app:

```powershell
docker compose restart app
```

---

## Step 6 — Verify Everything Works

### Check the health endpoint:

```powershell
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "mode": "FULLY OFFLINE",
  "llm": "Ollama (qwen2.5:14b)",
  "weaviate": true,
  "ollama_ready": true
}
```

### Open the chat interface:

Navigate to **http://localhost:8000** in your browser.

### Open the admin dashboard:

Navigate to **http://localhost:8000/admin**

Default credentials:
- **Username:** `admin`
- **Password:** `goeic2026`

> **Important:** Change the default password immediately after first login via **Dashboard → Admins → Change Password**.

---

## Step 7 — Index Your Data

### 7.1 Scrape the GOEIC Website

1. Log into the admin dashboard at http://localhost:8000/admin
2. Click the **Data Management** tab.
3. Click **Start Smart Scraping**.
4. Monitor progress using the progress bar — first scrape takes 20–60 minutes.

### 7.2 Upload Word Documents via Excel

Prepare an Excel file (`.xlsx`) with these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `url` | Canonical web URL for this document | `https://www.goeic.gov.eg/ar/services/cert` |
| `title` | Document title | `شهادة المطابقة` |
| `category` | Category label | `الخدمات` |
| `language` | Language code | `ar` / `en` / `fr` |
| `path` | Relative path to the `.docx` file | `docs/cert.docx` |

Then:

1. In the admin dashboard, go to **Data Management**.
2. Click **Choose File** and select your Excel file.
3. Click **Upload & Index Locally**.

---

## Common Commands

### View live logs

```powershell
docker compose logs -f app
```

### Stop all services

```powershell
docker compose down
```

### Stop and remove all data (⚠️ deletes database!)

```powershell
docker compose down -v
```

### Restart only the application (after config change)

```powershell
docker compose restart app
```

### Update to the latest app image

```powershell
docker compose pull app
docker compose up -d app
```

### Check disk usage of volumes

```powershell
docker system df -v
```

---

## Troubleshooting

### App shows "Ollama not ready"

- Check Ollama is running: `docker compose ps`
- Check Ollama logs: `docker compose logs ollama`
- Ensure the model is downloaded: `docker exec -it goeic-ollama ollama list`

### Weaviate connection error on startup

- Weaviate needs a few seconds to initialize. Wait 30 seconds and check: `docker compose logs weaviate`
- Ensure ports 8080 and 50051 are not used by another process.

### GPU not detected inside Ollama

- Confirm `nvidia-smi` works on the host: run `nvidia-smi` in PowerShell.
- Ensure Docker Desktop is using WSL 2 (Settings → General → "Use the WSL 2 based engine").
- Restart Docker Desktop after installing the NVIDIA driver.

### Port already in use

If port 8000, 8080, or 11434 is already used by another application, edit `docker-compose.yml` and change the host port (left side of the colon):

```yaml
ports:
  - "8001:8000"   # change 8000 to 8001 (host:container)
```

### Download logs from the admin dashboard

1. Go to **Data Management** tab.
2. Set the number of lines in the input box.
3. Click **Download Logs** — a `.txt` file will be saved to your Downloads folder.

---

## Upgrading

When a new version of the application is released on GitHub:

```powershell
# Pull the new image
docker compose pull app

# Restart the app container (Ollama and Weaviate data are unaffected)
docker compose up -d app
```

Your Weaviate database and Ollama models are stored in Docker named volumes and are never touched by an app upgrade.

---

## Architecture Reference

```
Browser / Client
      │
      ▼
  ┌─────────┐        ┌─────────────┐
  │  App    │──────▶│   Ollama    │  (LLM inference, GPU)
  │  :8000  │        │   :11434   │
  │         │        └─────────────┘
  │         │        ┌─────────────┐
  │         │──────▶│  Weaviate   │  (Vector DB, volume)
  └─────────┘        │  :8080     │
                     └─────────────┘
```

All traffic stays on your local machine. No data is sent to external APIs.