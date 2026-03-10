# ═══════════════════════════════════════════════════════════════
# GOEIC ENTERPRISE ASSISTANT - FULLY OFFLINE VERSION
# Uses: Ollama (Qwen), Local Embeddings, Local Whisper
# ═══════════════════════════════════════════════════════════════

import os
import logging
from logging.handlers import RotatingFileHandler
import re
import json
import shutil
import time
import base64
import uuid
import sys
import sqlite3
import secrets
import asyncio
import aiohttp
import requests
import io
import csv
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

# FastAPI
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, status, Depends, Response, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

# Security & Voice
import bcrypt
import edge_tts
from bs4 import BeautifulSoup
import torch

# AI Clients - LOCAL ONLY
from dotenv import load_dotenv
import weaviate
from weaviate.classes.query import MetadataQuery, Filter
from sentence_transformers import SentenceTransformer
import httpx

# Whisper for voice (local)
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️ Whisper not installed. Voice features disabled. Install with: pip install openai-whisper")

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# LOGGER SETUP
# ═══════════════════════════════════════════════════════════════
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "production_trace.log"

log_formatter = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(funcName)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler = RotatingFileHandler(
    LOG_FILE, mode='a', maxBytes=10*1024*1024,
    backupCount=10, encoding='utf-8'
)
file_handler.setFormatter(log_formatter)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

logger = logging.getLogger("GOEIC_Enterprise_Offline_V2")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# NOTE: WebSocket log streaming has been intentionally removed.
# Reason: persistent WebSocket connections add server load with no benefit
# for admin log monitoring. Use the REST endpoints instead:
#   GET /api/admin/logs/view?lines=N    → returns last N lines as JSON
#   GET /api/admin/logs/download?lines=N → streams log file as text attachment

# ═══════════════════════════════════════════════════════════════
# DATABASE & SECURITY
# ═══════════════════════════════════════════════════════════════
DB_PATH = "analytics.db"
active_sessions = {}

def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode('utf-8'), hashed.encode('utf-8'))
    except:
        return False

def init_db():
    """Initialize database with optimized settings"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_logs (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 request_id TEXT,
                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                 ip_address TEXT,
                 country TEXT,
                 device_type TEXT,
                 intent TEXT,
                 language TEXT,
                 query TEXT,
                 response_time REAL,
                 status_code INTEGER
                 )''')

    c.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON chat_logs(timestamp);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_intent ON chat_logs(intent);")
    c.execute("CREATE INDEX IF NOT EXISTS idx_country ON chat_logs(country);")

    c.execute('''CREATE TABLE IF NOT EXISTS users (
                 username TEXT PRIMARY KEY,
                 password_hash TEXT
                 )''')

    c.execute("SELECT * FROM users WHERE username='admin'")
    if not c.fetchone():
        hashed = get_password_hash("goeic2026")
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", ("admin", hashed))
        logger.info("✅ Default Admin Account Created")

    conn.commit()
    conn.close()

init_db()

# ═══════════════════════════════════════════════════════════════
# LOCAL AI CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Ollama Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "180"))

# Local Embedding Model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
local_embedder = None
ollama_client = None

# Whisper Model (for voice)
whisper_model = None
if WHISPER_AVAILABLE:
    WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

def init_embeddings():
    global local_embedder
    try:
        logger.info("🔄 Loading local embedding model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        local_embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🎮 GPU: {gpu_name} ({vram:.1f}GB VRAM)")
            logger.info(f"✅ Local Embeddings on GPU")
        else:
            logger.warning("⚠️ Local Embeddings on CPU (slower)")

        logger.info(f"✅ Embedding Model: {EMBEDDING_MODEL}")
    except Exception as e:
        logger.error(f"❌ Failed to load embeddings: {e}")
        raise

init_embeddings()

def init_whisper():
    global whisper_model
    if WHISPER_AVAILABLE and whisper_model is None:
        try:
            logger.info(f"🔄 Loading Whisper model ({WHISPER_MODEL_SIZE})...")
            whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
            logger.info("✅ Whisper model loaded")
        except Exception as e:
            logger.error(f"❌ Whisper load failed: {e}")

# Weaviate Connection
weaviate_host = os.getenv("WEAVIATE_HOST", "localhost")
client = None
collection = None

try:
    if weaviate_host == "localhost":
        client = weaviate.connect_to_local()
    else:
        client = weaviate.connect_to_custom(
            http_host=weaviate_host, http_port=8080, http_secure=False,
            grpc_host=weaviate_host, grpc_port=50051, grpc_secure=False
        )
    collection = client.collections.get("GOEIC_Knowledge_Base_V2")
    logger.info("✅ Weaviate Connected")
except Exception as e:
    logger.warning(f"❌ Weaviate Error: {e}")

# ═══════════════════════════════════════════════════════════════
# FASTAPI LIFESPAN
# ═══════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ollama_client

    try:
        ollama_client = httpx.AsyncClient(
            base_url=OLLAMA_HOST,
            timeout=httpx.Timeout(timeout=OLLAMA_TIMEOUT, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        response = await ollama_client.get("/api/tags")
        if response.status_code == 200:
            models = [m.get('name') for m in response.json().get('models', [])]
            logger.info(f"✅ Ollama Connected. Available models: {models}")
            if OLLAMA_MODEL not in str(models):
                logger.warning(f"⚠️ Model '{OLLAMA_MODEL}' not found! Run: ollama pull {OLLAMA_MODEL}")
        else:
            logger.warning(f"⚠️ Ollama health check failed: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Ollama init failed: {e}")
        logger.error(f"   Make sure Ollama is running: ollama serve")

    yield

    if ollama_client:
        await ollama_client.aclose()

app = FastAPI(
    title="GOEIC Enterprise Assistant Offline V2",
    description="Fully offline RAG system with local LLM and embeddings",
    version="2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════
# ENHANCED JOB SCRAPING (Async)
# ═══════════════════════════════════════════════════════════════

class AsyncJobsCache:
    def __init__(self, ttl_minutes: int = 30):
        self._cache = {}
        self._ttl = ttl_minutes
        self._lock = asyncio.Lock()

    async def get(self, lang: str) -> Optional[str]:
        async with self._lock:
            if lang in self._cache:
                item = self._cache[lang]
                if datetime.now() < item['expires']:
                    logger.info(f"✅ Jobs Cache Hit for {lang}")
                    return item['data']
                del self._cache[lang]
            return None

    async def set(self, lang: str, data: str):
        async with self._lock:
            self._cache[lang] = {
                'data': data,
                'expires': datetime.now() + timedelta(minutes=self._ttl)
            }

jobs_cache = AsyncJobsCache(ttl_minutes=60)

async def extract_job_details_async(session: aiohttp.ClientSession, job_url: str, headers: dict) -> dict:
    if not job_url or job_url.strip() == '':
        return {}

    job_url = job_url.strip().replace(' ', '')

    try:
        async with session.get(job_url, headers=headers, timeout=aiohttp.ClientTimeout(total=8)) as resp:
            if resp.status != 200:
                return {}

            html = await resp.text()
            soup = BeautifulSoup(html, 'html.parser')
            container = soup.find('div', class_='date_detail_content')

            if not container:
                return {}

            details = {}
            field_mapping = {
                'اسم الوظيفة': 'job_title',
                'وصف الوظيفة': 'job_description',
                'مهارات الوظيفة': 'job_skills',
                'المستندات المطلوبة': 'required_documents',
                'الملخص': 'summary',
                'المستوي الوظيفي': 'job_level',
                'نوع التوظيف': 'employment_type',
                'تاريخ الأعلان': 'announcement_date',
                'تاريخ الإنتهاء': 'deadline_date',
            }

            rows = container.find_all('div', class_='d-flex')

            for row in rows:
                label_tag = row.find('p', class_='p-gold')
                if not label_tag:
                    continue

                label = label_tag.get_text(strip=True).replace(':', '').replace('  ', ' ').strip()
                value_tag = label_tag.find_next_sibling(['p', 'div'])

                if not value_tag:
                    continue

                value = value_tag.get_text(separator='\n', strip=True)

                if label in field_mapping:
                    details[field_mapping[label]] = value
                else:
                    details[label] = value

            apply_btn = container.find('a', class_='btn-primary')
            if apply_btn and apply_btn.get('href'):
                details['apply_link'] = apply_btn.get('href')

            return details

    except Exception as e:
        logger.debug(f"Detail fetch error for {job_url}: {e}")
        return {}

async def fetch_live_jobs_async(lang: str) -> str:
    cached = await jobs_cache.get(lang)
    if cached:
        return cached

    url_map = {
        "ar": "https://www.goeic.gov.eg/ar/media-center/jobs",
        "en": "https://www.goeic.gov.eg/en/media-center/jobs",
        "fr": "https://www.goeic.gov.eg/fr/media-center/jobs"
    }

    url = url_map.get(lang, url_map["ar"])

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept-Language': f'{lang},en;q=0.9',
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status != 200:
                    return f"Error {response.status}. Visit: {url}"

                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                job_table = soup.find('table', id='myTableJob')

                if not job_table:
                    return "No jobs found at the moment."

                tbody = job_table.find('tbody')
                if not tbody:
                    return "No jobs found at the moment."

                rows = tbody.find_all('tr')

                jobs_basic = []
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) < 3:
                        continue

                    title_cell = cells[0]
                    title_link = title_cell.find('a')

                    title = title_link.get_text(strip=True) if title_link else title_cell.get_text(strip=True)
                    job_url = title_link.get('href', '') if title_link else ''

                    if job_url:
                        job_url = job_url.strip().replace(' ', '')
                        if not job_url.startswith('http'):
                            job_url = f"https://www.goeic.gov.eg{job_url}"

                    description = cells[1].get_text(separator=' ', strip=True) if len(cells) > 1 else ''
                    skills = cells[2].get_text(separator='\n', strip=True) if len(cells) > 2 else ''

                    jobs_basic.append({
                        'title': title,
                        'url': job_url,
                        'basic_description': description,
                        'basic_skills': skills
                    })

                semaphore = asyncio.Semaphore(5)

                async def fetch_with_limit(job):
                    async with semaphore:
                        details = await extract_job_details_async(session, job['url'], headers)
                        return {**job, 'details': details}

                jobs_with_details = await asyncio.gather(
                    *[fetch_with_limit(job) for job in jobs_basic],
                    return_exceptions=True
                )

                jobs_list = []
                for idx, job in enumerate(jobs_with_details, 1):
                    if isinstance(job, Exception):
                        continue

                    details = job.get('details', {})

                    job_entry = f"""
{'═' * 70}
📌 الوظيفة #{idx}: {details.get('job_title', job['title'])}
{'═' * 70}
📍 المستوى: {details.get('job_level', 'غير محدد')}
📅 إعلان: {details.get('announcement_date', 'غير محدد')} | انتهاء: {details.get('deadline_date', 'غير محدد')}

📝 الوصف:
{details.get('job_description', job['basic_description'])[:400]}

🛠 المهارات:
{details.get('job_skills', job['basic_skills'])[:300]}

📎 المستندات: {details.get('required_documents', 'غير متوفر')[:200]}
🔗 تقديم: {details.get('apply_link', job['url'])}
{'═' * 70}
"""
                    jobs_list.append(job_entry)

                if not jobs_list:
                    return "لا توجد وظائف متاحة حالياً."

                final_output = f"""
{'═' * 70}
🔴 الوظائف المتاحة في الهيئة العامة للرقابة على الصادرات والواردات
{'═' * 70}
📅 تحديث: {datetime.now().strftime('%Y-%m-%d %H:%M')}
📊 العدد: {len(jobs_list)} وظيفة

{''.join(jobs_list)}
{'═' * 70}
"""

                await jobs_cache.set(lang, final_output)
                return final_output

    except Exception as e:
        logger.error(f"Jobs error ({lang}): {e}")
        return f"خطأ في جلب الوظائف. يرجى زيارة: {url}"

# ═══════════════════════════════════════════════════════════════
# MIDDLEWARE & LOGGING
# ═══════════════════════════════════════════════════════════════
def get_location_from_ip(ip: str) -> str:
    if ip in ["127.0.0.1", "localhost", "::1"]:
        return "Localhost"
    try:
        response = requests.get(f"http://ip-api.com/json/{ip}", timeout=2)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                return data.get('country', 'Unknown')
    except:
        pass
    return "Unknown"

def log_to_db(req_id, ip, device, intent, lang, query, time_taken, status):
    try:
        country = get_location_from_ip(ip)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""INSERT INTO chat_logs
                     (request_id, ip_address, country, device_type, intent, language, query, response_time, status_code)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                  (req_id, ip, country, device, intent, lang, query, time_taken, status))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"DB Logging Error: {e}")

@app.middleware("http")
async def log_middleware(request: Request, call_next):
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    forwarded = request.headers.get("X-Forwarded-For")
    client_ip = forwarded.split(",")[0] if forwarded else (
        request.client.host if request.client else "Unknown"
    )
    ua = request.headers.get("User-Agent", "Unknown").lower()
    device_type = "📱 Mobile" if any(x in ua for x in ["mobile", "android", "iphone"]) else "💻 Desktop"
    request.state.client_ip = client_ip
    request.state.device_type = device_type
    request.state.request_id = request_id

    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        # Only log API calls; skip static files & log endpoints to avoid noise
        if "/api/" in request.url.path and "/logs/" not in request.url.path:
            logger.info(f"✅ [{request_id}] {request.method} {request.url.path} | {process_time:.2f}s")
        return response
    except Exception as e:
        logger.error(f"🔥 [{request_id}] ERROR: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "request_id": request_id}
        )

# ═══════════════════════════════════════════════════════════════
# RAG LOGIC
# ═══════════════════════════════════════════════════════════════
FALLBACK_LINKS = """
[OFFICIAL LINKS]
1. **Training Courses:** https://www.goeic.gov.eg/ar/training-courses/categories
2. **Online Booking:** https://www.goeic.gov.eg/ar/reserv-online
3. **Job Vacancies:** https://www.goeic.gov.eg/ar/media-center/jobs
4. **Contact Us:** https://www.goeic.gov.eg/ar/about-us/callUs
5. **Reports & Publications:** https://www.goeic.gov.eg/ar/services-and-activities/reports
6. **Events:** https://www.goeic.gov.eg/ar/media-center/events
7. **News:** https://www.goeic.gov.eg/ar/media-center/news
"""

LOGICAL_CLASS_WEIGHTS = {
    "web": 1.1, "pdf_ocr": 0.9, "api_db": 0.5,
    "golden_source": 0.7, "news": 0.5, "events": 0.5,
    "training": 0.8, "service_info": 0.9, "legal_decision": 1.0, "jobs": 2.0
}

def detect_intent(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["وظائف", "وظيفة", "jobs", "job", "vacancy", "vacancies", "career", "employment", "hiring"]):
        return "jobs"
    if re.search(r'\b\d{1,4}\b', q) and any(w in q for w in ["قرار", "decision", "رقم", "law", "قانون", "لائحة"]):
        return "legal_decision"
    if any(w in q for w in ["تدريب", "دورة", "دورات", "course", "training", "برنامج", "program", "حجز دورة", "register course"]):
        return "training"
    if any(w in q for w in ["حجز", "booking", "ميعاد", "appointment", "وحدة", "unit", "خدمة", "service", "شهادة", "certificate"]):
        return "service_info"
    if any(w in q for w in ["اخبار", "news", "صحافة", "press", "فعاليات", "events", "معرض", "exhibition"]):
        return "news"
    return "general"

def inject_dynamic_links(text: str, docs: List[Dict]) -> str:
    if not docs or not text:
        return text
    sorted_docs = sorted(docs, key=lambda x: len(x.get('title', '')), reverse=True)
    for doc in sorted_docs:
        url = doc.get('url', '')
        title = doc.get('title', '')
        if not url or not title or "categories" in url:
            continue
        safe_title = re.escape(title.strip())
        try:
            pattern = re.compile(f"(?<!\\[){safe_title}(?!\\])", re.IGNORECASE)
            if pattern.search(text):
                text = pattern.sub(f"[{title}]({url})", text, count=1)
        except Exception as e:
            logger.warning(f"Link injection failed for '{title}': {e}")
    return text

def fix_broken_links(text: str) -> str:
    if not text:
        return ""
    url_map = {
        "booking": "https://www.goeic.gov.eg/en/reserv-online",
        "حجز": "https://www.goeic.gov.eg/ar/reserv-online",
        "training": "https://www.goeic.gov.eg/en/training-courses/categories",
        "تدريب": "https://www.goeic.gov.eg/ar/training-courses/categories",
        "events": "https://www.goeic.gov.eg/en/media-center/events",
        "فعاليات": "https://www.goeic.gov.eg/ar/media-center/events",
        "jobs": "https://www.goeic.gov.eg/en/media-center/jobs",
        "وظائف": "https://www.goeic.gov.eg/ar/media-center/jobs",
        "news": "https://www.goeic.gov.eg/en/media-center/news",
        "اخبار": "https://www.goeic.gov.eg/ar/media-center/news",
    }
    def replace_bad_link(match):
        label = match.group(1)
        for key, url in url_map.items():
            if key in label.lower() or label.lower() in key:
                return f"[{label}]({url})"
        return f"[{label}](https://www.goeic.gov.eg/ar/about-us/callUs)"
    text = re.sub(r'\[(.*?)\]\((?:undefined|\[object Object\]|null|None)\)', replace_bad_link, text, flags=re.IGNORECASE)
    text = re.sub(r'\bundefined\b', '[الرابط الرسمي](https://www.goeic.gov.eg)', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*[\(\[]Source\s*\d+[\)\]]', '', text, flags=re.IGNORECASE)
    text = text.replace("•", "\n•").replace("- ", "\n- ").replace("  ", " ")
    return text.strip()

def get_embedding(text: str):
    try:
        if local_embedder:
            if len(text) > 2000:
                text = text[:2000]
            return local_embedder.encode(text, convert_to_tensor=False).tolist()
        else:
            logger.error("❌ Local embedder not initialized!")
            return None
    except Exception as e:
        logger.error(f"❌ Local Embedding Error: {e}")
        return None

def smart_rerank(results: List[Dict], intent: str, query: str) -> List[Dict]:
    scored_results = []
    numbers_in_query = re.findall(r'\d+', query)
    query_lower = query.lower()
    for doc in results:
        base_score = (doc.get('score', 0) or 0.0) * 0.4
        source_type = doc.get('source_type', 'web')
        url = doc.get('url', '').lower()
        if intent == "jobs" and "jobs" in url:
            data_weight = LOGICAL_CLASS_WEIGHTS["jobs"]
        elif "media-center" in url or "news" in url:
            data_weight = 1.0 if intent == "news" else LOGICAL_CLASS_WEIGHTS.get("news", 0.5)
        else:
            data_weight = LOGICAL_CLASS_WEIGHTS.get(source_type, 0.5)
        final_score = base_score + data_weight
        for num in numbers_in_query:
            if num in doc.get('title', ''):
                final_score += 0.6
            elif num in doc.get('content', ''):
                final_score += 0.3
        if intent == "training" and any(x in url for x in ["training", "course", "دورة"]):
            final_score += 0.8
        if intent == "service_info" and "booking" in url:
            final_score += 0.6
        if intent == "jobs" and "jobs" in url:
            final_score += 1.0
        content = doc.get('content', '')
        title = doc.get('title', '')
        if len(content) < 100 and query_lower not in title.lower():
            final_score -= 5.0
        doc['final_score'] = final_score
        scored_results.append(doc)
    scored_results.sort(key=lambda x: x['final_score'], reverse=True)
    return scored_results

# ═══════════════════════════════════════════════════════════════
# YOUR EXACT PROFESSIONAL PROMPT (PRESERVED)
# ═══════════════════════════════════════════════════════════════
def build_professional_prompt(docs: List[Dict], lang: str, intent: str, query: str) -> str:
    """YOUR EXACT PROFESSIONAL PROMPT - UNCHANGED"""
    context = ""
    for i, doc in enumerate(docs[:8], 1):
        title = doc.get('title', 'Untitled').replace(" - GOEIC", "").strip()
        url = doc.get('url', '')
        content = doc.get('content', '')[:5000]

        context += f"""
════════════════════════════════════════════════════════════════
📄 SOURCE {i}: {title}
🔗 URL: {url}
📝 CONTENT:
{content}
════════════════════════════════════════════════════════════════
"""

    lang_instruction = {
        "ar": "You MUST respond in Arabic (العربية) only. Use professional, formal Arabic.",
        "en": "You MUST respond in English only. Use professional, formal English.",
        "fr": "You MUST respond in French only. Use professional, formal French."
    }.get(lang, "Respond in Arabic")

    intent_instructions = ""

    if intent == "jobs":
        intent_instructions = """
🔴 CRITICAL - JOB VACANCIES HANDLING (LIVE DATA ONLY):
1. The context contains LIVE JOBS DATA scraped directly from the website in real-time
2. This is NOT database data - it's FRESH, CURRENT job listings from today
3. Extract ALL job details: title, description, requirements, deadline, application link
4. Format each job clearly with:
   - Job title as header
   - Description/requirements
   - Application deadline (if available)
   - Application link (if available)
5. If the LIVE JOBS DATA says "no vacancies", clearly state there are currently no job openings
6. Always include the jobs page link for users to apply: https://www.goeic.gov.eg/ar/media-center/jobs
7. DO NOT say "visit the website" - provide the actual job details from the LIVE JOBS DATA in the context
8. The data in the context is from TODAY, so trust it completely

IMPORTANT: This is real-time data, NOT old database information!
"""

    elif intent == "training":
        intent_instructions = """
🎓 TRAINING COURSES HANDLING:
1. List ALL available courses with their details from training data
2. Include: course name, duration, cost, prerequisites, registration deadline
3. Provide direct registration links for each course important if available as make easy to get the details
4. Do NOT provide the general 'reserv-online' link for training; use the specific training links provided.
5. Add this at the end: "[عرض كافة البرامج التدريبية](https://www.goeic.gov.eg/ar/training-courses/categories)"
"""

    elif intent == "service_info":
        intent_instructions = """
📋 SERVICE INFORMATION HANDLING:
1. Provide complete step-by-step procedures
2. List ALL required documents
3. Include fees, processing time, and location details
4. If booking is required, ask user to select branch first, then provide: https://www.goeic.gov.eg/ar/reserv-online
5. DO NOT say "check the website" - extract and provide the details
"""

    elif intent == "legal_decision":
        intent_instructions = """
⚖️ LEGAL DECISIONS HANDLING:
1. Provide the exact decision/law number if available
2. Include effective date, scope, and key provisions
3. Quote relevant articles or sections
4. Cite the official source document
"""

    prompt = f"""
═══════════════════════════════════════════════════════════════
🏛️ GOEIC ENTERPRISE AI ASSISTANT - PROFESSIONAL PROMPT
═══════════════════════════════════════════════════════════════

You are an expert consultant for GOEIC (General Organization for Export & Import Control - الهيئة العامة للرقابة على الصادرات والواردات), Egypt's official export/import regulatory authority.

🎯 USER QUERY:
{query}

🌐 LANGUAGE REQUIREMENT:
{lang_instruction}

📂 DETECTED INTENT: {intent.upper()}

{intent_instructions}

📚 KNOWLEDGE CONTEXT:
{context}

═══════════════════════════════════════════════════════════════
🎯 RESPONSE REQUIREMENTS (NON-NEGOTIABLE):
═══════════════════════════════════════════════════════════════

1. **COMPREHENSIVENESS:**
   - Extract EVERY relevant detail from the context
   - Do NOT summarize - provide complete information
   - Include all steps, requirements, fees, documents, dates, and links

2. **NO LAZY ANSWERS:**
   - NEVER say "visit the website for more details"
   - NEVER say "check the link"
   - PROVE you know the answer by writing details directly

3. **FORMATTING:**
   - Use clear markdown formatting
   - Use **bold** for headers and important terms
   - Use bullet points for lists
   - Use numbered lists for sequential steps
   - Add line breaks for readability

4. **LINKS:**
   - If a specific URL exists in context, link the relevant text
   - Ensure links are properly formatted: [Text](URL)
   - Do NOT create broken links

5. **SCOPE BOUNDARIES:**
   - ONLY answer questions about GOEIC services, regulations, and activities
   - REFUSE questions about: cooking, sports, politics, entertainment, personal advice
   - For out-of-scope questions, politely decline and redirect to GOEIC topics

6. **ACCURACY:**
   - Base answers ONLY on the provided context
   - If information is not in context, clearly state this
   - Do NOT invent or hallucinate information

7. **SUGGESTIONS:**
   - Provide 3 relevant follow-up questions users might ask
   - Questions should be natural and helpful

═══════════════════════════════════════════════════════════════
📤 OUTPUT FORMAT (STRICT JSON):
═══════════════════════════════════════════════════════════════

{{
    "answer_text": "Your detailed, well-formatted answer in {lang}...",
    "suggestions": [
        "Follow-up question 1 in {lang}",
        "Follow-up question 2 in {lang}",
        "Follow-up question 3 in {lang}"
    ]
}}

⚠️ CRITICAL RULES:
- Output ONLY the JSON object above
- NO markdown code blocks (```)
- NO explanations before or after JSON
- Ensure valid JSON syntax (escape quotes in text)
- Keep answer_text and suggestions as single-line strings (use \\n for line breaks)

BEGIN YOUR RESPONSE NOW:
"""
    return prompt

# ═══════════════════════════════════════════════════════════════
# TRANSLATION — detect & translate via local Ollama
# ═══════════════════════════════════════════════════════════════

# Patterns for each target language — covers Arabic, English, French phrasing
TRANSLATION_PATTERNS = {
    "en": [
        r"ترجم[ة]?\s*(إلى|الى|ل)\s*(الإنجليزية|الانجليزية|إنجليزي|انجليزي)",
        r"translate\s*(to|into)\s*(english|en)\b",
        r"traduire?\s*(en|vers)\s*(anglais|english)",
        r"بالإنجليزي", r"بالانجليزي",
    ],
    "ar": [
        r"ترجم[ة]?\s*(إلى|الى|ل)\s*(العربية|عربي)",
        r"translate\s*(to|into)\s*(arabic|ar)\b",
        r"traduire?\s*(en|vers)\s*(arabe|arabic)",
        r"بالعربي", r"بالعربية",
    ],
    "fr": [
        r"ترجم[ة]?\s*(إلى|الى|ل)\s*(الفرنسية|فرنسي)",
        r"translate\s*(to|into)\s*(french|fr)\b",
        r"traduire?\s*(en|vers)\s*(français|francais|french)",
        r"بالفرنسي", r"بالفرنسية",
    ],
}

def detect_translation_request(query: str) -> Optional[str]:
    """
    Returns the target language code ('en', 'ar', 'fr') if the user is asking
    to translate the previous response, otherwise returns None.
    """
    q = query.strip()
    for target_lang, patterns in TRANSLATION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, q, re.IGNORECASE):
                return target_lang
    return None

async def translate_text_with_ollama(text: str, target_lang: str) -> str:
    """
    Translates `text` to `target_lang` using the local Ollama model.
    Strict prompt: translate only, no commentary, no new content.
    """
    global ollama_client

    if not ollama_client:
        return text  # fallback: return original if Ollama is not ready

    lang_names = {"en": "English", "ar": "Arabic (فصحى)", "fr": "French"}
    lang_name = lang_names.get(target_lang, "English")

    # Clean markdown slightly so the model focuses on text, not symbols
    clean_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # [label](url) → label
    clean_text = re.sub(r'[*#`]', '', clean_text).strip()
    clean_text = clean_text[:4000]  # keep within context limits

    translation_prompt = f"""You are a professional Arabic/English/French translator.

TASK: Translate the following text into {lang_name}.

STRICT RULES:
- Output ONLY the translated text
- Do NOT add any explanation, commentary, or preamble
- Do NOT say "Here is the translation" or similar
- Preserve paragraph structure and line breaks
- Keep any proper nouns (GOEIC, Egypt, etc.) as-is
- Keep URLs and links intact

TEXT TO TRANSLATE:
{clean_text}

TRANSLATION:"""

    try:
        logger.info(f"🌐 Translating to {target_lang} ({len(clean_text)} chars)...")
        start = time.time()

        response = await ollama_client.post("/api/generate", json={
            "model": OLLAMA_MODEL,
            "prompt": translation_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 2048,
                "num_ctx": 4096,
                "num_gpu": -1,
                "repeat_penalty": 1.1,
                "stop": ["TEXT TO TRANSLATE:", "TASK:", "RULES:"]
            }
        }, timeout=120.0)

        elapsed = time.time() - start
        logger.info(f"✅ Translation done in {elapsed:.1f}s")

        if response.status_code == 200:
            raw = response.json().get("response", "").strip()
            # Strip any accidental preamble lines the model may add
            lines = raw.splitlines()
            # Remove lines that look like meta-commentary
            clean_lines = [
                l for l in lines
                if not re.match(r'^(here\s+is|translation:|ترجمة:|voici)', l, re.IGNORECASE)
            ]
            return "\n".join(clean_lines).strip() or text

        logger.warning("⚠️ Translation: Ollama non-200 response, returning original")
        return text

    except Exception as e:
        logger.error(f"❌ Translation error: {e}")
        return text  # safe fallback

# ═══════════════════════════════════════════════════════════════
# ROBUST JSON PARSER
# ═══════════════════════════════════════════════════════════════
def robust_json_parse(raw_text: str) -> Optional[Dict]:
    if not raw_text:
        return None

    cleaned = re.sub(r'```json\s*', '', raw_text, flags=re.IGNORECASE)
    cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()

    json_match = re.search(r'\{[\s\S]*\}', cleaned)
    if json_match:
        cleaned = json_match.group(0)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    try:
        fixed = re.sub(
            r'("(?:answer_text|suggestions)"\s*:\s*")(.*?)("(?:\s*[,}]|$))',
            lambda m: m.group(1) + m.group(2).replace('\n', '\\n').replace('\r', '') + m.group(3),
            cleaned,
            flags=re.DOTALL
        )
        return json.loads(fixed)
    except:
        pass

    try:
        answer_match = re.search(r'"answer_text"\s*:\s*"([\s\S]*?)(?:"\s*,|"\s*\})', cleaned, re.DOTALL)
        suggestions_match = re.search(r'"suggestions"\s*:\s*\[([\s\S]*?)\]', cleaned, re.DOTALL)

        if answer_match:
            answer = answer_match.group(1)
            answer = answer.replace('\\n', '\n').replace('\\"', '"')

            suggestions = []
            if suggestions_match:
                sugg_text = suggestions_match.group(1)
                suggestions = re.findall(r'"([^"]*)"', sugg_text)

            return {
                "answer_text": answer,
                "suggestions": suggestions[:3]
            }
    except Exception as e:
        logger.debug(f"Manual extraction failed: {e}")

    logger.warning(f"⚠️ All JSON parse strategies failed. Raw length: {len(raw_text)}")
    display_text = raw_text[:3000] if raw_text else "عذراً، لم أتمكن من معالجة الرد."

    return {
        "answer_text": display_text,
        "suggestions": []
    }

# ═══════════════════════════════════════════════════════════════
# OLLAMA GENERATION
# ═══════════════════════════════════════════════════════════════
async def generate_answer_ollama(system_prompt: str, user_prompt: str, history: List) -> Optional[Dict]:
    global ollama_client

    if not ollama_client:
        logger.error("❌ Ollama client not initialized")
        return None

    history_text = ""
    if history:
        recent = history[-2:]
        history_text = "📜 HISTORY:\n" + "\n".join([
            f"{m.role.upper()}: {m.content}" for m in recent
        ]) + "\n\n"

    final_prompt = f"{history_text}QUERY: {user_prompt}"

    json_instruction = """

⚠️ CRITICAL: You MUST respond with ONLY valid JSON. No markdown, no extra text.
Format:
{
    "answer_text": "your answer...",
    "suggestions": ["q1", "q2", "q3"]
}
"""

    full_prompt = f"{system_prompt}{json_instruction}\n\n{final_prompt}"

    try:
        logger.info(f"🤖 Sending to Ollama ({len(full_prompt)} chars)")
        start_time = time.time()

        response = await ollama_client.post("/api/generate", json={
            "model": OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 2048,
                "num_ctx": 8192,
                "num_gpu": -1,
                "num_thread": 6,
                "repeat_penalty": 1.1,
                "top_p": 0.9,
                "top_k": 40,
                "stop": ["Human:", "User:", "QUERY:"]
            }
        }, timeout=180.0)

        elapsed = time.time() - start_time
        logger.info(f"⏱️ Ollama response: {elapsed:.1f}s")

        if response.status_code == 200:
            result = response.json()
            raw_text = result.get("response", "")

            parsed = robust_json_parse(raw_text)
            if parsed and parsed.get("answer_text"):
                logger.info(f"✅ JSON parsed successfully")
                return parsed
            else:
                logger.warning(f"⚠️ Parse returned empty, using raw")
                return {"answer_text": raw_text, "suggestions": []}
        else:
            logger.error(f"❌ Ollama error: {response.status_code}")
            return None

    except asyncio.TimeoutError:
        logger.error("❌ Ollama timeout (180s)")
        return None
    except Exception as e:
        logger.error(f"❌ Ollama Error: {e}")
        return None

# ═══════════════════════════════════════════════════════════════
# MAIN QUERY PROCESSING
# ═══════════════════════════════════════════════════════════════
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    language: str = "ar"
    history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    answer: str
    sources: list
    suggestions: list = []
    translated: bool = False   # True when response is a translation of previous answer

async def process_user_query(
    question: str,
    language: str,
    history: List[ChatMessage]
) -> tuple:
    """
    Main query processing — 100% OFFLINE.
    Returns: (answer, sources, suggestions, translated_flag)

    Translation is intercepted BEFORE the RAG pipeline:
      - If user asks to translate → find last assistant message in history,
        translate it with Ollama, return immediately (no vector search).
      - Otherwise → normal RAG flow.
    """

    # ── TRANSLATION INTERCEPT ─────────────────────────────────
    target_lang = detect_translation_request(question)
    if target_lang:
        # Find last assistant message in conversation history
        last_assistant_msg = None
        for msg in reversed(history):
            if msg.role == "assistant":
                last_assistant_msg = msg.content
                break

        if last_assistant_msg:
            logger.info(f"🌐 Translation request detected → target: {target_lang}")
            translated = await translate_text_with_ollama(last_assistant_msg, target_lang)
            lang_labels = {
                "en": "🌐 **Translated to English:**\n\n",
                "ar": "🌐 **مترجم إلى العربية:**\n\n",
                "fr": "🌐 **Traduit en français:**\n\n",
            }
            label = lang_labels.get(target_lang, "🌐 **Translation:**\n\n")
            return label + translated, [], [], True
        else:
            # No previous message to translate
            no_history = {
                "ar": "لا يوجد رد سابق لترجمته. يرجى طرح سؤالك أولاً.",
                "en": "There is no previous response to translate. Please ask a question first.",
                "fr": "Il n'y a pas de réponse précédente à traduire. Veuillez d'abord poser une question.",
            }
            return no_history.get(language, no_history["ar"]), [], [], False

    # ── NORMAL RAG FLOW ───────────────────────────────────────
    intent = detect_intent(question)
    final_docs = []

    if intent == "jobs":
        live_jobs = await fetch_live_jobs_async(language)
        if live_jobs:
            final_docs.append({
                'title': f"Live Job Vacancies ({language.upper()})",
                'content': live_jobs,
                'url': f"https://www.goeic.gov.eg/{language}/media-center/jobs",
                'source_type': 'jobs',
                'score': 2.0
            })
        final_docs.append({
            'title': "Official GOEIC Links",
            'content': FALLBACK_LINKS,
            'url': "https://www.goeic.gov.eg",
            'source_type': 'golden_source',
            'score': 0.5
        })

    elif intent in ["training", "service_info", "general"]:
        final_docs.append({
            'title': "Official GOEIC Links",
            'content': FALLBACK_LINKS,
            'url': "https://www.goeic.gov.eg",
            'source_type': 'golden_source',
            'score': 0.5
        })

    if collection and intent != "jobs":
        vector = get_embedding(question)
        if vector:
            try:
                filters = (
                    Filter.by_property("language").equal(language) &
                    Filter.by_property("chunk_type").equal("child")
                )

                res = collection.query.hybrid(
                    query=question,
                    vector=vector,
                    limit=20,
                    alpha=0.5,
                    filters=filters,
                    return_metadata=MetadataQuery(score=True)
                )

                parent_scores = {}
                for obj in res.objects:
                    pid = obj.properties.get('parent_id')
                    if not pid:
                        continue
                    score = obj.metadata.score or 0
                    if pid not in parent_scores or score > parent_scores[pid]['score']:
                        parent_scores[pid] = {'score': score, 'obj': obj}

                top_pids = list(parent_scores.keys())[:12]
                if top_pids:
                    fetch_res = collection.query.fetch_objects(
                        filters=(
                            Filter.by_property("parent_id").contains_any(top_pids) &
                            Filter.by_property("chunk_type").equal("parent")
                        ),
                        limit=12
                    )

                    for obj in fetch_res.objects:
                        url = obj.properties.get('url', '')
                        if "ask-us" in url or "callUs" in url:
                            continue

                        final_docs.append({
                            'title': obj.properties.get('title', 'Untitled'),
                            'content': obj.properties.get('content', '')[:15000],
                            'url': url,
                            'source_type': obj.properties.get('source_type', 'web'),
                            'score': parent_scores.get(obj.properties.get('parent_id'), {}).get('score', 0.5)
                        })

                    logger.info(f"✅ Retrieved {len(final_docs)} docs from Weaviate")

            except Exception as e:
                logger.error(f"Weaviate Query Error: {e}")

    ranked_results = smart_rerank(final_docs, intent, question)

    is_irrelevant = False
    if not ranked_results:
        is_irrelevant = True
    elif len(ranked_results) > 0 and ranked_results[0].get('final_score', 0) < -2.0:
        is_irrelevant = True

    banned = ["recipe", "cook", "football", "match", "movie", "song", "game"]
    if any(b in question.lower() for b in banned):
        is_irrelevant = True

    if is_irrelevant:
        if language == "ar":
            return "عذراً، هذا السؤال خارج نطاق تخصصي. أنا مساعد رقمي متخصص في خدمات الهيئة العامة للرقابة على الصادرات والواردات فقط.", [], [], False
        elif language == "en":
            return "Sorry, this question is outside my expertise. I'm a digital assistant specialized only in GOEIC services.", [], [], False
        else:
            return "Désolé, cette question est hors de mon domaine. Je suis spécialisé uniquement dans les services GOEIC.", [], [], False

    prompt = build_professional_prompt(ranked_results, language, intent, question)
    response_data = await generate_answer_ollama(prompt, question, history)

    if not response_data:
        if language == "ar":
            return "عذراً، النظام مشغول حالياً. يرجى المحاولة مرة أخرى.", [], [], False
        elif language == "en":
            return "Sorry, the system is busy. Please try again.", [], [], False
        else:
            return "Désolé, le système est occupé. Veuillez réessayer.", [], [], False

    raw_answer = response_data.get("answer_text", "")
    suggestions = response_data.get("suggestions", [])

    answer = inject_dynamic_links(raw_answer, ranked_results)
    answer = fix_broken_links(answer)
    answer = re.sub(r'\n.*التسجيل.*', '', answer)

    final_sources = []
    seen_urls = set()

    for doc in ranked_results[:6]:
        url = doc.get('url', '')
        title = doc.get('title', 'Untitled')
        if url in seen_urls or doc.get('source_type') == 'golden_source':
            continue
        final_sources.append({
            "title": title[:60],
            "url": url,
            "type": "link"
        })
        seen_urls.add(url)

    return answer, final_sources, suggestions, False

# ═══════════════════════════════════════════════════════════════
# AUTHENTICATION
# ═══════════════════════════════════════════════════════════════
def require_admin(request: Request):
    token = request.cookies.get("session_token")
    if token and token in active_sessions:
        return active_sessions[token]
    raise HTTPException(status_code=401, detail="Not Authenticated")

@app.post("/api/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE username=?", (form_data.username,))
    row = c.fetchone()
    conn.close()
    if not row or not verify_password(form_data.password, row[0]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    token = secrets.token_hex(16)
    active_sessions[token] = form_data.username
    response = JSONResponse(content={"message": "Login successful"})
    response.set_cookie(key="session_token", value=token, httponly=True, samesite='lax', path="/")
    logger.info(f"✅ User logged in: {form_data.username}")
    return response

@app.post("/api/logout")
def logout(response: Response, request: Request):
    token = request.cookies.get("session_token")
    if token and token in active_sessions:
        del active_sessions[token]
    response.delete_cookie("session_token", path="/")
    return {"message": "Logged out successfully"}

@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request):
    token = request.cookies.get("session_token")
    file_path = "public/dashboard.html" if (token and token in active_sessions) else "public/login.html"
    response = FileResponse(file_path)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response

@app.get("/api/stats")
def get_stats(page: int = 1, page_size: int = 20, country: Optional[str] = None,
              start_date: Optional[str] = None, end_date: Optional[str] = None,
              ip_address: Optional[str] = None, user: str = Depends(require_admin)):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    where_clauses = []
    params = []
    if country:
        where_clauses.append("country = ?"); params.append(country)
    if start_date:
        where_clauses.append("timestamp >= ?"); params.append(f"{start_date} 00:00:00")
    if end_date:
        where_clauses.append("timestamp <= ?"); params.append(f"{end_date} 23:59:59")
    if ip_address:
        where_clauses.append("ip_address LIKE ?"); params.append(f"%{ip_address}%")
    where_str = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    try:
        total_rows = c.execute(f"SELECT COUNT(*) FROM chat_logs {where_str}", params).fetchone()[0]
        offset = (page - 1) * page_size
        logs = c.execute(f"SELECT * FROM chat_logs {where_str} ORDER BY id DESC LIMIT ? OFFSET ?", params + [page_size, offset]).fetchall()
        stats = {
            'devices': [list(r) for r in c.execute(f"SELECT device_type, COUNT(*) FROM chat_logs {where_str} GROUP BY device_type", params).fetchall()],
            'countries_list': [list(r) for r in c.execute("SELECT DISTINCT country FROM chat_logs WHERE country IS NOT NULL").fetchall()],
            'countries_chart': [list(r) for r in c.execute(f"SELECT country, COUNT(*) FROM chat_logs {where_str} GROUP BY country ORDER BY COUNT(*) DESC LIMIT 5", params).fetchall()],
            'languages': [list(r) for r in c.execute(f"SELECT language, COUNT(*) FROM chat_logs {where_str} GROUP BY language", params).fetchall()],
            'intents': [list(r) for r in c.execute(f"SELECT intent, COUNT(*) FROM chat_logs {where_str} GROUP BY intent ORDER BY COUNT(*) DESC LIMIT 10", params).fetchall()],
            'recent_logs': [dict(row) for row in logs],
            'total_pages': (total_rows + page_size - 1) // page_size,
            'current_page': page,
            'total_records': total_rows,
            'avg_time': c.execute(f"SELECT AVG(response_time) FROM chat_logs {where_str}", params).fetchone()[0] or 0
        }
        return stats
    except Exception as e:
        logger.error(f"Stats Error: {e}")
        return {"error": str(e)}
    finally:
        conn.close()

@app.get("/api/export")
def export_stats(country: Optional[str] = None, start_date: Optional[str] = None,
                 end_date: Optional[str] = None, user: str = Depends(require_admin)):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    where_clauses = []; params = []
    if country: where_clauses.append("country = ?"); params.append(country)
    if start_date: where_clauses.append("timestamp >= ?"); params.append(f"{start_date} 00:00:00")
    if end_date: where_clauses.append("timestamp <= ?"); params.append(f"{end_date} 23:59:59")
    where_str = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    rows = c.execute(f"SELECT timestamp, device_type, country, language, intent, query, response_time FROM chat_logs {where_str} ORDER BY id DESC", params).fetchall()
    conn.close()
    output = io.StringIO()
    output.write('\ufeff')
    writer = csv.writer(output)
    writer.writerow(['Timestamp', 'Device', 'Country', 'Language', 'Intent', 'Query', 'Response Time'])
    writer.writerows(rows)
    return Response(content=output.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": "attachment; filename=analytics.csv"})

class NewUser(BaseModel):
    username: str
    password: str

@app.post("/api/users/add")
def add_user(new_user: NewUser, user: str = Depends(require_admin)):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        hashed = get_password_hash(new_user.password)
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (new_user.username, hashed))
        conn.commit()
        return {"message": "User created"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username exists")
    finally:
        conn.close()

@app.put("/api/users/change-password")
def change_password(user_data: NewUser, user: str = Depends(require_admin)):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    hashed = get_password_hash(user_data.password)
    c.execute("UPDATE users SET password_hash=? WHERE username=?", (hashed, user_data.username))
    conn.commit()
    conn.close()
    return {"message": "Password updated"}

# ═══════════════════════════════════════════════════════════════
# PUBLIC ENDPOINTS
# ═══════════════════════════════════════════════════════════════
public_path = Path(__file__).parent / "public"
if public_path.exists():
    app.mount("/public", StaticFiles(directory=public_path), name="public")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return FileResponse(public_path / "index.html")

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: Request, req: ChatRequest):
    start_time = time.time()

    answer, sources, suggestions, translated = await process_user_query(
        req.question, req.language, req.history
    )

    intent = detect_intent(req.question)
    response_time = time.time() - start_time
    log_to_db(request.state.request_id, request.state.client_ip,
              request.state.device_type, intent, req.language,
              req.question, response_time, 200)

    return ChatResponse(answer=answer, sources=sources, suggestions=suggestions, translated=translated)

@app.post("/api/voice")
async def voice_endpoint(request: Request, file: UploadFile = File(...), language: str = Form("ar")):
    """Voice chat with LOCAL Whisper + Edge TTS — Cross-platform"""
    if not WHISPER_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "Whisper not installed. Run: pip install openai-whisper ffmpeg-python"}
        )

    init_whisper()
    if not whisper_model:
        return JSONResponse(
            status_code=503,
            content={"error": "Whisper model failed to load. Check FFmpeg is installed."}
        )

    import tempfile
    tmp_dir = Path(tempfile.gettempdir())
    temp_audio = tmp_dir / f"goeic_{uuid.uuid4()}.webm"
    output_audio = tmp_dir / f"goeic_{uuid.uuid4()}.mp3"

    with open(temp_audio, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        logger.info("🎤 Transcribing with local Whisper...")
        result = whisper_model.transcribe(str(temp_audio), language=language)
        user_text = result["text"].strip()
        logger.info(f"🎤 Transcription: {user_text[:50]}...")

        start_time = time.time()
        answer, sources, suggestions, translated = await process_user_query(user_text, language, [])

        device = getattr(request.state, 'device_type', 'Unknown')
        log_to_db(getattr(request.state, 'request_id', 'unknown'),
                  getattr(request.state, 'client_ip', 'unknown'),
                  f"🎤 Voice ({device})", "voice_query", language, user_text,
                  time.time() - start_time, 200)

        clean_answer = re.sub(r'[*#\-\•]', '', re.sub(r'\[.*?\]\(.*?\)', '', answer))[:4000]
        voice_map = {"ar": "ar-EG-SalmaNeural", "en": "en-US-AriaNeural", "fr": "fr-FR-DeniseNeural"}
        voice = voice_map.get(language, "ar-EG-SalmaNeural")

        communicate = edge_tts.Communicate(clean_answer, voice)
        await communicate.save(str(output_audio))

        with open(output_audio, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')

        return {
            "question": user_text,
            "answer": answer,
            "audio": audio_base64,
            "sources": sources,
            "suggestions": suggestions,
            "translated": translated
        }
    except Exception as e:
        logger.error(f"Voice error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Voice failed: {str(e)}"})
    finally:
        for fp in [temp_audio, output_audio]:
            try:
                if fp.exists():
                    fp.unlink()
            except:
                pass

@app.get("/health")
async def health_check():
    gpu_info = "N/A"
    if torch.cuda.is_available():
        gpu_info = f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory // 1024**3}GB)"

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mode": "FULLY OFFLINE",
        "llm": f"Ollama ({OLLAMA_MODEL})",
        "embeddings": f"Local ({EMBEDDING_MODEL})",
        "gpu": gpu_info,
        "whisper": "Available" if WHISPER_AVAILABLE else "Not installed",
        "weaviate": client is not None,
        "ollama_ready": ollama_client is not None
    }

# ═══════════════════════════════════════════════════════════════
# ADMIN LOGS — REST ENDPOINTS (No WebSocket, no server load)
# ═══════════════════════════════════════════════════════════════

@app.get("/api/admin/logs/view")
async def view_logs(lines: int = 200, user: str = Depends(require_admin)):
    """
    Returns last N lines of the log file as JSON.
    Zero persistent connection — called only when admin clicks 'View Logs'.
    No server load between requests.
    """
    lines = max(10, min(lines, 5000))  # clamp: 10 ≤ lines ≤ 5000

    if not LOG_FILE.exists():
        return {"lines": [], "total": 0, "file": str(LOG_FILE)}

    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()

        last_lines = all_lines[-lines:]
        return {
            "lines": [l.rstrip("\n") for l in last_lines],
            "total": len(all_lines),
            "showing": len(last_lines),
            "file": str(LOG_FILE),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Log view error: {e}")
        raise HTTPException(status_code=500, detail=f"Could not read log file: {e}")


@app.get("/api/admin/logs/download")
async def download_logs(lines: int = 1000, user: str = Depends(require_admin)):
    """
    Streams last N lines of the log file as a downloadable .txt attachment.
    Triggers a file download in the browser — no streaming connection kept open.
    """
    lines = max(10, min(lines, 50000))  # clamp: 10 ≤ lines ≤ 50000

    if not LOG_FILE.exists():
        raise HTTPException(status_code=404, detail="Log file not found")

    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()

        last_lines = all_lines[-lines:]
        content = "".join(last_lines)

        filename = f"goeic_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        return Response(
            content=content,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        logger.error(f"Log download error: {e}")
        raise HTTPException(status_code=500, detail=f"Could not read log file: {e}")

# ═══════════════════════════════════════════════════════════════
# ADMIN PANEL - DATA MANAGEMENT ENDPOINTS
# ═══════════════════════════════════════════════════════════════

scraping_status = {"running": False, "progress": 0, "message": "Idle"}
upload_status = {"running": False, "progress": 0, "message": "Idle"}

@app.post("/api/admin/scrape-website")
async def trigger_website_scraping(background_tasks: BackgroundTasks, user: str = Depends(require_admin)):
    global scraping_status

    if scraping_status["running"]:
        raise HTTPException(status_code=409, detail="Scraping already in progress")

    scraping_status = {"running": True, "progress": 0, "message": "Starting scraper..."}
    background_tasks.add_task(run_smart_scraper)

    logger.info(f"🌐 Website scraping triggered by: {user}")

    return {
        "message": "Website scraping started in background",
        "status": "running",
        "check_progress_at": "/api/admin/scraping-status"
    }

async def run_smart_scraper():
    global scraping_status

    try:
        scraping_status.update({"message": "🔌 Connecting to database...", "progress": 5})

        def scraper_worker():
            try:
                import sys, os, asyncio
                sys.path.insert(0, os.getcwd())

                import weaviate as _weaviate
                from weaviate.classes.init import AdditionalConfig, Timeout
                from weaviate.classes.query import Filter as _Filter
                from smart_scraper_offline import (
                    SmartGOEICScraper,
                    DuplicateDetector,
                    index_to_weaviate,
                )

                client = _weaviate.connect_to_local(
                    additional_config=AdditionalConfig(
                        timeout=Timeout(init=60, query=180, insert=180)
                    )
                )
                collection = client.collections.get("GOEIC_Knowledge_Base_V2")
                logger.info("✅ Weaviate connected for scraping task")

                scraping_status.update({
                    "message": "🧹 Step 1/4 — Pre-scrape cleanup (removing any existing duplicates)...",
                    "progress": 10
                })
                logger.info("🧹 Running pre-scrape duplicate cleanup...")

                iterator = collection.iterator(
                    return_properties=["url", "chunk_type"],
                    include_vector=False,
                )
                seen_urls_cleanup: dict = {}
                dup_uuids: list = []
                for obj in iterator:
                    props = obj.properties or {}
                    if props.get("chunk_type") != "parent":
                        continue
                    url = props.get("url")
                    uid = str(obj.uuid)
                    if not url:
                        continue
                    if url in seen_urls_cleanup:
                        dup_uuids.append(uid)
                    else:
                        seen_urls_cleanup[url] = uid

                if dup_uuids:
                    logger.info(f"🗑️ Cleanup: removing {len(dup_uuids)} duplicate parents...")
                    BATCH = 50
                    for i in range(0, len(dup_uuids), BATCH):
                        batch = dup_uuids[i:i + BATCH]
                        collection.data.delete_many(
                            where=_Filter.by_property("parent_id").contains_any(batch)
                        )
                        collection.data.delete_many(
                            where=_Filter.by_id().contains_any(batch)
                        )
                    logger.info(f"✅ Cleanup done: removed {len(dup_uuids)} duplicates")
                    scraping_status["message"] = (
                        f"🧹 Cleanup done — removed {len(dup_uuids)} duplicate entries"
                    )
                else:
                    logger.info("✅ Cleanup: database is already clean")
                    scraping_status["message"] = "🧹 Database clean — no duplicates found"

                scraping_status.update({
                    "message": "🔍 Step 2/4 — Loading existing URL index from database...",
                    "progress": 20
                })
                duplicate_detector = DuplicateDetector(client, "GOEIC_Knowledge_Base_V2")
                logger.info(
                    f"🔍 Duplicate detector ready: {len(duplicate_detector.existing_urls)} existing URLs"
                )

                scraping_status.update({
                    "message": "🕷️ Step 3/4 — Scraping website (filters: blacklist, junk, login, duplicates)...",
                    "progress": 30
                })
                scraper = SmartGOEICScraper(duplicate_detector)

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                new_data, updated_data = loop.run_until_complete(scraper.scrape())
                loop.close()

                s = scraper.stats
                logger.info(
                    f"🕷️ Scrape complete — "
                    f"new={s['new']} updated={s['updated']} "
                    f"skipped_junk={s['skipped_junk']} "
                    f"skipped_session_dup={s['skipped_in_session_dup']} "
                    f"errors={s['errors']}"
                )
                scraping_status.update({
                    "message": (
                        f"🕷️ Scraped — {s['new']} new, {s['updated']} updated, "
                        f"{s['skipped_junk']} junk filtered, {s['errors']} errors"
                    ),
                    "progress": 70
                })

                if new_data or updated_data:
                    scraping_status.update({
                        "message": f"💾 Step 4/4 — Indexing {len(new_data)} new + {len(updated_data)} updated pages...",
                        "progress": 80
                    })
                    index_to_weaviate(new_data, updated_data, collection)
                else:
                    logger.info("✅ No new or updated pages — database is up to date")

                client.close()

                summary = (
                    f"✅ Done — "
                    f"{s['new']} new, {s['updated']} updated, "
                    f"{s['skipped_junk']} junk removed, "
                    f"{s['skipped_in_session_dup']} session-dups removed"
                )
                scraping_status.update({"message": summary, "progress": 100})
                logger.info(f"🎉 Scraping pipeline complete: {summary}")
                return True

            except Exception as e:
                logger.error(f"❌ Scraper worker error: {e}", exc_info=True)
                scraping_status.update({
                    "message": f"❌ Error: {str(e)[:120]}",
                    "progress": 0
                })
                return False

        running_loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            success = await running_loop.run_in_executor(executor, scraper_worker)

        if not success:
            scraping_status["progress"] = 0

    except Exception as e:
        scraping_status.update({"message": f"Error: {str(e)}", "progress": 0})
        logger.error(f"❌ Scraping task error: {e}", exc_info=True)

    finally:
        scraping_status["running"] = False

@app.get("/api/admin/scraping-status")
async def get_scraping_status(user: str = Depends(require_admin)):
    return scraping_status

@app.post("/api/admin/upload-excel")
async def trigger_excel_upload(
    background_tasks: BackgroundTasks,
    excel_file: UploadFile = File(...),
    base_dir: str = Form("."),
    user: str = Depends(require_admin)
):
    global upload_status

    if upload_status["running"]:
        raise HTTPException(status_code=409, detail="Upload already in progress")

    excel_path = f"uploads/excel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    os.makedirs("uploads", exist_ok=True)

    with open(excel_path, "wb") as f:
        shutil.copyfileobj(excel_file.file, f)

    logger.info(f"📄 Excel uploaded: {excel_path} by {user}")

    upload_status = {"running": True, "progress": 0, "message": "Starting upload..."}
    background_tasks.add_task(run_smart_uploader, excel_path, base_dir)

    return {
        "message": "Excel upload started in background",
        "excel_file": excel_path,
        "status": "running",
        "check_progress_at": "/api/admin/upload-status"
    }

async def run_smart_uploader(excel_path: str, base_dir: str):
    global upload_status

    try:
        upload_status["message"] = "Reading Excel file..."
        upload_status["progress"] = 10

        def uploader_worker():
            try:
                logger.info(f"📊 Starting Excel upload: {excel_path}")

                import sys, os
                import pandas as pd
                sys.path.insert(0, os.getcwd())

                from smart_excel_uploader_offline import (
                    smart_upload_from_excel,
                    DocumentDuplicateDetector,
                    get_local_embedding,
                    generate_uuid,
                    clean_text
                )
                import weaviate
                from weaviate.classes.init import AdditionalConfig, Timeout
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                from docx import Document
                from tqdm import tqdm

                upload_status["message"] = "Connecting to database..."
                upload_status["progress"] = 20

                if not os.path.exists(excel_path):
                    raise FileNotFoundError(f"Excel not found: {excel_path}")

                df = pd.read_excel(excel_path).fillna("")
                logger.info(f"📊 Found {len(df)} rows in Excel")

                upload_status["message"] = f"Processing {len(df)} rows..."
                upload_status["progress"] = 30

                client = weaviate.connect_to_local(
                    additional_config=AdditionalConfig(
                        timeout=Timeout(init=60, query=180, insert=180)
                    )
                )
                collection = client.collections.get("GOEIC_Knowledge_Base_V2")

                dup_detector = DocumentDuplicateDetector(client, "GOEIC_Knowledge_Base_V2")

                upload_status["message"] = "Indexing documents..."
                upload_status["progress"] = 50

                child_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", "。", ".", " ", ""]
                )

                stats = {
                    'total': len(df),
                    'new': 0,
                    'updated': 0,
                    'skipped': 0,
                    'missing_files': 0,
                    'errors': 0
                }

                docs_to_delete = []

                with collection.batch.dynamic() as batch:
                    for index, row in df.iterrows():
                        try:
                            web_url = str(row['url']).strip()
                            title = str(row['title']).strip()
                            category = str(row['category']).strip()
                            lang = str(row.get('language', 'ar')).strip().lower()

                            parent_id = generate_uuid(web_url)

                            raw_path = str(row['path']).strip().replace('"', '').replace("'", "")
                            doc_path = os.path.join(base_dir, raw_path)

                            if not os.path.exists(doc_path):
                                stats['missing_files'] += 1
                                continue

                            doc = Document(doc_path)
                            full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                            full_text = clean_text(full_text)

                            if len(full_text) < 50:
                                stats['errors'] += 1
                                continue

                            is_dup, reason = dup_detector.is_duplicate(parent_id, full_text)

                            if is_dup and reason == "exact_duplicate":
                                stats['skipped'] += 1
                                continue

                            if reason == "content_updated":
                                docs_to_delete.append(parent_id)
                                stats['updated'] += 1
                            else:
                                stats['new'] += 1

                            batch.add_object(
                                properties={
                                    "content": full_text,
                                    "url": web_url,
                                    "title": title,
                                    "category": category,
                                    "language": lang,
                                    "source_type": "pdf_ocr",
                                    "chunk_type": "parent",
                                    "parent_id": parent_id
                                },
                                uuid=parent_id
                            )

                            chunks = child_splitter.split_text(full_text)
                            for i, chunk in enumerate(chunks):
                                vector = get_local_embedding(chunk)
                                if vector:
                                    chunk_id = generate_uuid(f"{web_url}_{i}")
                                    batch.add_object(
                                        properties={
                                            "content": chunk,
                                            "url": web_url,
                                            "title": title,
                                            "category": category,
                                            "language": lang,
                                            "source_type": "pdf_ocr",
                                            "chunk_type": "child",
                                            "parent_id": parent_id
                                        },
                                        vector=vector,
                                        uuid=chunk_id
                                    )

                            progress = 50 + int((index / len(df)) * 40)
                            upload_status["progress"] = progress

                        except Exception as e:
                            stats['errors'] += 1
                            logger.error(f"❌ Error processing row {index}: {e}")

                if docs_to_delete:
                    upload_status["message"] = f"Removing {len(docs_to_delete)} outdated docs..."
                    upload_status["progress"] = 95

                    for parent_id in docs_to_delete:
                        try:
                            from weaviate.classes.query import Filter
                            collection.data.delete_many(
                                where=Filter.by_property("parent_id").equal(parent_id)
                            )
                        except Exception as e:
                            logger.warning(f"Delete error: {e}")

                client.close()

                upload_status["message"] = f"✅ Completed: {stats['new']} new, {stats['updated']} updated, {stats['skipped']} skipped"
                upload_status["progress"] = 100

                logger.info(f"✅ Upload completed: {stats}")
                return True

            except Exception as e:
                logger.error(f"❌ Uploader worker error: {e}", exc_info=True)
                upload_status["message"] = f"Error: {str(e)[:100]}"
                upload_status["progress"] = 0
                return False

        running_loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            success = await running_loop.run_in_executor(executor, uploader_worker)

        if not success:
            upload_status["progress"] = 0

    except Exception as e:
        upload_status["message"] = f"Error: {str(e)}"
        upload_status["progress"] = 0
        logger.error(f"❌ Upload error: {e}", exc_info=True)

    finally:
        upload_status["running"] = False

@app.get("/api/admin/upload-status")
async def get_upload_status(user: str = Depends(require_admin)):
    return upload_status

@app.get("/api/admin/database-stats")
async def get_database_stats(user: str = Depends(require_admin)):
    try:
        if not collection:
            return {"error": "Database not connected"}

        total_parents = collection.aggregate.over_all(
            filters=Filter.by_property("chunk_type").equal("parent")
        ).total_count

        total_chunks = collection.aggregate.over_all(
            filters=Filter.by_property("chunk_type").equal("child")
        ).total_count

        web_docs = collection.aggregate.over_all(
            filters=(
                Filter.by_property("chunk_type").equal("parent") &
                Filter.by_property("source_type").equal("web")
            )
        ).total_count

        pdf_docs = collection.aggregate.over_all(
            filters=(
                Filter.by_property("chunk_type").equal("parent") &
                Filter.by_property("source_type").equal("pdf_ocr")
            )
        ).total_count

        ar_docs = collection.aggregate.over_all(
            filters=(
                Filter.by_property("chunk_type").equal("parent") &
                Filter.by_property("language").equal("ar")
            )
        ).total_count

        en_docs = collection.aggregate.over_all(
            filters=(
                Filter.by_property("chunk_type").equal("parent") &
                Filter.by_property("language").equal("en")
            )
        ).total_count

        fr_docs = collection.aggregate.over_all(
            filters=(
                Filter.by_property("chunk_type").equal("parent") &
                Filter.by_property("language").equal("fr")
            )
        ).total_count

        return {
            "total_documents": total_parents,
            "total_chunks": total_chunks,
            "by_source": {"web": web_docs, "pdf_ocr": pdf_docs},
            "by_language": {"ar": ar_docs, "en": en_docs, "fr": fr_docs}
        }

    except Exception as e:
        logger.error(f"Database stats error: {e}")
        return {"error": str(e)}

# ═══════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    logger.info("=" * 80)
    logger.info("🚀 GOEIC Enterprise Assistant OFFLINE V2 Starting...")
    logger.info(f"🤖 LLM: Ollama ({OLLAMA_MODEL})")
    logger.info(f"📊 Embeddings: Local ({EMBEDDING_MODEL})")
    logger.info(f"🎮 GPU: {gpu_name}")
    logger.info(f"🎤 Voice: {'Whisper + Edge TTS' if WHISPER_AVAILABLE else 'Disabled'}")
    logger.info(f"💰 API Costs: $0.00 (100% OFFLINE)")
    logger.info("=" * 80)

    uvicorn.run("main_offline:app", host="0.0.0.0", port=8000, reload=False, log_level="info")