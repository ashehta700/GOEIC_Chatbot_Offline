# ═══════════════════════════════════════════════════════════════
# SMART WEB SCRAPER WITH DUPLICATE PREVENTION
# Scrapes GOEIC website and indexes ONLY new/changed pages
# ═══════════════════════════════════════════════════════════════

import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import os
import logging
import hashlib
from typing import Set, Dict, List, Tuple
from tqdm import tqdm
import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.classes.query import Filter
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime

# Configuration
BASE_URLS = [
    "https://www.goeic.gov.eg/ar/siteMap",
    "https://www.goeic.gov.eg/ar",
    "https://www.goeic.gov.eg/en",
    "https://www.goeic.gov.eg/fr"
]

MAX_WEB_PAGES = 6000
COLLECTION_NAME = "GOEIC_Knowledge_Base_V2"

# Blacklist: Titles/URLs to EXCLUDE (Junk Data)
BLACKLIST_TITLES = [
    "المستخدمين", "خريطة الموقع", "Users", "Site Map", 
    "تسجيل الدخول", "Login", "Goeic", "الرئيسية", "Home",
    "Sign In", "Register", "Forgot Password", "نسيت كلمة المرور"
]

BLACKLIST_URL_PATTERNS = [
    "login", "signin", "register", "forgot-password",
    "ask-us", "callUs", "contact-form"
]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize embedding model
logger.info("📥 Loading embedding model...")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
logger.info("✅ Embedding model loaded")

# ═══════════════════════════════════════════════════════════════
# DUPLICATE DETECTION SYSTEM
# ═══════════════════════════════════════════════════════════════

class DuplicateDetector:
    """Detects duplicates by comparing URL hashes and content hashes"""
    
    def __init__(self, client, collection_name):
        self.client = client
        self.collection_name = collection_name
        self.existing_urls = set()
        self.existing_hashes = {}  # url -> content_hash
        self.load_existing_data()
    
    def load_existing_data(self):
        """Load all existing URLs and content hashes from Weaviate"""
        try:
            collection = self.client.collections.get(self.collection_name)
            
            # Fetch all parent documents (full pages, not chunks)
            response = collection.query.fetch_objects(
                limit=10000,  # Adjust if you have more pages
                return_properties=["url", "content"],
                filters=Filter.by_property("chunk_type").equal("parent")
            )
            
            for obj in response.objects:
                url = obj.properties.get('url')
                content = obj.properties.get('content', '')
                
                if url:
                    self.existing_urls.add(url)
                    self.existing_hashes[url] = self._hash_content(content)
            
            logger.info(f"✅ Loaded {len(self.existing_urls)} existing URLs from database")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load existing data: {e}")
    
    def _hash_content(self, content: str) -> str:
        """Generate hash of content for change detection"""
        # Normalize content: remove extra whitespace, lowercase
        normalized = ' '.join(content.lower().split())
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def is_duplicate(self, url: str, content: str) -> Tuple[bool, str]:
        """
        Check if URL/content is duplicate
        Returns: (is_duplicate, reason)
        """
        if url not in self.existing_urls:
            return False, "new_url"
        
        # URL exists, check if content changed
        new_hash = self._hash_content(content)
        old_hash = self.existing_hashes.get(url)
        
        if old_hash and new_hash == old_hash:
            return True, "exact_duplicate"
        
        return False, "content_updated"
    
    def mark_as_processed(self, url: str, content: str):
        """Mark URL as processed in memory"""
        self.existing_urls.add(url)
        self.existing_hashes[url] = self._hash_content(content)

# ═══════════════════════════════════════════════════════════════
# SMART SCRAPER
# ═══════════════════════════════════════════════════════════════

class SmartGOEICScraper:
    def __init__(self, duplicate_detector: DuplicateDetector):
        self.visited: Set[str] = set()
        self.new_data: List[Dict] = []
        self.updated_data: List[Dict] = []
        self.queue = list(BASE_URLS)
        self.duplicate_detector = duplicate_detector
        
        # In-session MD5 dedup: catches same content at different URLs
        # (e.g. same page served under /ar and /en with identical body)
        self.seen_content_hashes: Set[str] = set()
        
        # Stats — mirrors your cleaner script output
        self.stats = {
            'scraped': 0,
            'new': 0,
            'updated': 0,
            'skipped': 0,
            'skipped_junk': 0,            # blacklist / short / login form
            'skipped_in_session_dup': 0,  # same content, different URL this run
            'errors': 0
        }
    
    def is_valid_url(self, url: str) -> bool:
        parsed = urlparse(url)
        if "goeic.gov.eg" not in parsed.netloc:
            return False
        exclude = ['.pdf', '.doc', '.docx', '.jpg', '.png', '.zip', 'login', 'search', 'contact']
        return not any(x in url.lower() for x in exclude)
    
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning to remove UI noise and junk"""
        # Step 1: Remove extra whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)
        
        # Step 2: Remove UI noise (login forms, navigation, etc.)
        noise_patterns = [
            "تسجيل الدخول", "تذكر كلمة المرور", "نسيت كلمة مرورك ?",
            "تحتاج إلى مساعدة؟", "غير مسجل؟ إنشاء حساب", "شروط الإستخدام",
            "Start Service", "Book an appointment", "Content evaluation",
            "اسم المستخدم", "كلمة المرور", "Login", "Sign in", "Register",
            "Cookie Policy", "Privacy Policy", "Terms of Service",
            "Share on Facebook", "Share on Twitter", "Print this page"
        ]
        
        for noise in noise_patterns:
            text = text.replace(noise, "")
        
        # Step 3: Remove excessive newlines
        text = "\n".join([line for line in text.split("\n") if line.strip()])
        
        return text.strip()
    
    def detect_language(self, url: str) -> str:
        if "/ar" in url:
            return "ar"
        if "/en" in url:
            return "en"
        if "/fr" in url:
            return "fr"
        return "ar"
    
    async def fetch_page(self, session, url):
        try:
            async with session.get(url, timeout=25) as response:
                if response.status == 200 and 'text/html' in response.headers.get('Content-Type', ''):
                    return await response.text()
        except Exception as e:
            logger.warning(f"❌ Failed: {url} - {e}")
            self.stats['errors'] += 1
        return None
    
    def extract_metadata(self, soup, url) -> Tuple[str, str]:
        title = ""
        category = "Website"
        
        # Breadcrumb
        breadcrumb = soup.select("#breadcrumb li")
        if breadcrumb:
            items = [li.get_text(strip=True) for li in breadcrumb]
            if items and ("بوابة" in items[0] or "Home" in items[0]):
                items.pop(0)
            if items:
                title = items[-1]
                if len(items) > 1:
                    category = " > ".join(items[:-1])
        
        # Fallbacks
        if not title:
            banner_h2 = soup.select_one('.banner h2')
            if banner_h2:
                title = banner_h2.get_text(strip=True)
        
        if not title:
            main_h1 = soup.select_one('.PageContent h1')
            if main_h1:
                title = main_h1.get_text(strip=True)
        
        if not title:
            if soup.title:
                title = soup.title.string.split('|')[0].strip()
            else:
                title = url
        
        return title, category
    
    def extract_main_content(self, soup):
        noise_selectors = [
            'nav', 'footer', 'header', 'script', 'style', 'noscript', 'iframe', 'aside',
            '.header-lg', '.mfa-container', '.banner', '#breadcrumb',
            '#sidebarCollapse', '.sidebar_widget',
            '.date_detail_content2', '.service-list-card',
            '.card-box-2', '.opinion-sec', '#poll_form',
            '.rights', '.logo-box', '.logo-img', '.links'
        ]
        
        for selector in noise_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        content_div = (soup.find('div', class_='PageContent') or
                      soup.find('div', class_='article-body') or
                      soup.find('div', class_='col-md-9') or
                      soup.find('main'))
        
        if content_div:
            return content_div.get_text(separator='\n')
        
        return soup.body.get_text(separator='\n') if soup.body else ""
    
    async def scrape(self):
        logger.info(f"🚀 Starting smart scrape (max: {MAX_WEB_PAGES} pages)")
        logger.info(f"📊 Existing URLs in DB: {len(self.duplicate_detector.existing_urls)}")
        
        async with aiohttp.ClientSession() as session:
            with tqdm(total=MAX_WEB_PAGES, desc="Scraping") as pbar:
                while self.queue and len(self.visited) < MAX_WEB_PAGES:
                    url = self.queue.pop(0)
                    
                    if url in self.visited or not self.is_valid_url(url):
                        continue
                    
                    self.visited.add(url)
                    html = await self.fetch_page(session, url)
                    
                    if not html:
                        continue
                    
                    soup = BeautifulSoup(html, 'html.parser')
                    title, category = self.extract_metadata(soup, url)
                    text = self.clean_text(self.extract_main_content(soup))
                    
                    # ════════════════════════════════════════
                    # FILTER PIPELINE (mirrors your cleaner script)
                    # ════════════════════════════════════════
                    
                    # FILTER 1: Skip blacklisted titles (login pages, site map, etc.)
                    if title in BLACKLIST_TITLES:
                        self.stats['skipped_junk'] += 1
                        self.stats['skipped'] += 1
                        logger.debug(f"⏭️ Skipped (blacklist title): {title}")
                        pbar.update(1)
                        continue
                    
                    # FILTER 2: Skip blacklisted URL patterns
                    if any(pattern in url.lower() for pattern in BLACKLIST_URL_PATTERNS):
                        self.stats['skipped_junk'] += 1
                        self.stats['skipped'] += 1
                        logger.debug(f"⏭️ Skipped (blacklist URL): {url}")
                        pbar.update(1)
                        continue
                    
                    # FILTER 3: Skip very short content (empty / navigation-only pages)
                    if len(text) < 50:
                        self.stats['skipped_junk'] += 1
                        self.stats['skipped'] += 1
                        logger.debug(f"⏭️ Skipped (too short {len(text)} chars): {title}")
                        pbar.update(1)
                        continue
                    
                    # FILTER 4: Skip login form content
                    if "اسم المستخدم" in text and "كلمة المرور" in text:
                        self.stats['skipped_junk'] += 1
                        self.stats['skipped'] += 1
                        logger.debug(f"⏭️ Skipped (login form): {title}")
                        pbar.update(1)
                        continue
                    
                    # FILTER 5: In-session MD5 dedup
                    # Catches the same content appearing at multiple URLs in this run
                    # (same logic as your cleaner script's hashlib.md5 step)
                    session_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                    if session_hash in self.seen_content_hashes:
                        self.stats['skipped_in_session_dup'] += 1
                        self.stats['skipped'] += 1
                        logger.debug(f"⏭️ Skipped (in-session duplicate content): {title}")
                        pbar.update(1)
                        continue
                    self.seen_content_hashes.add(session_hash)
                    
                    # ════════════════════════════════════════
                    # DUPLICATE CHECK vs DATABASE
                    # ════════════════════════════════════════
                    self.stats['scraped'] += 1
                    
                    is_dup, reason = self.duplicate_detector.is_duplicate(url, text)
                    
                    if is_dup and reason == "exact_duplicate":
                        self.stats['skipped'] += 1
                        logger.debug(f"⏭️ Skipped (already in DB, unchanged): {title}")
                        pbar.update(1)
                        continue
                    
                    # Prepare data
                    page_data = {
                        "url": url,
                        "title": title,
                        "content": text,
                        "category": category,
                        "source_type": "web",
                        "language": self.detect_language(url)
                    }
                    
                    if reason == "new_url":
                        self.new_data.append(page_data)
                        self.stats['new'] += 1
                        logger.info(f"🆕 NEW: {title}")
                    elif reason == "content_updated":
                        self.updated_data.append(page_data)
                        self.stats['updated'] += 1
                        logger.info(f"🔄 UPDATED: {title}")
                    
                    self.duplicate_detector.mark_as_processed(url, text)
                    pbar.update(1)
                    
                    # Find new links
                    link_soup = BeautifulSoup(html, 'html.parser')
                    for a in link_soup.find_all('a', href=True):
                        full_link = urljoin(url, a['href'])
                        if full_link not in self.visited:
                            self.queue.append(full_link)
        
        return self.new_data, self.updated_data

# ═══════════════════════════════════════════════════════════════
# SMART INDEXER (Only indexes new/updated)
# ═══════════════════════════════════════════════════════════════

def get_local_embedding(text: str):
    """Generate embedding using LOCAL model"""
    try:
        return embedding_model.encode(text).tolist()
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None

def generate_uuid(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def index_to_weaviate(new_data: List[Dict], updated_data: List[Dict], collection):
    """Index only new and updated pages"""
    
    # Text splitter
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "。", "،", "؟", "?", "!", " ", ""]
    )
    
    total_items = len(new_data) + len(updated_data)
    logger.info(f"📊 Indexing {len(new_data)} new + {len(updated_data)} updated = {total_items} pages")
    
    # Delete updated pages first (to re-index with new content)
    if updated_data:
        logger.info(f"🗑️ Deleting {len(updated_data)} outdated pages...")
        for item in tqdm(updated_data, desc="Deleting old versions"):
            parent_id = generate_uuid(item['url'])
            try:
                # Delete parent and all children
                collection.data.delete_many(
                    where=Filter.by_property("parent_id").equal(parent_id)
                )
            except Exception as e:
                logger.warning(f"Delete error for {item['url']}: {e}")
    
    # Index all (new + updated)
    all_data = new_data + updated_data
    
    with collection.batch.dynamic() as batch:
        for item in tqdm(all_data, desc="Indexing"):
            parent_text = item.get('content', '')
            if not parent_text or len(parent_text) < 50:
                continue
            
            title = item.get('title', 'No Title')
            url = item.get('url', '#')
            category = item.get('category', 'General')
            lang_code = item.get('language', 'ar')
            source_type = item.get('source_type', 'web')
            
            parent_id = generate_uuid(url)
            
            # Store parent
            batch.add_object(
                properties={
                    "content": parent_text,
                    "url": url,
                    "title": title,
                    "category": category,
                    "language": lang_code,
                    "source_type": source_type,
                    "chunk_type": "parent",
                    "parent_id": parent_id
                },
                vector=None,
                uuid=parent_id
            )
            
            # Store children
            chunks = child_splitter.split_text(parent_text)
            for i, chunk in enumerate(chunks):
                if len(chunk) < 50:
                    continue
                
                enriched_text = f"Title: {title}\nCategory: {category}\nContent: {chunk}"
                vector = get_local_embedding(enriched_text)
                
                if vector:
                    chunk_id = generate_uuid(f"{url}_{i}")
                    batch.add_object(
                        properties={
                            "content": chunk,
                            "url": url,
                            "title": title,
                            "category": category,
                            "language": lang_code,
                            "source_type": source_type,
                            "chunk_type": "child",
                            "parent_id": parent_id
                        },
                        vector=vector,
                        uuid=chunk_id
                    )
    
    logger.info("✅ Indexing complete!")

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

async def main():
    # Connect to Weaviate
    logger.info("🔌 Connecting to Weaviate...")
    try:
        client = weaviate.connect_to_local(
            additional_config=AdditionalConfig(
                timeout=Timeout(init=60, query=180, insert=180)
            )
        )
        collection = client.collections.get(COLLECTION_NAME)
        logger.info("✅ Connected to Weaviate")
    except Exception as e:
        logger.error(f"❌ Weaviate connection failed: {e}")
        return
    
    # Initialize duplicate detector
    duplicate_detector = DuplicateDetector(client, COLLECTION_NAME)
    
    # Scrape with smart duplicate detection
    scraper = SmartGOEICScraper(duplicate_detector)
    new_data, updated_data = await scraper.scrape()
    
    # Print statistics — same layout as your cleaner script output
    s = scraper.stats
    total_visited = len(scraper.visited)
    logger.info("=" * 60)
    logger.info("📊 SCRAPING STATISTICS")
    logger.info(f"   URLs visited         : {total_visited}")
    logger.info(f"   Passed all filters   : {s['scraped']}")
    logger.info(f"   ✅ New pages         : {s['new']}")
    logger.info(f"   🔄 Updated pages     : {s['updated']}")
    logger.info(f"   ⏭️ Skipped total     : {s['skipped']}")
    logger.info(f"      ├─ Junk/blacklist : {s['skipped_junk']}")
    logger.info(f"      ├─ In-session dup : {s['skipped_in_session_dup']}")
    logger.info(f"      └─ Already in DB  : {s['skipped'] - s['skipped_junk'] - s['skipped_in_session_dup']}")
    logger.info(f"   ❌ Fetch errors      : {s['errors']}")
    logger.info("=" * 60)
    
    # Index to Weaviate
    if new_data or updated_data:
        logger.info("\n🔄 Starting indexing...")
        index_to_weaviate(new_data, updated_data, collection)
    else:
        logger.info("\n✅ No new or updated pages to index!")
    
    client.close()
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "stats": scraper.stats,
        "new_pages": len(new_data),
        "updated_pages": len(updated_data)
    }
    
    with open(f"scrape_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info("\n🎉 All done! Check summary file for details.")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())