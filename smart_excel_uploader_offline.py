# ═══════════════════════════════════════════════════════════════
# SMART EXCEL/PDF UPLOADER WITH DUPLICATE PREVENTION
# Updates vector database from Excel metadata + Word docs
# Only indexes new/changed documents
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import os
import weaviate
from docx import Document
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.classes.query import Filter
from sentence_transformers import SentenceTransformer
import logging
from datetime import datetime

# Configuration
EXCEL_FILE_PATH = os.getenv("EXCEL_FILE_PATH", "sheet_new.xlsx")
BASE_DOCS_DIR = os.getenv("BASE_DOCS_DIR", ".")
COLLECTION_NAME = "GOEIC_Knowledge_Base_V2"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize embedding model
logger.info("📥 Loading embedding model...")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
logger.info("✅ Embedding model loaded")

# ═══════════════════════════════════════════════════════════════
# HELPERS
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

def hash_content(text: str) -> str:
    """Generate hash for duplicate detection"""
    normalized = ' '.join(text.lower().split())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

def clean_text(text):
    if not text:
        return ""
    return " ".join(text.split())

# ═══════════════════════════════════════════════════════════════
# DUPLICATE DETECTOR FOR PDF/DOC
# ═══════════════════════════════════════════════════════════════

class DocumentDuplicateDetector:
    """Detects duplicate documents by URL and content hash"""
    
    def __init__(self, client, collection_name):
        self.client = client
        self.collection_name = collection_name
        self.existing_docs = {}  # parent_id -> content_hash
        self.load_existing_docs()
    
    def load_existing_docs(self):
        """Load existing PDF/OCR documents from Weaviate"""
        try:
            collection = self.client.collections.get(self.collection_name)
            
            # Fetch all PDF/OCR parent documents
            response = collection.query.fetch_objects(
                limit=10000,
                return_properties=["parent_id", "content", "url"],
                filters=(
                    Filter.by_property("chunk_type").equal("parent") &
                    Filter.by_property("source_type").equal("pdf_ocr")
                )
            )
            
            for obj in response.objects:
                parent_id = obj.properties.get('parent_id')
                content = obj.properties.get('content', '')
                
                if parent_id:
                    self.existing_docs[parent_id] = hash_content(content)
            
            logger.info(f"✅ Loaded {len(self.existing_docs)} existing PDF/DOC files")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load existing docs: {e}")
    
    def is_duplicate(self, parent_id: str, content: str) -> tuple:
        """
        Check if document is duplicate
        Returns: (is_duplicate, reason)
        """
        if parent_id not in self.existing_docs:
            return False, "new_document"
        
        # Document exists, check content
        new_hash = hash_content(content)
        old_hash = self.existing_docs.get(parent_id)
        
        if old_hash and new_hash == old_hash:
            return True, "exact_duplicate"
        
        return False, "content_updated"

# ═══════════════════════════════════════════════════════════════
# SMART UPLOADER
# ═══════════════════════════════════════════════════════════════

def smart_upload_from_excel():
    """
    Read Excel, extract from Word docs, index to Weaviate
    ONLY new/changed documents
    """
    logger.info(f"📂 Reading Excel: {EXCEL_FILE_PATH}")
    
    if not os.path.exists(EXCEL_FILE_PATH):
        logger.error(f"❌ Excel file not found: {os.path.abspath(EXCEL_FILE_PATH)}")
        return
    
    df = pd.read_excel(EXCEL_FILE_PATH).fillna("")
    logger.info(f"📊 Found {len(df)} rows in Excel")
    
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
    dup_detector = DocumentDuplicateDetector(client, COLLECTION_NAME)
    
    # Text splitter
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", ".", " ", ""]
    )
    
    # Statistics
    stats = {
        'total': len(df),
        'new': 0,
        'updated': 0,
        'skipped': 0,
        'missing_files': 0,
        'errors': 0
    }
    
    # Documents to delete (for updates)
    docs_to_delete = []
    
    logger.info(f"🔄 Processing {len(df)} documents...")
    
    with collection.batch.dynamic() as batch:
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            try:
                # Extract metadata
                web_url = str(row['url']).strip()
                title = str(row['title']).strip()
                category = str(row['category']).strip()
                lang = str(row.get('language', 'ar')).strip().lower()
                
                parent_id = generate_uuid(web_url)
                
                # Get document path
                raw_path = str(row['path']).strip().replace('"', '').replace("'", "")
                doc_path = os.path.join(BASE_DOCS_DIR, raw_path)
                
                if not os.path.exists(doc_path):
                    stats['missing_files'] += 1
                    logger.warning(f"⚠️ File not found: {doc_path}")
                    continue
                
                # Read Word document
                try:
                    doc = Document(doc_path)
                    full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                    full_text = clean_text(full_text)
                except Exception as e:
                    stats['errors'] += 1
                    logger.error(f"❌ Error reading {doc_path}: {e}")
                    continue
                
                if len(full_text) < 50:
                    stats['errors'] += 1
                    logger.warning(f"⚠️ Document too short: {title}")
                    continue
                
                # Check for duplicates
                is_dup, reason = dup_detector.is_duplicate(parent_id, full_text)
                
                if is_dup and reason == "exact_duplicate":
                    stats['skipped'] += 1
                    logger.debug(f"⏭️ Skipped (duplicate): {title}")
                    continue
                
                # Mark for deletion if updated
                if reason == "content_updated":
                    docs_to_delete.append(parent_id)
                    stats['updated'] += 1
                    logger.info(f"🔄 Will update: {title}")
                else:
                    stats['new'] += 1
                    logger.info(f"🆕 New: {title}")
                
                # Insert parent document
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
                
                # Insert child chunks
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
                
            except Exception as e:
                stats['errors'] += 1
                logger.error(f"❌ Error processing row {index}: {e}")
    
    # Delete outdated documents
    if docs_to_delete:
        logger.info(f"🗑️ Deleting {len(docs_to_delete)} outdated documents...")
        for parent_id in tqdm(docs_to_delete, desc="Deleting"):
            try:
                collection.data.delete_many(
                    where=Filter.by_property("parent_id").equal(parent_id)
                )
            except Exception as e:
                logger.warning(f"Delete error for {parent_id}: {e}")
    
    # Print statistics
    logger.info("=" * 60)
    logger.info("📊 UPLOAD STATISTICS:")
    logger.info(f"   Total rows: {stats['total']}")
    logger.info(f"   ✅ New documents: {stats['new']}")
    logger.info(f"   🔄 Updated documents: {stats['updated']}")
    logger.info(f"   ⏭️ Skipped (duplicates): {stats['skipped']}")
    logger.info(f"   ⚠️ Missing files: {stats['missing_files']}")
    logger.info(f"   ❌ Errors: {stats['errors']}")
    logger.info("=" * 60)
    
    client.close()
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "excel_file": EXCEL_FILE_PATH,
        "stats": stats
    }
    
    with open(f"upload_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w', encoding='utf-8') as f:
        import json
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info("\n🎉 Upload complete! Check summary file for details.")

if __name__ == "__main__":
    smart_upload_from_excel()
