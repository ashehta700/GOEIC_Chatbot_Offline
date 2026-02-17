# ═══════════════════════════════════════════════════════════════
# SMART CLEANUP TOOL
# Removes duplicate parent documents (and their children) from
# Weaviate — offline version, no API keys needed.
#
# Run this whenever you suspect duplicates have built up, or
# after a failed/interrupted scraping run.
#
# Usage:
#   python smart_cleanup.py              ← interactive (asks before deleting)
#   python smart_cleanup.py --dry-run    ← only shows what WOULD be deleted
#   python smart_cleanup.py --force      ← deletes without asking
# ═══════════════════════════════════════════════════════════════

import sys
import argparse
import weaviate
from weaviate.classes.query import Filter
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()

COLLECTION_NAME = "GOEIC_Knowledge_Base_V2"
WEAVIATE_HOST   = os.getenv("WEAVIATE_HOST", "localhost")

# ───────────────────────────────────────────────────────────────
def connect():
    if WEAVIATE_HOST == "localhost":
        return weaviate.connect_to_local()
    return weaviate.connect_to_custom(
        http_host=WEAVIATE_HOST, http_port=8080, http_secure=False,
        grpc_host=WEAVIATE_HOST, grpc_port=50051, grpc_secure=False,
    )

# ───────────────────────────────────────────────────────────────
def smart_cleanup(dry_run: bool = False, force: bool = False):
    client = None
    try:
        print("🔌 Connecting to Weaviate...")
        client = connect()
        collection = client.collections.get(COLLECTION_NAME)

        # ── STEP 1: Scan all PARENT objects ──────────────────────
        print("🔍 Scanning for duplicate PARENT documents...")

        iterator = collection.iterator(
            return_properties=["url", "chunk_type"],
            include_vector=False,
        )

        seen_urls: dict[str, str] = {}   # url → first uuid
        duplicate_uuids: list[str] = []  # uuids of 2nd+ copies

        for obj in iterator:
            props = obj.properties or {}
            if props.get("chunk_type") != "parent":
                continue
            url  = props.get("url")
            uid  = str(obj.uuid)
            if not url:
                continue
            if url in seen_urls:
                duplicate_uuids.append(uid)
            else:
                seen_urls[url] = uid

        # ── STEP 2: Report ────────────────────────────────────────
        if not duplicate_uuids:
            print("🎉 Database is clean — no duplicate parents found.")
            return

        chunk_estimate = len(duplicate_uuids) * 10
        print(f"\n⚠️  Found {len(duplicate_uuids)} duplicate PARENT documents")
        print(f"   (~{chunk_estimate} child chunks will also be removed)")

        if dry_run:
            print("\n📋 DRY-RUN mode — nothing will be deleted.")
            print("   Re-run without --dry-run to actually remove them.")
            return

        # ── STEP 3: Confirm ───────────────────────────────────────
        if not force:
            answer = input("\n❓ Confirm deletion? (yes / no): ").strip().lower()
            if answer not in ("yes", "y"):
                print("🚫 Cancelled.")
                return

        # ── STEP 4: Delete in batches ─────────────────────────────
        print("\n🗑️  Deleting duplicates...")
        BATCH = 50
        total_deleted = 0

        for i in tqdm(range(0, len(duplicate_uuids), BATCH), desc="Deleting"):
            batch = duplicate_uuids[i : i + BATCH]

            # Delete children first (by parent_id)
            collection.data.delete_many(
                where=Filter.by_property("parent_id").contains_any(batch)
            )
            # Delete parents
            collection.data.delete_many(
                where=Filter.by_id().contains_any(batch)
            )
            total_deleted += len(batch)

        print(f"\n✅ Cleanup complete — removed {total_deleted} duplicate entries.")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise
    finally:
        if client:
            client.close()


# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove duplicate entries from Weaviate")
    parser.add_argument("--dry-run", action="store_true", help="Show duplicates without deleting")
    parser.add_argument("--force",   action="store_true", help="Delete without confirmation prompt")
    args = parser.parse_args()

    smart_cleanup(dry_run=args.dry_run, force=args.force)