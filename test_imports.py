#!/usr/bin/env python3
"""
Test script to check imports and code structure
Run: python3 test_imports.py
"""

import sys
from pathlib import Path

# Test imports (no Flask needed)
try:
    from config import DATA_DIR, DATABASE_PATH, ALLOWED_EXTENSIONS
    print("✓ config.py import successful")
except Exception as e:
    print(f"✗ config.py import error: {e}")
    sys.exit(1)

try:
    from models import Document, Chunk
    print("✓ models.py import successful")
except Exception as e:
    print(f"✗ models.py import error: {e}")
    sys.exit(1)

try:
    from database import get_db_connection, init_db
    print("✓ database.py import successful")
except Exception as e:
    print(f"✗ database.py import error: {e}")
    sys.exit(1)

try:
    from services.document_service import DocumentService
    print("✓ document_service.py import successful")
except Exception as e:
    print(f"✗ document_service.py import error: {e}")
    sys.exit(1)

try:
    from services.chunking_service import ChunkingService
    print("✓ chunking_service.py import successful")
except Exception as e:
    print(f"✗ chunking_service.py import error: {e}")
    sys.exit(1)

# Test data directory
if DATA_DIR.exists():
    files = list(DATA_DIR.glob("*.md"))
    print(f"✓ Found {len(files)} .md files in data/")
    for f in files:
        print(f"  - {f.name}")
else:
    print(f"⚠ Directory {DATA_DIR} does not exist (will be created when running app)")

print("\n✅ All imports successful! Code structure OK.")
print("\nTo run app:")
print("1. pip install -r requirements.txt")
print("2. python3 app.py")
