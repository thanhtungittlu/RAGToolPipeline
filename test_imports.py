#!/usr/bin/env python3
"""
Test script để kiểm tra imports và cấu trúc code
Chạy: python3 test_imports.py
"""

import sys
from pathlib import Path

# Test imports (không cần Flask)
try:
    from config import DATA_DIR, DATABASE_PATH, ALLOWED_EXTENSIONS
    print("✓ config.py import thành công")
except Exception as e:
    print(f"✗ config.py import lỗi: {e}")
    sys.exit(1)

try:
    from models import Document, Chunk
    print("✓ models.py import thành công")
except Exception as e:
    print(f"✗ models.py import lỗi: {e}")
    sys.exit(1)

try:
    from database import get_db_connection, init_db
    print("✓ database.py import thành công")
except Exception as e:
    print(f"✗ database.py import lỗi: {e}")
    sys.exit(1)

try:
    from services.document_service import DocumentService
    print("✓ document_service.py import thành công")
except Exception as e:
    print(f"✗ document_service.py import lỗi: {e}")
    sys.exit(1)

try:
    from services.chunking_service import ChunkingService
    print("✓ chunking_service.py import thành công")
except Exception as e:
    print(f"✗ chunking_service.py import lỗi: {e}")
    sys.exit(1)

# Test data directory
if DATA_DIR.exists():
    files = list(DATA_DIR.glob("*.md"))
    print(f"✓ Tìm thấy {len(files)} file .md trong data/")
    for f in files:
        print(f"  - {f.name}")
else:
    print(f"⚠ Thư mục {DATA_DIR} chưa tồn tại (sẽ được tạo khi chạy app)")

print("\n✅ Tất cả imports thành công! Code structure OK.")
print("\nĐể chạy app:")
print("1. pip install -r requirements.txt")
print("2. python3 app.py")
