"""
Document Service - Xử lý upload, list, discover documents
"""
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import logging

# Secure filename helper (không cần werkzeug)
def secure_filename(filename: str) -> str:
    """Sanitize filename để an toàn"""
    import re
    # Remove path separators và các ký tự đặc biệt
    filename = re.sub(r'[^\w\s-]', '', filename)
    filename = re.sub(r'[-\s]+', '-', filename)
    return filename.strip('-_')

from config import DATA_DIR, ALLOWED_EXTENSIONS
from database import get_db_connection, execute_query
from models import Document

logger = logging.getLogger(__name__)

class DocumentService:
    """Service để quản lý documents"""
    
    @staticmethod
    def is_allowed_file(filename: str) -> bool:
        """Kiểm tra file extension có được phép không"""
        ext = Path(filename).suffix.lower()
        return ext in ALLOWED_EXTENSIONS
    
    @staticmethod
    def analyze_file(filepath: Path) -> dict:
        """Phân tích file và trả về thống kê"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            num_lines = len(content.splitlines())
            num_chars = len(content)
            file_size = filepath.stat().st_size
            
            return {
                'num_lines': num_lines,
                'num_chars': num_chars,
                'file_size': file_size
            }
        except Exception as e:
            logger.error(f"Error analyzing file {filepath}: {e}")
            return {'num_lines': 0, 'num_chars': 0, 'file_size': 0}
    
    @staticmethod
    def save_uploaded_file(file, filename: Optional[str] = None) -> Path:
        """Lưu file được upload"""
        if filename is None:
            filename = file.filename
        
        # Secure filename
        filename = secure_filename(filename)
        
        # Đảm bảo có extension
        if not Path(filename).suffix:
            filename += '.md'
        
        filepath = DATA_DIR / filename
        
        # Nếu file đã tồn tại, thêm timestamp
        if filepath.exists():
            stem = filepath.stem
            suffix = filepath.suffix
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{stem}_{timestamp}{suffix}"
            filepath = DATA_DIR / filename
        
        file.save(str(filepath))
        logger.info(f"File saved: {filepath}")
        return filepath
    
    @staticmethod
    def save_pasted_text(text: str, extension: str = '.md') -> Path:
        """Lưu text được paste thành file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pasted_{timestamp}{extension}"
        filepath = DATA_DIR / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        
        logger.info(f"Pasted text saved: {filepath}")
        return filepath
    
    @staticmethod
    def register_document(filepath: Path, stats: dict) -> int:
        """Đăng ký document vào database"""
        filename = filepath.name
        
        # Kiểm tra xem đã tồn tại chưa
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT doc_id FROM documents WHERE filename = ?', (filename,))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing
            doc_id = existing['doc_id']
            cursor.execute('''
                UPDATE documents 
                SET num_lines = ?, num_chars = ?, file_size = ?, updated_at = CURRENT_TIMESTAMP
                WHERE doc_id = ?
            ''', (stats['num_lines'], stats['num_chars'], stats['file_size'], doc_id))
        else:
            # Insert new
            cursor.execute('''
                INSERT INTO documents (filename, filepath, num_lines, num_chars, file_size)
                VALUES (?, ?, ?, ?, ?)
            ''', (filename, str(filepath), stats['num_lines'], stats['num_chars'], stats['file_size']))
            doc_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        return doc_id
    
    @staticmethod
    def discover_documents() -> List[Document]:
        """Quét thư mục data và discover tất cả documents"""
        documents = []
        
        if not DATA_DIR.exists():
            DATA_DIR.mkdir(exist_ok=True)
            return documents
        
        for filepath in DATA_DIR.iterdir():
            if filepath.is_file() and DocumentService.is_allowed_file(filepath.name):
                stats = DocumentService.analyze_file(filepath)
                doc_id = DocumentService.register_document(filepath, stats)
                
                # Lấy document từ DB
                doc = DocumentService.get_document_by_id(doc_id)
                if doc:
                    documents.append(doc)
        
        logger.info(f"Discovered {len(documents)} documents")
        return documents
    
    @staticmethod
    def get_all_documents(search: Optional[str] = None) -> List[Document]:
        """Lấy tất cả documents, có thể filter theo search"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if search:
            cursor.execute('''
                SELECT * FROM documents 
                WHERE filename LIKE ?
                ORDER BY created_at DESC
            ''', (f'%{search}%',))
        else:
            cursor.execute('SELECT * FROM documents ORDER BY created_at DESC')
        
        rows = cursor.fetchall()
        conn.close()
        
        return [Document.from_row(row) for row in rows]
    
    @staticmethod
    def get_document_by_id(doc_id: int) -> Optional[Document]:
        """Lấy document theo ID"""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM documents WHERE doc_id = ?', (doc_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return Document.from_row(row)
        return None
    
    @staticmethod
    def get_document_content(doc_id: int) -> Optional[str]:
        """Lấy nội dung của document"""
        doc = DocumentService.get_document_by_id(doc_id)
        if not doc:
            return None
        
        filepath = Path(doc.filepath)
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {e}")
            return None
    
    @staticmethod
    def get_documents_by_ids(doc_ids: List[int]) -> List[Document]:
        """Lấy nhiều documents theo list IDs"""
        if not doc_ids:
            return []
        
        conn = get_db_connection()
        cursor = conn.cursor()
        placeholders = ','.join('?' * len(doc_ids))
        cursor.execute(f'SELECT * FROM documents WHERE doc_id IN ({placeholders})', doc_ids)
        rows = cursor.fetchall()
        conn.close()
        
        return [Document.from_row(row) for row in rows]
