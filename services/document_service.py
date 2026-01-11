"""
Document Service - Handle upload, list, discover documents
"""
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import logging

# Secure filename helper (no need for werkzeug)
def secure_filename(filename: str) -> str:
    """Sanitize filename for security"""
    import re
    # Remove path separators and special characters
    filename = re.sub(r'[^\w\s-]', '', filename)
    filename = re.sub(r'[-\s]+', '-', filename)
    return filename.strip('-_')

from config import DATA_DIR, ALLOWED_EXTENSIONS
from database import get_db_connection, execute_query
from models import Document

logger = logging.getLogger(__name__)

class DocumentService:
    """Service to manage documents"""
    
    @staticmethod
    def is_allowed_file(filename: str) -> bool:
        """Check if file extension is allowed"""
        ext = Path(filename).suffix.lower()
        return ext in ALLOWED_EXTENSIONS
    
    @staticmethod
    def analyze_file(filepath: Path) -> dict:
        """Analyze file and return statistics"""
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
        """Save uploaded file"""
        if filename is None:
            filename = file.filename
        
        # Secure filename
        filename = secure_filename(filename)
        
        # Ensure extension exists
        if not Path(filename).suffix:
            filename += '.md'
        
        filepath = DATA_DIR / filename
        
        # If file already exists, add timestamp
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
        """Save pasted text as file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pasted_{timestamp}{extension}"
        filepath = DATA_DIR / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        
        logger.info(f"Pasted text saved: {filepath}")
        return filepath
    
    @staticmethod
    def register_document(filepath: Path, stats: dict) -> int:
        """Register document in database"""
        filename = filepath.name
        
        # Check if already exists
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
        """Scan data directory and discover all documents"""
        documents = []
        
        if not DATA_DIR.exists():
            DATA_DIR.mkdir(exist_ok=True)
            return documents
        
        for filepath in DATA_DIR.iterdir():
            if filepath.is_file() and DocumentService.is_allowed_file(filepath.name):
                stats = DocumentService.analyze_file(filepath)
                doc_id = DocumentService.register_document(filepath, stats)
                
                # Get document from DB
                doc = DocumentService.get_document_by_id(doc_id)
                if doc:
                    documents.append(doc)
        
        logger.info(f"Discovered {len(documents)} documents")
        return documents
    
    @staticmethod
    def get_all_documents(search: Optional[str] = None) -> List[Document]:
        """Get all documents, optionally filter by search"""
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
        """Get document by ID"""
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
        """Get document content"""
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
        """Get multiple documents by list of IDs"""
        if not doc_ids:
            return []
        
        conn = get_db_connection()
        cursor = conn.cursor()
        placeholders = ','.join('?' * len(doc_ids))
        cursor.execute(f'SELECT * FROM documents WHERE doc_id IN ({placeholders})', doc_ids)
        rows = cursor.fetchall()
        conn.close()
        
        return [Document.from_row(row) for row in rows]
