"""
Document Service - Handle upload, list, discover documents
Works directly with filesystem, no database required
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
from models import Document

logger = logging.getLogger(__name__)

class DocumentService:
    """Service to manage documents from filesystem"""
    
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
    def filepath_to_document(filepath: Path) -> Document:
        """Convert filepath to Document object"""
        stats = DocumentService.analyze_file(filepath)
        file_stat = filepath.stat()
        
        return Document(
            doc_id=None,  # No database ID needed
            filename=filepath.name,
            filepath=str(filepath),
            num_lines=stats['num_lines'],
            num_chars=stats['num_chars'],
            file_size=stats['file_size'],
            created_at=datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            updated_at=datetime.fromtimestamp(file_stat.st_mtime).isoformat()
        )
    
    @staticmethod
    def discover_documents() -> List[Document]:
        """Scan data directory and discover all documents"""
        documents = []
        
        if not DATA_DIR.exists():
            DATA_DIR.mkdir(exist_ok=True)
            return documents
        
        for filepath in DATA_DIR.iterdir():
            if filepath.is_file() and DocumentService.is_allowed_file(filepath.name):
                doc = DocumentService.filepath_to_document(filepath)
                documents.append(doc)
        
        logger.info(f"Discovered {len(documents)} documents")
        return documents
    
    @staticmethod
    def get_all_documents(search: Optional[str] = None) -> List[Document]:
        """Get all documents from filesystem, optionally filter by search"""
        documents = DocumentService.discover_documents()
        
        if search:
            search_lower = search.lower()
            documents = [doc for doc in documents if search_lower in doc.filename.lower()]
        
        # Sort by filename
        documents.sort(key=lambda x: x.filename)
        
        return documents
    
    @staticmethod
    def get_document_by_filename(filename: str) -> Optional[Document]:
        """Get document by filename"""
        filepath = DATA_DIR / filename
        if not filepath.exists() or not DocumentService.is_allowed_file(filename):
            return None
        
        return DocumentService.filepath_to_document(filepath)
    
    @staticmethod
    def get_document_content(filename: str) -> Optional[str]:
        """Get document content by filename"""
        filepath = DATA_DIR / filename
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {e}")
            return None
    
    @staticmethod
    def get_documents_by_filenames(filenames: List[str]) -> List[Document]:
        """Get multiple documents by list of filenames"""
        documents = []
        for filename in filenames:
            doc = DocumentService.get_document_by_filename(filename)
            if doc:
                documents.append(doc)
        return documents
