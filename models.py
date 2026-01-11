"""
Data models cho Documents và Chunks
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
import json

@dataclass
class Document:
    """Model cho Document"""
    doc_id: Optional[int] = None
    filename: str = ""
    filepath: str = ""
    num_lines: int = 0
    num_chars: int = 0
    file_size: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def to_dict(self):
        """Convert Document thành dictionary"""
        return {
            'doc_id': self.doc_id,
            'filename': self.filename,
            'filepath': self.filepath,
            'num_lines': self.num_lines,
            'num_chars': self.num_chars,
            'file_size': self.file_size,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_row(cls, row):
        """Tạo Document từ database row"""
        return cls(
            doc_id=row['doc_id'],
            filename=row['filename'],
            filepath=row['filepath'],
            num_lines=row['num_lines'],
            num_chars=row['num_chars'],
            file_size=row['file_size'],
            created_at=row['created_at'],
            updated_at=row['updated_at']
        )

@dataclass
class Chunk:
    """Model cho Chunk"""
    chunk_id: Optional[int] = None
    doc_id: int = 0
    strategy: str = ""
    params_json: Optional[str] = None
    position: int = 0
    text: str = ""
    len_chars: int = 0
    created_at: Optional[str] = None
    
    def to_dict(self):
        """Convert Chunk thành dictionary"""
        params = json.loads(self.params_json) if self.params_json else {}
        return {
            'chunk_id': self.chunk_id,
            'doc_id': self.doc_id,
            'strategy': self.strategy,
            'params': params,
            'position': self.position,
            'text': self.text,
            'len_chars': self.len_chars,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_row(cls, row):
        """Tạo Chunk từ database row"""
        return cls(
            chunk_id=row['chunk_id'],
            doc_id=row['doc_id'],
            strategy=row['strategy'],
            params_json=row['params_json'],
            position=row['position'],
            text=row['text'],
            len_chars=row['len_chars'],
            created_at=row['created_at']
        )
