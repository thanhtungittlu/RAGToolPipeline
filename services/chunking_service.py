"""
Chunking Service - Handle chunking with multiple strategies
"""
import re
import json
from typing import List, Dict, Any, Optional
import logging

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import get_db_connection
from models import Chunk
from services.document_service import DocumentService
from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL, OLLAMA_LLM_MODEL
)

# Try import libraries for semantic chunking
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

logger = logging.getLogger(__name__)

class ChunkingService:
    """Service to chunk documents with multiple strategies"""
    
    @staticmethod
    def fixed_size_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Chunk by fixed size with overlap"""
        if not text:
            return []
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end >= text_len:
                break
            
            start = end - overlap
        
        return chunks
    
    @staticmethod
    def markdown_header_chunk(text: str, max_depth: int = 3) -> List[str]:
        """Chunk by markdown headers"""
        if not text:
            return []
        
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_level = 0
        
        for line in lines:
            # Detect markdown header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if header_match:
                level = len(header_match.group(1))
                
                # If new header found and has content, save old chunk
                if current_chunk and level <= max_depth:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                
                current_level = level
            
            current_chunk.append(line)
        
        # Add last chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks if chunks else [text]
    
    @staticmethod
    def recursive_chunk(text: str, max_chars: int = 500, separators: List[str] = None) -> List[str]:
        """Recursive chunking - split by separators"""
        if not text:
            return []
        
        if separators is None:
            separators = ['\n\n', '\n', '. ', ' ', '']
        
        def _recursive_split(text: str, separators: List[str]) -> List[str]:
            if len(text) <= max_chars:
                return [text]
            
            if not separators:
                return [text]
            
            separator = separators[0]
            remaining_separators = separators[1:]
            
            if separator:
                splits = text.split(separator)
            else:
                # Last resort: split by character
                splits = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
                return splits
            
            chunks = []
            current_chunk = ""
            
            for split in splits:
                test_chunk = current_chunk + (separator if current_chunk else "") + split
                
                if len(test_chunk) <= max_chars:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    # Recursively split the split that's too long
                    if len(split) > max_chars:
                        chunks.extend(_recursive_split(split, remaining_separators))
                    else:
                        current_chunk = split
            
            if current_chunk:
                chunks.append(current_chunk)
            
            return chunks
        
        return _recursive_split(text, separators)
    
    @staticmethod
    def paragraph_chunk(text: str, max_chars: int = 500) -> List[str]:
        """Chunk by paragraphs"""
        if not text:
            return []
        
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            test_chunk = current_chunk + ('\n\n' if current_chunk else '') + para
            
            if len(test_chunk) <= max_chars:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If paragraph too long, split it
                if len(para) > max_chars:
                    chunks.extend(ChunkingService.fixed_size_chunk(para, max_chars, 0))
                    current_chunk = ""
                else:
                    current_chunk = para
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [text]
    
    @staticmethod
    def sliding_window_chunk(text: str, window_size: int = 500, step_size: int = 250) -> List[str]:
        """Sliding window chunking"""
        if not text:
            return []
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + window_size, text_len)
            chunk = text[start:end]
            chunks.append(chunk)
            start += step_size
        
        return chunks
    
    @staticmethod
    def _get_embeddings_ollama(texts: List[str], model: Optional[str] = None, base_url: Optional[str] = None) -> Optional[List[List[float]]]:
        """Get embeddings from Ollama API (local server at localhost:11434)"""
        if not HAS_REQUESTS:
            return None
        
        # Use default config if not provided
        if model is None:
            model = OLLAMA_EMBEDDING_MODEL
        if base_url is None:
            base_url = OLLAMA_BASE_URL
        
        try:
            # Ollama local API endpoint
            url = f"{base_url.rstrip('/')}/api/embeddings"
            
            embeddings = []
            
            for i, text in enumerate(texts):
                try:
                    # Ollama local API format
                    payload = {
                        "model": model,
                        "prompt": text
                    }
                    
                    response = requests.post(url, json=payload, timeout=60)
                    if response.status_code == 200:
                        data = response.json()
                        embedding = data.get("embedding", [])
                        
                        if embedding:
                            embeddings.append(embedding)
                        else:
                            logger.warning(f"Ollama returned empty embedding for text {i}")
                            return None
                    else:
                        logger.warning(f"Ollama API error: {response.status_code} - {response.text}")
                        return None
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Ollama request error: {e}")
                    return None
            
            return embeddings if len(embeddings) == len(texts) else None
        except Exception as e:
            logger.warning(f"Ollama API not available: {e}")
            return None
    
    @staticmethod
    def _get_embeddings_sentence_transformers(texts: List[str]) -> Optional[List[List[float]]]:
        """Get embeddings from sentence-transformers"""
        if not HAS_SENTENCE_TRANSFORMERS:
            return None
        
        try:
            # Use lightweight model, supports Vietnamese
            # Model will be cached after first use
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            logger.warning(f"Sentence transformers not available: {e}")
            return None
    
    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between 2 vectors"""
        if not HAS_NUMPY:
            # Fallback calculation without numpy
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(b * b for b in vec2) ** 0.5
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            return dot_product / (magnitude1 * magnitude2)
        else:
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)
            dot_product = np.dot(vec1_np, vec2_np)
            magnitude1 = np.linalg.norm(vec1_np)
            magnitude2 = np.linalg.norm(vec2_np)
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            return float(dot_product / (magnitude1 * magnitude2))
    
    @staticmethod
    def semantic_chunk(text: str, chunk_size: int = 500, model: str = "ollama", ollama_url: Optional[str] = None, ollama_model: Optional[str] = None) -> List[str]:
        """
        Semantic chunking thực sự sử dụng embeddings từ Ollama server
        
        Args:
            text: Text cần chunk
            chunk_size: Kích thước chunk mong muốn (ký tự)
            model: "ollama" (mặc định) hoặc "sentence-transformers" (fallback)
            ollama_url: URL của Ollama API (mặc định http://localhost:11434)
            ollama_model: Tên model cụ thể (nếu không dùng mặc định từ config)
        
        Returns:
            List các chunks
        """
        if not text:
            return []
        
        # Split text into sentences
        sentences = re.split(r'([.!?]\s+)', text)
        proper_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                proper_sentences.append(sentences[i] + sentences[i + 1])
            else:
                proper_sentences.append(sentences[i])
        
        # Filter empty sentences
        proper_sentences = [s.strip() for s in proper_sentences if s.strip()]
        
        if len(proper_sentences) <= 1:
            return [text] if text else []
        
        # Get embeddings - prioritize Ollama (default)
        embeddings = None
        
        # If ollama selected (default), try connecting to Ollama server
        if model.lower() == "ollama" or model.lower() != "sentence-transformers":
            logger.info(f"Using Ollama at {ollama_url or OLLAMA_BASE_URL}")
            use_model = ollama_model or OLLAMA_EMBEDDING_MODEL
            embeddings = ChunkingService._get_embeddings_ollama(
                proper_sentences, 
                model=use_model,
                base_url=ollama_url
            )
            
            # If ollama fails, fallback to sentence-transformers
            if embeddings is None and HAS_SENTENCE_TRANSFORMERS:
                logger.info("Ollama not available, fallback to sentence-transformers")
                embeddings = ChunkingService._get_embeddings_sentence_transformers(proper_sentences)
        
        # If sentence-transformers selected, use sentence-transformers
        elif model.lower() == "sentence-transformers":
            if HAS_SENTENCE_TRANSFORMERS:
                logger.info("Using sentence-transformers for semantic chunking")
                embeddings = ChunkingService._get_embeddings_sentence_transformers(proper_sentences)
            else:
                logger.warning("Sentence-transformers not available, fallback to simple semantic chunking")
                return ChunkingService._semantic_chunk_simple(text, chunk_size)
        
        # If still no embeddings, fallback to simple version
        if embeddings is None or len(embeddings) != len(proper_sentences):
            logger.info("No embeddings available, using simple semantic chunking")
            return ChunkingService._semantic_chunk_simple(text, chunk_size)
        
        # Calculate similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = ChunkingService._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        
        # Split chunks based on similarity and size
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, sentence in enumerate(proper_sentences):
            sentence_len = len(sentence)
            
            # If current sentence too long, split chunk
            if current_length + sentence_len > chunk_size * 1.5 and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_len
                continue
            
            # Check similarity with previous sentence
            if i > 0:
                similarity = similarities[i - 1]
                
                # If similarity low (< 0.5), could be a good split point
                # But only split if reached minimum size
                if similarity < 0.5 and current_length >= chunk_size * 0.5 and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_len
                    continue
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_len + 1  # +1 for space
            
            # If reached size, split chunk
            if current_length >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
        
        # Add last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [text]
    
    @staticmethod
    def _semantic_chunk_simple(text: str, chunk_size: int = 500) -> List[str]:
        """
        Fallback simple semantic chunking (without embeddings)
        """
        if not text:
            return []
        
        # Split by sentences
        sentences = re.split(r'([.!?]\s+)', text)
        proper_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                proper_sentences.append(sentences[i] + sentences[i + 1])
            else:
                proper_sentences.append(sentences[i])
        
        chunks = []
        current_chunk = ""
        
        for sentence in proper_sentences:
            test_chunk = current_chunk + (' ' if current_chunk else '') + sentence
            
            if len(test_chunk) <= chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [text]
    
    @staticmethod
    def chunk_document(doc_id: int, strategy: str, params: Dict[str, Any]) -> List[Chunk]:
        """Chunk a document with given strategy and params"""
        # Get document content
        content = DocumentService.get_document_content(doc_id)
        if not content:
            logger.warning(f"Document {doc_id} has no content")
            return []
        
        # Select strategy and chunk
        chunks_text = []
        
        if strategy == 'fixed_size':
            chunk_size = params.get('chunk_size', 500)
            overlap = params.get('overlap', 50)
            chunks_text = ChunkingService.fixed_size_chunk(content, chunk_size, overlap)
        
        elif strategy == 'markdown_header':
            max_depth = params.get('max_depth', 3)
            chunks_text = ChunkingService.markdown_header_chunk(content, max_depth)
        
        elif strategy == 'recursive':
            max_chars = params.get('max_chars', 500)
            separators = params.get('separators', ['\n\n', '\n', '. ', ' '])
            chunks_text = ChunkingService.recursive_chunk(content, max_chars, separators)
        
        elif strategy == 'paragraph':
            max_chars = params.get('max_chars', 500)
            chunks_text = ChunkingService.paragraph_chunk(content, max_chars)
        
        elif strategy == 'sliding_window':
            window_size = params.get('window_size', 500)
            step_size = params.get('step_size', 250)
            chunks_text = ChunkingService.sliding_window_chunk(content, window_size, step_size)
        
        elif strategy == 'semantic':
            chunk_size = params.get('chunk_size', 500)
            model = params.get('model', 'ollama')  # Default use Ollama
            ollama_url = params.get('ollama_url', None)
            ollama_model = params.get('ollama_model', None)
            chunks_text = ChunkingService.semantic_chunk(
                content, chunk_size, model, 
                ollama_url=ollama_url,
                ollama_model=ollama_model
            )
        
        else:
            logger.error(f"Unknown strategy: {strategy}")
            return []
        
        # Save chunks to database
        saved_chunks = []
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Delete old chunks for this document with this strategy (if any)
        cursor.execute('DELETE FROM chunks WHERE doc_id = ? AND strategy = ?', (doc_id, strategy))
        
        params_json = json.dumps(params)
        
        for position, chunk_text in enumerate(chunks_text, start=1):
            len_chars = len(chunk_text)
            cursor.execute('''
                INSERT INTO chunks (doc_id, strategy, params_json, position, text, len_chars)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (doc_id, strategy, params_json, position, chunk_text, len_chars))
            
            chunk_id = cursor.lastrowid
            saved_chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                strategy=strategy,
                params_json=params_json,
                position=position,
                text=chunk_text,
                len_chars=len_chars
            ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Created {len(saved_chunks)} chunks for doc {doc_id} with strategy {strategy}")
        return saved_chunks
    
    @staticmethod
    def chunk_multiple_documents(doc_ids: List[int], strategy: str, params: Dict[str, Any]) -> List[Chunk]:
        """Chunk multiple documents"""
        all_chunks = []
        for doc_id in doc_ids:
            chunks = ChunkingService.chunk_document(doc_id, strategy, params)
            all_chunks.extend(chunks)
        return all_chunks
    
    @staticmethod
    def get_chunks_by_doc(doc_id: int, strategy: Optional[str] = None) -> List[Chunk]:
        """Get chunks of a document"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if strategy:
            cursor.execute('''
                SELECT * FROM chunks 
                WHERE doc_id = ? AND strategy = ?
                ORDER BY position
            ''', (doc_id, strategy))
        else:
            cursor.execute('''
                SELECT * FROM chunks 
                WHERE doc_id = ?
                ORDER BY position
            ''', (doc_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [Chunk.from_row(row) for row in rows]
    
    @staticmethod
    def get_chunks_by_strategy(strategy: str, limit: int = 10, offset: int = 0) -> List[Chunk]:
        """Get chunks by strategy with pagination"""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM chunks 
            WHERE strategy = ?
            ORDER BY doc_id, position
            LIMIT ? OFFSET ?
        ''', (strategy, limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [Chunk.from_row(row) for row in rows]
    
    @staticmethod
    def get_chunk_statistics(doc_ids: List[int], strategy: str) -> Dict[str, Any]:
        """Get chunk statistics"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        placeholders = ','.join('?' * len(doc_ids))
        cursor.execute(f'''
            SELECT 
                COUNT(*) as total_chunks,
                AVG(len_chars) as avg_len,
                MIN(len_chars) as min_len,
                MAX(len_chars) as max_len
            FROM chunks
            WHERE doc_id IN ({placeholders}) AND strategy = ?
        ''', doc_ids + [strategy])
        
        row = cursor.fetchone()
        conn.close()
        
        return {
            'total_chunks': row['total_chunks'] or 0,
            'avg_len': round(row['avg_len'] or 0, 2),
            'min_len': row['min_len'] or 0,
            'max_len': row['max_len'] or 0
        }
