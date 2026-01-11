"""
Flask Routes - API endpoints cho RAG Tool
"""
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.exceptions import BadRequest
import logging
from pathlib import Path

from config import DATA_DIR, ALLOWED_EXTENSIONS
from database import get_db_connection
from services.document_service import DocumentService
from services.chunking_service import ChunkingService

logger = logging.getLogger(__name__)

def register_routes(app: Flask):
    """Đăng ký tất cả routes"""
    
    @app.route('/')
    def index():
        """Trang chủ - single page application"""
        return render_template('index.html')
    
    @app.route('/api/documents', methods=['GET'])
    def list_documents():
        """API: Lấy danh sách documents"""
        try:
            search = request.args.get('search', '').strip()
            documents = DocumentService.get_all_documents(search=search if search else None)
            return jsonify({
                'success': True,
                'documents': [doc.to_dict() for doc in documents]
            })
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/documents/discover', methods=['POST'])
    def discover_documents():
        """API: Quét và discover documents từ data directory"""
        try:
            documents = DocumentService.discover_documents()
            return jsonify({
                'success': True,
                'message': f'Discovered {len(documents)} documents',
                'documents': [doc.to_dict() for doc in documents]
            })
        except Exception as e:
            logger.error(f"Error discovering documents: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/documents/upload', methods=['POST'])
    def upload_document():
        """API: Upload document file"""
        try:
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'}), 400
            
            if not DocumentService.is_allowed_file(file.filename):
                return jsonify({
                    'success': False, 
                    'error': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
                }), 400
            
            # Lưu file
            filepath = DocumentService.save_uploaded_file(file)
            
            # Phân tích và đăng ký
            stats = DocumentService.analyze_file(filepath)
            doc_id = DocumentService.register_document(filepath, stats)
            
            doc = DocumentService.get_document_by_id(doc_id)
            
            return jsonify({
                'success': True,
                'message': 'File uploaded successfully',
                'document': doc.to_dict()
            })
        except Exception as e:
            logger.error(f"Error uploading document: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/documents/paste', methods=['POST'])
    def paste_text():
        """API: Paste text và lưu thành file"""
        try:
            data = request.get_json()
            if not data or 'text' not in data:
                return jsonify({'success': False, 'error': 'No text provided'}), 400
            
            text = data['text'].strip()
            if not text:
                return jsonify({'success': False, 'error': 'Text is empty'}), 400
            
            extension = data.get('extension', '.md')
            if extension not in ALLOWED_EXTENSIONS:
                extension = '.md'
            
            # Lưu text
            filepath = DocumentService.save_pasted_text(text, extension)
            
            # Phân tích và đăng ký
            stats = DocumentService.analyze_file(filepath)
            doc_id = DocumentService.register_document(filepath, stats)
            
            doc = DocumentService.get_document_by_id(doc_id)
            
            return jsonify({
                'success': True,
                'message': 'Text saved successfully',
                'document': doc.to_dict()
            })
        except Exception as e:
            logger.error(f"Error pasting text: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/documents/<int:doc_id>/content', methods=['GET'])
    def get_document_content(doc_id):
        """API: Lấy nội dung của document"""
        try:
            content = DocumentService.get_document_content(doc_id)
            if content is None:
                return jsonify({'success': False, 'error': 'Document not found'}), 404
            
            return jsonify({
                'success': True,
                'content': content
            })
        except Exception as e:
            logger.error(f"Error getting document content: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/chunking/strategies', methods=['GET'])
    def get_chunking_strategies():
        """API: Lấy danh sách chunking strategies và params"""
        strategies = {
            'fixed_size': {
                'name': 'Fixed Size Chunking',
                'description': 'Chia text thành các chunk có kích thước cố định',
                'params': {
                    'chunk_size': {'type': 'number', 'default': 500, 'label': 'Chunk Size'},
                    'overlap': {'type': 'number', 'default': 50, 'label': 'Overlap'}
                }
            },
            'markdown_header': {
                'name': 'Markdown Header Chunking',
                'description': 'Chia theo markdown headers (# ## ###)',
                'params': {
                    'max_depth': {'type': 'number', 'default': 3, 'label': 'Max Header Depth'}
                }
            },
            'recursive': {
                'name': 'Recursive Chunking',
                'description': 'Chia đệ quy theo separators (paragraphs, sentences, words)',
                'params': {
                    'max_chars': {'type': 'number', 'default': 500, 'label': 'Max Characters'}
                }
            },
            'paragraph': {
                'name': 'Paragraph-based Chunking',
                'description': 'Chia theo paragraphs (double newlines)',
                'params': {
                    'max_chars': {'type': 'number', 'default': 500, 'label': 'Max Characters'}
                }
            },
            'sliding_window': {
                'name': 'Sliding Window Chunking',
                'description': 'Chia với sliding window (overlap giữa các chunks)',
                'params': {
                    'window_size': {'type': 'number', 'default': 500, 'label': 'Window Size'},
                    'step_size': {'type': 'number', 'default': 250, 'label': 'Step Size'}
                }
            },
            'semantic': {
                'name': 'Semantic Chunking',
                'description': 'Chia dựa trên semantic similarity sử dụng embeddings từ Ollama server',
                'params': {
                    'chunk_size': {'type': 'number', 'default': 500, 'label': 'Chunk Size'},
                    'model': {'type': 'select', 'default': 'ollama', 'label': 'Embedding Model', 
                             'options': [
                                 {'value': 'ollama', 'label': 'Ollama (mặc định - localhost:11434)'},
                                 {'value': 'sentence-transformers', 'label': 'Sentence Transformers (fallback)'}
                             ]},
                    'ollama_url': {'type': 'text', 'default': 'http://localhost:11434', 'label': 'Ollama Server URL'},
                    'ollama_model': {'type': 'text', 'default': 'nomic-embed-text', 'label': 'Ollama Model Name (mặc định: nomic-embed-text)'}
                }
            }
        }
        
        return jsonify({
            'success': True,
            'strategies': strategies
        })
    
    @app.route('/api/chunking/run', methods=['POST'])
    def run_chunking():
        """API: Chạy chunking cho các documents đã chọn"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'error': 'No data provided'}), 400
            
            doc_ids = data.get('doc_ids', [])
            strategy = data.get('strategy', '')
            params = data.get('params', {})
            
            if not doc_ids:
                return jsonify({'success': False, 'error': 'No documents selected'}), 400
            
            if not strategy:
                return jsonify({'success': False, 'error': 'No strategy selected'}), 400
            
            # Validate doc_ids tồn tại
            documents = DocumentService.get_documents_by_ids(doc_ids)
            if len(documents) != len(doc_ids):
                return jsonify({'success': False, 'error': 'Some documents not found'}), 400
            
            # Chạy chunking
            chunks = ChunkingService.chunk_multiple_documents(doc_ids, strategy, params)
            
            # Lấy statistics
            stats = ChunkingService.get_chunk_statistics(doc_ids, strategy)
            
            return jsonify({
                'success': True,
                'message': f'Created {len(chunks)} chunks',
                'statistics': stats,
                'chunks': [chunk.to_dict() for chunk in chunks[:10]]  # Preview 10 chunks đầu
            })
        except Exception as e:
            logger.error(f"Error running chunking: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/chunks', methods=['GET'])
    def get_chunks():
        """API: Lấy chunks với pagination"""
        try:
            doc_id = request.args.get('doc_id', type=int)
            strategy = request.args.get('strategy', '')
            limit = request.args.get('limit', 10, type=int)
            offset = request.args.get('offset', 0, type=int)
            
            if doc_id:
                chunks = ChunkingService.get_chunks_by_doc(doc_id, strategy if strategy else None)
                # Apply pagination manually
                total = len(chunks)
                chunks = chunks[offset:offset+limit]
            else:
                if not strategy:
                    return jsonify({'success': False, 'error': 'doc_id or strategy required'}), 400
                # Get total count
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM chunks WHERE strategy = ?', (strategy,))
                total = cursor.fetchone()[0]
                conn.close()
                
                chunks = ChunkingService.get_chunks_by_strategy(strategy, limit, offset)
            
            return jsonify({
                'success': True,
                'chunks': [chunk.to_dict() for chunk in chunks],
                'total': total,
                'limit': limit,
                'offset': offset
            })
        except Exception as e:
            logger.error(f"Error getting chunks: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'success': False, 'error': 'Not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal error: {error}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500
