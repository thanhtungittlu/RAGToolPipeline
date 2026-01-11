"""
Flask Routes - API endpoints for RAG Tool
"""
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.exceptions import BadRequest
import logging
from pathlib import Path

from config import DATA_DIR, ALLOWED_EXTENSIONS
from services.document_service import DocumentService
from services.chunking_service import ChunkingService

logger = logging.getLogger(__name__)

def register_routes(app: Flask):
    """Register all routes"""
    
    @app.route('/')
    def index():
        """Home page - single page application"""
        return render_template('index.html')
    
    @app.route('/api/documents', methods=['GET'])
    def list_documents():
        """API: Get list of documents"""
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
        """API: Scan and discover documents from data directory"""
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
            
            # Save file
            filepath = DocumentService.save_uploaded_file(file)
            
            # Create document object from filepath
            doc = DocumentService.filepath_to_document(filepath)
            
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
        """API: Paste text and save as file"""
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
            
            # Save text
            filepath = DocumentService.save_pasted_text(text, extension)
            
            # Create document object from filepath
            doc = DocumentService.filepath_to_document(filepath)
            
            return jsonify({
                'success': True,
                'message': 'Text saved successfully',
                'document': doc.to_dict()
            })
        except Exception as e:
            logger.error(f"Error pasting text: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/documents/<filename>/content', methods=['GET'])
    def get_document_content(filename):
        """API: Get document content"""
        try:
            from urllib.parse import unquote
            filename = unquote(filename)  # Decode URL-encoded filename
            
            content = DocumentService.get_document_content(filename)
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
        """API: Get list of chunking strategies and parameters"""
        strategies = {
            'fixed_size': {
                'name': 'Fixed Size Chunking',
                'description': 'Split text into chunks of fixed size',
                'params': {
                    'chunk_size': {'type': 'number', 'default': 500, 'label': 'Chunk Size'},
                    'overlap': {'type': 'number', 'default': 50, 'label': 'Overlap'}
                }
            },
            'markdown_header': {
                'name': 'Markdown Header Chunking',
                'description': 'Split by markdown headers (# ## ###)',
                'params': {
                    'max_depth': {'type': 'number', 'default': 3, 'label': 'Max Header Depth'}
                }
            },
            'recursive': {
                'name': 'Recursive Chunking',
                'description': 'Recursively split by separators (paragraphs, sentences, words)',
                'params': {
                    'max_chars': {'type': 'number', 'default': 500, 'label': 'Max Characters'}
                }
            },
            'paragraph': {
                'name': 'Paragraph-based Chunking',
                'description': 'Split by paragraphs (double newlines)',
                'params': {
                    'max_chars': {'type': 'number', 'default': 500, 'label': 'Max Characters'}
                }
            },
            'sliding_window': {
                'name': 'Sliding Window Chunking',
                'description': 'Split with sliding window (overlap between chunks)',
                'params': {
                    'window_size': {'type': 'number', 'default': 500, 'label': 'Window Size'},
                    'step_size': {'type': 'number', 'default': 250, 'label': 'Step Size'}
                }
            },
            'semantic': {
                'name': 'Semantic Chunking',
                'description': 'Split based on semantic similarity using embeddings from Ollama server',
                'params': {
                    'chunk_size': {'type': 'number', 'default': 500, 'label': 'Chunk Size'},
                    'model': {'type': 'select', 'default': 'ollama', 'label': 'Embedding Model', 
                             'options': [
                                 {'value': 'ollama', 'label': 'Ollama (default - localhost:11434)'},
                                 {'value': 'sentence-transformers', 'label': 'Sentence Transformers (fallback)'}
                             ]},
                    'ollama_url': {'type': 'text', 'default': 'http://localhost:11434', 'label': 'Ollama Server URL'},
                    'ollama_model': {'type': 'text', 'default': 'nomic-embed-text', 'label': 'Ollama Model Name (default: nomic-embed-text)'}
                }
            }
        }
        
        return jsonify({
            'success': True,
            'strategies': strategies
        })
    
    @app.route('/api/chunking/run', methods=['POST'])
    def run_chunking():
        """API: Run chunking for selected documents"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'error': 'No data provided'}), 400
            
            filenames = data.get('filenames', [])
            strategy = data.get('strategy', '')
            params = data.get('params', {})
            
            if not filenames:
                return jsonify({'success': False, 'error': 'No documents selected'}), 400
            
            if not strategy:
                return jsonify({'success': False, 'error': 'No strategy selected'}), 400
            
            # Validate filenames exist
            documents = DocumentService.get_documents_by_filenames(filenames)
            if len(documents) != len(filenames):
                return jsonify({'success': False, 'error': 'Some documents not found'}), 400
            
            # Run chunking
            chunks = ChunkingService.chunk_multiple_documents(filenames, strategy, params)
            
            # Get statistics
            stats = ChunkingService.get_chunk_statistics(chunks)
            
            return jsonify({
                'success': True,
                'message': f'Created {len(chunks)} chunks',
                'statistics': stats,
                'chunks': chunks  # Return all chunks (no database, so return all)
            })
        except Exception as e:
            logger.error(f"Error running chunking: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/chunks', methods=['GET'])
    def get_chunks():
        """API: Get chunks with pagination (stored in session/memory - not implemented)"""
        # This endpoint is not needed without database
        # Chunks are returned directly from /api/chunking/run
        return jsonify({
            'success': False,
            'error': 'Chunks are returned directly from /api/chunking/run endpoint'
        }), 400
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'success': False, 'error': 'Not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal error: {error}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500
