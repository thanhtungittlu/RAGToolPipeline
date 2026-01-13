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
from services.embedding_service import EmbeddingService
from services.retrieval_service import RetrievalService
from services.ragas_service import RAGASService
from services.visualization_service import VisualizationService
from services.retrieval_service import RetrievalService
from services.ragas_service import RAGASService
from services.visualization_service import VisualizationService

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
                'description': 'Recursively split by separators (paragraphs, sentences, words, markdown headers)',
                'params': {
                    'max_chars': {'type': 'number', 'default': 500, 'label': 'Max Characters'},
                    'separators': {'type': 'text', 'default': '\\n\\n,\\n,. , ,#', 'label': 'Separators (comma-separated)', 'placeholder': 'e.g., \\n\\n,\\n,. , ,#'}
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
    
    @app.route('/api/chunking/evaluate', methods=['POST'])
    def evaluate_chunking():
        """API: Evaluate chunking quality (Boundary score, Completeness, Coherence)"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'error': 'No data provided'}), 400
            
            chunks = data.get('chunks', [])
            if not chunks:
                return jsonify({'success': False, 'error': 'No chunks provided'}), 400
            
            # Evaluate Boundary Quality (only metric displayed)
            boundary_result = ChunkingService.evaluate_boundary_score_fast(chunks)
            if boundary_result.get('success') and boundary_result.get('score') is not None:
                boundary_result['quality_level'] = 'PASS' if boundary_result['score'] >= 0.8 else 'FAIL'
            
            return jsonify({
                'success': boundary_result.get('success', False),
                'results': {
                    'boundary': boundary_result
                }
            })
        except Exception as e:
            logger.error(f"Error evaluating chunking: {e}")
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
    
    @app.route('/api/embeddings/generate', methods=['POST'])
    def generate_embeddings():
        """API: Generate embeddings for chunks"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'error': 'No data provided'}), 400
            
            chunks = data.get('chunks', [])
            if not chunks:
                return jsonify({'success': False, 'error': 'No chunks provided'}), 400
            
            # Get embedding method
            method = data.get('method', 'ollama')  # 'ollama' or 'sentence-transformers'
            
            # Extract text from chunks
            texts = [chunk.get('text', '') for chunk in chunks]
            if not any(texts):
                return jsonify({'success': False, 'error': 'No text found in chunks'}), 400
            
            # Generate embeddings based on method
            embeddings = None
            if method == 'ollama':
                embeddings = EmbeddingService.get_embeddings_ollama(texts)
                if embeddings:
                    logger.info(f"Generated {len(embeddings)} embeddings using Ollama")
            elif method == 'sentence-transformers':
                embeddings = EmbeddingService.get_embeddings_sentence_transformers(texts)
                if embeddings:
                    logger.info(f"Generated {len(embeddings)} embeddings using sentence-transformers")
            
            # Fallback to default if method-specific failed
            if not embeddings:
                embeddings = EmbeddingService.get_embeddings(texts)
            
            if not embeddings:
                return jsonify({'success': False, 'error': 'Failed to generate embeddings'}), 500
            
            # Return embeddings with chunk info
            result = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                result.append({
                    'chunk_index': i,
                    'chunk_id': chunk.get('chunk_id', i),
                    'filename': chunk.get('filename', ''),
                    'position': chunk.get('position', 0),
                    'embedding': embedding,
                    'embedding_dim': len(embedding)
                })
            
            return jsonify({
                'success': True,
                'embeddings': result,
                'total': len(result),
                'embedding_dim': len(embeddings[0]) if embeddings else 0
            })
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/embeddings/evaluate', methods=['POST'])
    def evaluate_embeddings():
        """
        API: Evaluate embeddings using specified metric or comprehensive evaluation
        
        Supports:
        - Single metric: 'silhouette', 'davies_bouldin'
        - Comprehensive evaluation: metric=None or 'comprehensive'
        """
        try:
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'error': 'No data provided'}), 400
            
            embeddings = data.get('embeddings', [])
            if not embeddings:
                return jsonify({'success': False, 'error': 'No embeddings provided'}), 400
            
            # Extract embedding vectors (handle both formats)
            embedding_vectors = []
            for emb in embeddings:
                if isinstance(emb, list):
                    embedding_vectors.append(emb)
                elif isinstance(emb, dict) and 'embedding' in emb:
                    embedding_vectors.append(emb['embedding'])
                else:
                    return jsonify({'success': False, 'error': 'Invalid embedding format'}), 400
            
            if len(embedding_vectors) < 2:
                return jsonify({'success': False, 'error': 'Need at least 2 embeddings for evaluation'}), 400
            
            metric = data.get('metric')  # None, 'comprehensive', or specific metric name
            n_clusters = data.get('n_clusters', 10)  # Default to 10 for comprehensive evaluation
            
            # Comprehensive evaluation (recommended approach from guide)
            if metric is None or metric == 'comprehensive':
                comprehensive_result = EmbeddingService.comprehensive_embedding_evaluation(embedding_vectors, n_clusters)
                
                if comprehensive_result.get('success'):
                    return jsonify({
                        'success': True,
                        'method': 'comprehensive',
                        'results': comprehensive_result.get('results')
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': comprehensive_result.get('error', 'Comprehensive evaluation failed')
                    }), 500
            
            # If metric is not specified, evaluate all basic metrics (legacy mode)
            if metric == 'all':
                results = {}
                
                # Evaluate Silhouette Score
                silhouette_result = EmbeddingService.evaluate_silhouette_score(embedding_vectors, n_clusters)
                if silhouette_result.get('success') and silhouette_result.get('score') is not None:
                    silhouette_result['quality_level'] = EmbeddingService.get_embedding_quality_level(
                        'silhouette', silhouette_result['score']
                    )
                results['silhouette'] = silhouette_result
                
                # Evaluate Davies-Bouldin Index
                davies_bouldin_result = EmbeddingService.evaluate_davies_bouldin_index(embedding_vectors, n_clusters)
                if davies_bouldin_result.get('success') and davies_bouldin_result.get('score') is not None:
                    davies_bouldin_result['quality_level'] = EmbeddingService.get_embedding_quality_level(
                        'davies_bouldin', davies_bouldin_result['score']
                    )
                results['davies_bouldin'] = davies_bouldin_result
                
                # Return success if at least one metric succeeded
                all_success = any(
                    r.get('success') and r.get('score') is not None 
                    for r in results.values()
                )
                
                return jsonify({
                    'success': all_success,
                    'results': results
                })
            
            # Evaluate single metric
            if metric == 'silhouette':
                result = EmbeddingService.evaluate_silhouette_score(embedding_vectors, n_clusters)
            elif metric == 'davies_bouldin':
                result = EmbeddingService.evaluate_davies_bouldin_index(embedding_vectors, n_clusters)
            elif metric == 'intra_cluster_distance':
                method = data.get('intra_method', 'centroid')
                result = EmbeddingService.evaluate_intra_cluster_distance(embedding_vectors, n_clusters, method)
            else:
                return jsonify({'success': False, 'error': f'Unknown metric: {metric}. Supported: silhouette, davies_bouldin, intra_cluster_distance, comprehensive'}), 400
            
            # Add quality_level to single metric result
            if result.get('success') and result.get('score') is not None:
                result['quality_level'] = EmbeddingService.get_embedding_quality_level(metric, result['score'])
            
            return jsonify({
                'success': result.get('success', False),
                'metric': metric,
                'score': result.get('score'),
                'n_clusters': result.get('n_clusters'),
                'description': result.get('description'),
                'error': result.get('error'),
                'quality_level': result.get('quality_level')
            })
        except Exception as e:
            logger.error(f"Error evaluating embeddings: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/retrieval/evaluate', methods=['POST'])
    def evaluate_retrieval():
        """
        API: Evaluate retrieval quality (Layer 3)
        
        Supports single query or multiple queries evaluation
        """
        try:
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'error': 'No data provided'}), 400
            
            # Check if single query or multiple queries
            if 'query_embedding' in data:
                # Single query evaluation
                query_embedding = data.get('query_embedding')
                document_embeddings = data.get('document_embeddings', [])
                relevant_doc_indices = data.get('relevant_doc_indices', [])
                k_values = data.get('k_values', [5, 10])
                relevance_scores = data.get('relevance_scores')  # Optional
                
                if not query_embedding or not document_embeddings:
                    return jsonify({'success': False, 'error': 'Missing required fields'}), 400
                
                if isinstance(relevant_doc_indices, list):
                    relevant_doc_indices = set(relevant_doc_indices)
                
                result = RetrievalService.evaluate_retrieval_quality(
                    query_embedding,
                    document_embeddings,
                    relevant_doc_indices,
                    k_values,
                    relevance_scores
                )
                
                return jsonify(result)
            
            elif 'test_queries' in data:
                # Multiple queries evaluation
                test_queries = data.get('test_queries', [])
                document_embeddings = data.get('document_embeddings', [])
                k_values = data.get('k_values', [5, 10])
                
                if not test_queries or not document_embeddings:
                    return jsonify({'success': False, 'error': 'Missing required fields'}), 400
                
                result = RetrievalService.evaluate_multiple_queries(
                    test_queries,
                    document_embeddings,
                    k_values
                )
                
                return jsonify(result)
            else:
                return jsonify({'success': False, 'error': 'Must provide either query_embedding or test_queries'}), 400
                
        except Exception as e:
            logger.error(f"Error evaluating retrieval: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/ragas/evaluate', methods=['POST'])
    def evaluate_ragas():
        """
        API: Evaluate RAG quality using RAGAS framework
        """
        try:
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'error': 'No data provided'}), 400
            
            question = data.get('question')
            answer = data.get('answer')
            contexts = data.get('contexts', [])
            relevant_contexts = data.get('relevant_contexts')  # Optional
            answer_embedding = data.get('answer_embedding')  # Optional
            question_embedding = data.get('question_embedding')  # Optional
            
            if not question or not answer or not contexts:
                return jsonify({'success': False, 'error': 'Missing required fields: question, answer, contexts'}), 400
            
            # Check if single metric or comprehensive
            metric = data.get('metric')
            
            if metric and metric != 'comprehensive':
                # Single metric evaluation
                if metric == 'faithfulness':
                    result = RAGASService.evaluate_faithfulness(answer, contexts)
                elif metric == 'answer_relevancy':
                    result = RAGASService.evaluate_answer_relevancy(
                        question, answer, answer_embedding, question_embedding
                    )
                elif metric == 'context_precision':
                    if not relevant_contexts:
                        return jsonify({'success': False, 'error': 'relevant_contexts required for context_precision'}), 400
                    result = RAGASService.evaluate_context_precision(contexts, relevant_contexts)
                elif metric == 'context_recall':
                    if not relevant_contexts:
                        return jsonify({'success': False, 'error': 'relevant_contexts required for context_recall'}), 400
                    result = RAGASService.evaluate_context_recall(contexts, relevant_contexts)
                else:
                    return jsonify({'success': False, 'error': f'Unknown metric: {metric}'}), 400
                
                return jsonify(result)
            else:
                # Comprehensive evaluation
                result = RAGASService.comprehensive_ragas_evaluation(
                    question,
                    answer,
                    contexts,
                    relevant_contexts,
                    answer_embedding,
                    question_embedding
                )
                
                return jsonify(result)
                
        except Exception as e:
            logger.error(f"Error evaluating RAGAS: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/visualization/reduce', methods=['POST'])
    def reduce_dimensions():
        """
        API: Reduce embedding dimensions for visualization (UMAP or t-SNE)
        """
        try:
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'error': 'No data provided'}), 400
            
            embeddings = data.get('embeddings', [])
            if not embeddings:
                return jsonify({'success': False, 'error': 'No embeddings provided'}), 400
            
            # Extract embedding vectors
            embedding_vectors = []
            for emb in embeddings:
                if isinstance(emb, list):
                    embedding_vectors.append(emb)
                elif isinstance(emb, dict) and 'embedding' in emb:
                    embedding_vectors.append(emb['embedding'])
                else:
                    return jsonify({'success': False, 'error': 'Invalid embedding format'}), 400
            
            method = data.get('method', 'umap').lower()  # Always 'umap' now
            n_components = data.get('n_components', 2)  # 2 or 3
            labels = data.get('labels')  # Optional
            chunks = data.get('chunks')  # Optional
            
            # Prepare visualization data
            result = VisualizationService.prepare_visualization_data(
                embedding_vectors,
                labels,
                chunks,
                method,
                n_components=n_components
            )
            
            return jsonify(result)
                
        except Exception as e:
            logger.error(f"Error reducing dimensions: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'success': False, 'error': 'Not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal error: {error}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500
