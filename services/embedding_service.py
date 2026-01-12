"""
Embedding Service - Generate embeddings and evaluate embedding quality
"""
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("requests not available, Ollama API will not work")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("numpy not available, evaluation metrics will not work")

try:
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("sklearn not available, evaluation metrics will not work")

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers not available, fallback embedding will be used")


class EmbeddingService:
    """Service for generating embeddings and evaluating embedding quality"""
    
    @staticmethod
    def get_embeddings_ollama(texts: List[str], model: Optional[str] = None, base_url: Optional[str] = None) -> Optional[List[List[float]]]:
        """Get embeddings from Ollama API"""
        if not HAS_REQUESTS:
            return None
        
        try:
            import os
            from config import OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL
            
            if base_url is None:
                base_url = os.getenv('OLLAMA_BASE_URL', OLLAMA_BASE_URL)
            if model is None:
                model = os.getenv('OLLAMA_EMBEDDING_MODEL', OLLAMA_EMBEDDING_MODEL)
            
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
                        logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                        return None
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Ollama request error: {e}")
                    return None
            
            return embeddings if len(embeddings) == len(texts) else None
        except Exception as e:
            logger.error(f"Error getting embeddings from Ollama: {e}")
            return None
    
    @staticmethod
    def get_embeddings_sentence_transformers(texts: List[str], model: Optional[str] = None) -> Optional[List[List[float]]]:
        """Get embeddings using sentence-transformers"""
        if not HAS_SENTENCE_TRANSFORMERS:
            return None
        
        try:
            if model is None:
                model = "all-MiniLM-L6-v2"  # Default lightweight model
            
            logger.info(f"Loading sentence-transformers model: {model}")
            encoder = SentenceTransformer(model)
            embeddings = encoder.encode(texts, show_progress_bar=False)
            
            # Convert numpy array to list of lists
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error getting embeddings from sentence-transformers: {e}")
            return None
    
    @staticmethod
    def get_embeddings(texts: List[str], model: Optional[str] = None, base_url: Optional[str] = None) -> Optional[List[List[float]]]:
        """Get embeddings with fallback mechanism"""
        # Try Ollama first
        embeddings = EmbeddingService.get_embeddings_ollama(texts, model, base_url)
        if embeddings:
            logger.info(f"Generated {len(embeddings)} embeddings using Ollama")
            return embeddings
        
        # Fallback to sentence-transformers
        embeddings = EmbeddingService.get_embeddings_sentence_transformers(texts, model)
        if embeddings:
            logger.info(f"Generated {len(embeddings)} embeddings using sentence-transformers")
            return embeddings
        
        logger.error("Failed to generate embeddings using any method")
        return None
    
    @staticmethod
    def get_embedding_quality_level(metric: str, score: float) -> str:
        """Get quality level based on metric and score"""
        if metric == 'silhouette':
            # Silhouette Score: range -1 to 1
            # -1 to 0: Bad, 0 to 0.3: Fair, 0.3 to 0.5: Good, 0.5 to 0.7: Very Good, 0.7+: Excellent
            if score < 0:
                return 'BAD'
            elif score < 0.3:
                return 'FAIR'
            elif score < 0.5:
                return 'GOOD'
            elif score < 0.7:
                return 'VERY GOOD'
            else:
                return 'EXCELLENT'
        elif metric == 'davies_bouldin':
            # Davies-Bouldin Index: lower is better (typically 0 to ~2-3)
            # > 1.5: Bad, 1.0-1.5: Fair, 0.5-1.0: Good, 0.3-0.5: Very Good, < 0.3: Excellent
            if score > 1.5:
                return 'BAD'
            elif score > 1.0:
                return 'FAIR'
            elif score > 0.5:
                return 'GOOD'
            elif score > 0.3:
                return 'VERY GOOD'
            else:
                return 'EXCELLENT'
        elif metric == 'calinski_harabasz':
            # Calinski-Harabasz Index: higher is better (typically 0 to several hundred/thousand)
            # The score depends heavily on data size, so we use relative thresholds
            # < 50: Bad, 50-100: Fair, 100-200: Good, 200-300: Very Good, > 300: Excellent
            if score < 50:
                return 'BAD'
            elif score < 100:
                return 'FAIR'
            elif score < 200:
                return 'GOOD'
            elif score < 300:
                return 'VERY GOOD'
            else:
                return 'EXCELLENT'
        else:
            return 'UNKNOWN'
    
    @staticmethod
    def evaluate_silhouette_score(embeddings: List[List[float]], n_clusters: int = 5) -> Dict:
        """Evaluate embeddings using Silhouette Score"""
        if not HAS_SKLEARN:
            return {
                'success': False,
                'error': 'sklearn not available',
                'score': None
            }
        
        try:
            X = np.array(embeddings)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=min(n_clusters, len(embeddings)), random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            # Calculate silhouette score
            if len(set(cluster_labels)) < 2:
                return {
                    'success': False,
                    'error': 'Need at least 2 clusters for Silhouette Score',
                    'score': None
                }
            
            score = silhouette_score(X, cluster_labels)
            
            return {
                'success': True,
                'score': float(score),
                'n_clusters': int(len(set(cluster_labels))),
                'description': 'Silhouette Score ranges from -1 to 1. Higher is better. Measures how similar an object is to its own cluster compared to other clusters.'
            }
        except Exception as e:
            logger.error(f"Error calculating Silhouette Score: {e}")
            return {
                'success': False,
                'error': str(e),
                'score': None
            }
    
    @staticmethod
    def evaluate_davies_bouldin_index(embeddings: List[List[float]], n_clusters: int = 5) -> Dict:
        """Evaluate embeddings using Davies-Bouldin Index"""
        if not HAS_SKLEARN:
            return {
                'success': False,
                'error': 'sklearn not available',
                'score': None
            }
        
        try:
            X = np.array(embeddings)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=min(n_clusters, len(embeddings)), random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            # Calculate Davies-Bouldin Index
            if len(set(cluster_labels)) < 2:
                return {
                    'success': False,
                    'error': 'Need at least 2 clusters for Davies-Bouldin Index',
                    'score': None
                }
            
            score = davies_bouldin_score(X, cluster_labels)
            
            return {
                'success': True,
                'score': float(score),
                'n_clusters': int(len(set(cluster_labels))),
                'description': 'Davies-Bouldin Index is a measure of cluster quality. Lower is better. Measures the average similarity ratio of clusters.'
            }
        except Exception as e:
            logger.error(f"Error calculating Davies-Bouldin Index: {e}")
            return {
                'success': False,
                'error': str(e),
                'score': None
            }
    
    @staticmethod
    def evaluate_calinski_harabasz_index(embeddings: List[List[float]], n_clusters: int = 5) -> Dict:
        """Evaluate embeddings using Calinski-Harabasz Index (Variance Ratio Criterion)"""
        if not HAS_SKLEARN:
            return {
                'success': False,
                'error': 'sklearn not available',
                'score': None
            }
        
        try:
            X = np.array(embeddings)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=min(n_clusters, len(embeddings)), random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            # Calculate Calinski-Harabasz Index
            if len(set(cluster_labels)) < 2:
                return {
                    'success': False,
                    'error': 'Need at least 2 clusters for Calinski-Harabasz Index',
                    'score': None
                }
            
            score = calinski_harabasz_score(X, cluster_labels)
            
            return {
                'success': True,
                'score': float(score),
                'n_clusters': int(len(set(cluster_labels))),
                'description': 'Calinski-Harabasz Index (Variance Ratio Criterion) measures cluster quality. Higher is better. Ratio of between-cluster variance to within-cluster variance.'
            }
        except Exception as e:
            logger.error(f"Error calculating Calinski-Harabasz Index: {e}")
            return {
                'success': False,
                'error': str(e),
                'score': None
            }
