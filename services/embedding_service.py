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
    from sklearn.metrics.pairwise import cosine_distances
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
                'description': 'Silhouette Score có giá trị từ -1 đến 1. Giá trị cao hơn là tốt hơn. Đo lường mức độ tương tự của một điểm với cluster của nó so với các cluster khác.'
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
                'description': 'Davies-Bouldin Index đo lường chất lượng cluster. Giá trị thấp hơn là tốt hơn. Đo lường tỷ lệ tương tự trung bình giữa các cluster.'
            }
        except Exception as e:
            logger.error(f"Error calculating Davies-Bouldin Index: {e}")
            return {
                'success': False,
                'error': str(e),
                'score': None
            }
    
    @staticmethod
    def evaluate_intra_cluster_distance(embeddings: List[List[float]], n_clusters: int = 5, method: str = 'centroid') -> Dict:
        """
        Evaluate intra-cluster distance (cluster compactness)
        
        Args:
            embeddings: List of embedding vectors
            n_clusters: Number of clusters
            method: 'centroid' or 'average'
        
        Returns:
            Dict with cluster distances and average
        """
        if not HAS_SKLEARN or not HAS_NUMPY:
            return {
                'success': False,
                'error': 'sklearn or numpy not available',
                'distances': None
            }
        
        try:
            from sklearn.metrics.pairwise import cosine_distances
            
            X = np.array(embeddings)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=min(n_clusters, len(embeddings)), random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            if len(set(cluster_labels)) < 2:
                return {
                    'success': False,
                    'error': 'Need at least 2 clusters for intra-cluster distance',
                    'distances': None
                }
            
            unique_labels = np.unique(cluster_labels)
            distances = {}
            
            for label in unique_labels:
                cluster_points = X[cluster_labels == label]
                
                if method == 'centroid':
                    centroid = np.mean(cluster_points, axis=0)
                    dist = np.mean(cosine_distances(cluster_points, [centroid]))
                elif method == 'average':
                    if len(cluster_points) > 1:
                        dist_matrix = cosine_distances(cluster_points)
                        # Get upper triangle (exclude diagonal)
                        triu_indices = np.triu_indices(len(cluster_points), k=1)
                        dist = np.mean(dist_matrix[triu_indices])
                    else:
                        dist = 0.0
                else:
                    return {
                        'success': False,
                        'error': f'Unknown method: {method}',
                        'distances': None
                    }
                
                distances[f"cluster_{int(label)}"] = float(dist)
            
            distances['average'] = float(np.mean(list(distances.values())))
            
            return {
                'success': True,
                'distances': distances,
                'method': method,
                'n_clusters': int(len(unique_labels)),
                'description': f'Intra-cluster distance ({method} method). Lower is better - indicates more compact clusters.'
            }
        except Exception as e:
            logger.error(f"Error calculating intra-cluster distance: {e}")
            return {
                'success': False,
                'error': str(e),
                'distances': None
            }
    
    @staticmethod
    def comprehensive_embedding_evaluation(embeddings: List[List[float]], n_clusters: int = 10) -> Dict:
        """
        Comprehensive embedding quality evaluation (Layer 2)
        
        Evaluates:
        - Silhouette Score
        - Davies-Bouldin Index
        - Intra-cluster Distance
        
        Returns quality assessment based on thresholds from the guide.
        """
        if not HAS_SKLEARN or not HAS_NUMPY:
            return {
                'success': False,
                'error': 'sklearn or numpy not available',
                'results': None
            }
        
        try:
            from sklearn.metrics.pairwise import cosine_distances
            
            X = np.array(embeddings)
            
            n_samples = len(embeddings)
            
            if n_samples < 2:
                return {
                    'success': False,
                    'error': 'Need at least 2 embeddings for evaluation',
                    'results': None
                }
            
            # Adjust n_clusters if needed - must be less than n_samples for some metrics
            # For 2 samples, we can only have 1 cluster, which won't work for clustering metrics
            # So we need at least 3 samples for proper evaluation
            if n_samples == 2:
                return {
                    'success': False,
                    'error': 'Need at least 3 embeddings for clustering-based evaluation. With 2 embeddings, clustering metrics cannot be calculated.',
                    'results': None
                }
            
            # Ensure n_clusters < n_samples for Calinski-Harabasz
            n_clusters = min(n_clusters, n_samples - 1)
            if n_clusters < 2:
                n_clusters = 2  # Minimum 2 clusters
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            actual_clusters = len(set(cluster_labels))
            if actual_clusters < 2:
                return {
                    'success': False,
                    'error': 'Need at least 2 clusters for evaluation',
                    'results': None
                }
            
            # Calculate all metrics with error handling
            silhouette = None
            silhouette_error = None
            davies_bouldin = None
            davies_bouldin_error = None
            
            try:
                silhouette = silhouette_score(X, cluster_labels, metric='cosine')
            except Exception as e:
                logger.warning(f"Could not calculate Silhouette Score: {e}")
                silhouette_error = str(e)
            
            try:
                davies_bouldin = davies_bouldin_score(X, cluster_labels)
            except Exception as e:
                logger.warning(f"Could not calculate Davies-Bouldin Index: {e}")
                davies_bouldin_error = str(e)
            
            # Calinski-Harabasz requires: n_clusters < n_samples
            # If actual_clusters >= n_samples, skip this metric
            calinski_harabasz = None
            calinski_harabasz_error = None
            if actual_clusters < n_samples:
                try:
                    calinski_harabasz = calinski_harabasz_score(X, cluster_labels)
                except Exception as e:
                    logger.warning(f"Could not calculate Calinski-Harabasz Index: {e}")
                    calinski_harabasz_error = str(e)
            else:
                calinski_harabasz_error = f"Cannot calculate Calinski-Harabasz Index: number of clusters ({actual_clusters}) must be less than number of samples ({n_samples})"
            
            # Intra-cluster distance (centroid method)
            intra_distances = {}
            unique_labels = np.unique(cluster_labels)
            for label in unique_labels:
                cluster_points = X[cluster_labels == label]
                if len(cluster_points) > 0:
                    centroid = np.mean(cluster_points, axis=0)
                    dist = np.mean(cosine_distances(cluster_points, [centroid]))
                    intra_distances[f"cluster_{int(label)}"] = float(dist)
            avg_intra_distance = float(np.mean(list(intra_distances.values()))) if intra_distances else 0.0
            
            # Quality assessment based on guide thresholds
            # Only assess if both metrics are available
            quality_level = 'NEEDS_IMPROVEMENT'
            if silhouette is not None and davies_bouldin is not None:
                if silhouette >= 0.5 and davies_bouldin <= 1.0:
                    quality_level = 'EXCELLENT'
                elif silhouette >= 0.35 and davies_bouldin <= 1.5:
                    quality_level = 'GOOD'
                elif silhouette >= 0.25 and davies_bouldin <= 2.0:
                    quality_level = 'ACCEPTABLE'
            
            results = {}
            
            # Add Silhouette Score
            if silhouette is not None:
                results['silhouette_score'] = {
                    'score': float(silhouette),
                    'quality_level': EmbeddingService.get_embedding_quality_level('silhouette', silhouette),
                    'thresholds': {
                        'minimum': 0.25,
                        'good': 0.35,
                        'excellent': 0.50
                    },
                    'description': 'Silhouette Score có giá trị từ -1 đến 1. Giá trị cao hơn là tốt hơn. Đo lường mức độ tương tự của một điểm với cluster của nó so với các cluster khác.'
                }
            else:
                results['silhouette_score'] = {
                    'score': None,
                    'quality_level': 'N/A',
                    'error': silhouette_error or 'Cannot calculate Silhouette Score',
                    'description': 'Silhouette Score có giá trị từ -1 đến 1. Giá trị cao hơn là tốt hơn. Đo lường mức độ tương tự của một điểm với cluster của nó so với các cluster khác.'
                }
            
            # Add Davies-Bouldin Index
            if davies_bouldin is not None:
                results['davies_bouldin'] = {
                    'score': float(davies_bouldin),
                    'quality_level': EmbeddingService.get_embedding_quality_level('davies_bouldin', davies_bouldin),
                    'thresholds': {
                        'minimum': 2.0,
                        'good': 1.5,
                        'excellent': 1.0
                    },
                    'description': 'Davies-Bouldin Index đo lường chất lượng cluster. Giá trị thấp hơn là tốt hơn. Đo lường tỷ lệ tương tự trung bình giữa các cluster.'
                }
            else:
                results['davies_bouldin'] = {
                    'score': None,
                    'quality_level': 'N/A',
                    'error': davies_bouldin_error or 'Cannot calculate Davies-Bouldin Index',
                    'description': 'Davies-Bouldin Index đo lường chất lượng cluster. Giá trị thấp hơn là tốt hơn. Đo lường tỷ lệ tương tự trung bình giữa các cluster.'
                }
            
            # Add intra-cluster distance
            results['intra_cluster_distance'] = {
                'average': avg_intra_distance,
                'per_cluster': intra_distances,
                'description': 'Khoảng cách trong cluster (phương pháp centroid). Giá trị thấp hơn là tốt hơn - cho thấy các cluster gắn kết hơn.'
            }
            
            # Add overall quality and metadata
            results['overall_quality'] = quality_level
            results['n_clusters'] = int(actual_clusters)
            results['n_samples'] = n_samples
            
            # Add Calinski-Harabasz only if calculated successfully
            if calinski_harabasz is not None:
                results['calinski_harabasz_score'] = {
                    'score': float(calinski_harabasz),
                    'quality_level': EmbeddingService.get_embedding_quality_level('calinski_harabasz', calinski_harabasz),
                    'thresholds': {
                        'minimum': 0,
                        'good': 100,
                        'excellent': 200
                    },
                    'description': 'Calinski-Harabasz Index đo lường chất lượng cluster. Giá trị cao hơn là tốt hơn. Tỷ lệ giữa độ phân tán between-cluster và within-cluster.'
                }
            else:
                # Add error info for Calinski-Harabasz
                results['calinski_harabasz_score'] = {
                    'score': None,
                    'quality_level': 'N/A',
                    'error': calinski_harabasz_error or 'Cannot calculate Calinski-Harabasz Index',
                    'description': 'Calinski-Harabasz Index đo lường chất lượng cluster. Giá trị cao hơn là tốt hơn. Tỷ lệ giữa độ phân tán between-cluster và within-cluster. (Yêu cầu: số clusters < số samples)'
                }
            
            return {
                'success': True,
                'results': results
            }
        except Exception as e:
            logger.error(f"Error in comprehensive embedding evaluation: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': None
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
