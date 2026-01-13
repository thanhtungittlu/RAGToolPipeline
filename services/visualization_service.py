"""
Visualization Service - UMAP and t-SNE for Embedding Visualization
Based on guide: "Evaluating Embedding Quality Before Ingesting into Vector Database"
"""
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("numpy not available, visualization will not work")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    logger.warning("umap-learn not available, UMAP visualization will not work")

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False
    logger.warning("sklearn not available, t-SNE visualization will not work")


class VisualizationService:
    """Service for embedding visualization using UMAP and t-SNE"""
    
    @staticmethod
    def umap_reduction(embeddings: List[List[float]], 
                      n_components: int = 2,
                      n_neighbors: int = 15,
                      min_dist: float = 0.1,
                      metric: str = 'cosine',
                      random_state: int = 42) -> Dict[str, Any]:
        """
        Reduce embedding dimensions using UMAP
        
        Recommended by guide: Fast, scalable, preserves global structure
        
        Args:
            embeddings: List of embedding vectors
            n_components: Number of dimensions for output (2 for visualization, 10+ for clustering)
            n_neighbors: Number of neighbors (15 for viz, 30 for clustering)
            min_dist: Minimum distance between points (0.1 for viz, 0.0 for clustering)
            metric: Distance metric ('cosine', 'euclidean', etc.)
            random_state: Random seed for reproducibility
        
        Returns:
            Dict with 2D/3D coordinates and metadata
        """
        if not HAS_UMAP or not HAS_NUMPY:
            return {
                'success': False,
                'error': 'umap-learn or numpy not available',
                'coordinates': None
            }
        
        try:
            X = np.array(embeddings)
            
            # Create UMAP reducer
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                metric=metric,
                random_state=random_state
            )
            
            # Fit and transform
            embedding_2d = reducer.fit_transform(X)
            
            # Convert to list of lists for JSON serialization
            coordinates = embedding_2d.tolist()
            
            return {
                'success': True,
                'coordinates': coordinates,
                'method': 'UMAP',
                'n_components': n_components,
                'n_neighbors': n_neighbors,
                'min_dist': min_dist,
                'metric': metric,
                'n_samples': len(embeddings),
                'description': 'UMAP dimensionality reduction. Preserves both local and global structure. Fast and scalable.'
            }
        except Exception as e:
            logger.error(f"Error in UMAP reduction: {e}")
            return {
                'success': False,
                'error': str(e),
                'coordinates': None
            }
    
    @staticmethod
    def tsne_reduction(embeddings: List[List[float]],
                      n_components: int = 2,
                      perplexity: float = 30.0,
                      random_state: int = 42,
                      n_iter: int = 1000) -> Dict[str, Any]:
        """
        Reduce embedding dimensions using t-SNE
        
        Note from guide: Slower than UMAP, mainly preserves local structure
        
        Args:
            embeddings: List of embedding vectors
            n_components: Number of dimensions (typically 2)
            perplexity: Balance between local/global structure (typically 5-50)
            random_state: Random seed
            n_iter: Number of iterations
        
        Returns:
            Dict with 2D coordinates and metadata
        """
        if not HAS_TSNE or not HAS_NUMPY:
            return {
                'success': False,
                'error': 'sklearn or numpy not available',
                'coordinates': None
            }
        
        try:
            X = np.array(embeddings)
            
            # Limit perplexity based on sample size
            n_samples = len(embeddings)
            if perplexity >= n_samples:
                perplexity = max(5, n_samples - 1)
            
            # Create t-SNE reducer
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
                n_iter=n_iter,
                metric='cosine'
            )
            
            # Fit and transform
            embedding_2d = reducer.fit_transform(X)
            
            # Convert to list of lists
            coordinates = embedding_2d.tolist()
            
            return {
                'success': True,
                'coordinates': coordinates,
                'method': 't-SNE',
                'n_components': n_components,
                'perplexity': perplexity,
                'n_iter': n_iter,
                'n_samples': len(embeddings),
                'description': 't-SNE dimensionality reduction. Preserves local structure well. Slower than UMAP for large datasets.'
            }
        except Exception as e:
            logger.error(f"Error in t-SNE reduction: {e}")
            return {
                'success': False,
                'error': str(e),
                'coordinates': None
            }
    
    @staticmethod
    def prepare_visualization_data(embeddings: List[List[float]], 
                                  labels: Optional[List[str]] = None,
                                  chunks: Optional[List[Dict[str, Any]]] = None,
                                  method: str = 'umap',
                                  n_components: int = 2) -> Dict[str, Any]:
        """
        Prepare data for visualization with metadata
        
        Args:
            embeddings: List of embedding vectors
            labels: Optional cluster labels or category labels
            chunks: Optional chunk metadata (for tooltips/hover info)
            method: 'umap' (recommended) or 'tsne'
        
        Returns:
            Dict with coordinates, labels, and metadata for frontend visualization
        """
        try:
            # Reduce dimensions - Always use UMAP
            reduction_result = VisualizationService.umap_reduction(
                embeddings,
                n_components=n_components
            )
            
            if not reduction_result.get('success'):
                return reduction_result
            
            coordinates = reduction_result.get('coordinates', [])
            
            # Prepare data points
            data_points = []
            for i, coord in enumerate(coordinates):
                point = {
                    'x': coord[0] if len(coord) > 0 else 0,
                    'y': coord[1] if len(coord) > 1 else 0,
                    'z': coord[2] if len(coord) > 2 else None,  # 3D if available
                    'index': i
                }
                
                # Add label if provided
                if labels and i < len(labels):
                    point['label'] = labels[i]
                
                # Add chunk metadata if provided
                if chunks and i < len(chunks):
                    chunk = chunks[i]
                    point['chunk_id'] = chunk.get('chunk_id', i)
                    point['filename'] = chunk.get('filename', '')
                    point['text_preview'] = chunk.get('text', '')[:200] if chunk.get('text') else ''
                    point['full_text'] = chunk.get('text', '')  # Full text for tooltip
                    point['position'] = chunk.get('position', 0)
                
                data_points.append(point)
            
            return {
                'success': True,
                'data': {
                    'points': data_points,
                    'method': reduction_result.get('method'),
                    'n_components': reduction_result.get('n_components', 2),
                    'n_samples': len(embeddings),
                    'description': reduction_result.get('description', '')
                }
            }
        except Exception as e:
            logger.error(f"Error preparing visualization data: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': None
            }
