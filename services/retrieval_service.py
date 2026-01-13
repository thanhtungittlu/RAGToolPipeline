"""
Retrieval Service - Layer 3: Retrieval Quality Evaluation
Based on guide: "Evaluating Embedding Quality Before Ingesting into Vector Database"
"""
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
import math

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("numpy not available, some metrics will not work")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("sklearn not available, similarity search will not work")


class RetrievalService:
    """Service for evaluating retrieval quality (Layer 3)"""
    
    @staticmethod
    def cosine_similarity_custom(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not HAS_NUMPY:
            # Manual calculation
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(a * a for a in vec2))
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            return dot_product / (magnitude1 * magnitude2)
        else:
            vec1_arr = np.array(vec1)
            vec2_arr = np.array(vec2)
            return float(np.dot(vec1_arr, vec2_arr) / (np.linalg.norm(vec1_arr) * np.linalg.norm(vec2_arr)))
    
    @staticmethod
    def search_similar(query_embedding: List[float], 
                      document_embeddings: List[List[float]], 
                      top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for most similar documents using cosine similarity
        
        Returns:
            List of (index, similarity_score) tuples, sorted by score descending
        """
        similarities = []
        
        for i, doc_emb in enumerate(document_embeddings):
            if HAS_SKLEARN:
                sim = float(cosine_similarity([query_embedding], [doc_emb])[0][0])
            else:
                sim = RetrievalService.cosine_similarity_custom(query_embedding, doc_emb)
            similarities.append((i, sim))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    @staticmethod
    def precision_at_k(retrieved_indices: List[int], 
                      relevant_indices: Set[int], 
                      k: int) -> float:
        """
        Calculate Precision@K
        
        Args:
            retrieved_indices: List of retrieved document indices (top-K)
            relevant_indices: Set of relevant document indices
            k: K value for precision@K
        
        Returns:
            Precision@K score (0.0 to 1.0)
        """
        if k == 0:
            return 0.0
        
        top_k_retrieved = retrieved_indices[:k]
        relevant_in_top_k = sum(1 for idx in top_k_retrieved if idx in relevant_indices)
        
        return relevant_in_top_k / k
    
    @staticmethod
    def recall_at_k(retrieved_indices: List[int], 
                    relevant_indices: Set[int], 
                    k: int) -> float:
        """
        Calculate Recall@K
        
        Args:
            retrieved_indices: List of retrieved document indices
            relevant_indices: Set of relevant document indices
            k: K value for recall@K
        
        Returns:
            Recall@K score (0.0 to 1.0)
        """
        if not relevant_indices:
            return 0.0
        
        top_k_retrieved = retrieved_indices[:k]
        relevant_retrieved = sum(1 for idx in top_k_retrieved if idx in relevant_indices)
        
        return relevant_retrieved / len(relevant_indices)
    
    @staticmethod
    def mean_reciprocal_rank(test_queries: List[Dict[str, Any]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        
        Args:
            test_queries: List of dicts with 'retrieved_indices' and 'relevant_indices'
        
        Returns:
            MRR score (0.0 to 1.0)
        """
        if not test_queries:
            return 0.0
        
        reciprocal_ranks = []
        
        for query_data in test_queries:
            retrieved_indices = query_data.get('retrieved_indices', [])
            relevant_indices = set(query_data.get('relevant_indices', []))
            
            if not relevant_indices:
                continue
            
            # Find first relevant document in retrieved list
            for rank, idx in enumerate(retrieved_indices, start=1):
                if idx in relevant_indices:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                # No relevant document found
                reciprocal_ranks.append(0.0)
        
        if not reciprocal_ranks:
            return 0.0
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks)
    
    @staticmethod
    def ndcg_at_k(retrieved_indices: List[int], 
                  relevance_scores: Dict[int, float], 
                  k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K (NDCG@K)
        
        Args:
            retrieved_indices: List of retrieved document indices
            relevance_scores: Dict mapping document index to relevance score (0-1)
            k: K value for NDCG@K
        
        Returns:
            NDCG@K score (0.0 to 1.0)
        """
        if k == 0:
            return 0.0
        
        top_k_retrieved = retrieved_indices[:k]
        
        # Calculate DCG@K
        dcg = 0.0
        for i, idx in enumerate(top_k_retrieved, start=1):
            relevance = relevance_scores.get(idx, 0.0)
            dcg += relevance / math.log2(i + 1)
        
        # Calculate IDCG@K (Ideal DCG)
        ideal_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(rel / math.log2(i + 1) for i, rel in enumerate(ideal_relevances, start=1))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def evaluate_retrieval_quality(query_embedding: List[float],
                                   document_embeddings: List[List[float]],
                                   relevant_doc_indices: Set[int],
                                   k_values: List[int] = [5, 10],
                                   relevance_scores: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
        """
        Comprehensive retrieval quality evaluation
        
        Args:
            query_embedding: Query embedding vector
            document_embeddings: List of document embedding vectors
            relevant_doc_indices: Set of indices of relevant documents
            k_values: List of K values to evaluate (default: [5, 10])
            relevance_scores: Optional dict mapping doc index to relevance score (for NDCG)
        
        Returns:
            Dict with evaluation results
        """
        try:
            # Search for similar documents
            all_similarities = RetrievalService.search_similar(
                query_embedding, 
                document_embeddings, 
                top_k=max(k_values) if k_values else 10
            )
            
            retrieved_indices = [idx for idx, _ in all_similarities]
            
            results = {}
            
            # Precision@K and Recall@K
            for k in k_values:
                precision = RetrievalService.precision_at_k(retrieved_indices, relevant_doc_indices, k)
                recall = RetrievalService.recall_at_k(retrieved_indices, relevant_doc_indices, k)
                
                results[f'precision_at_{k}'] = {
                    'score': round(precision, 4),
                    'thresholds': {
                        'minimum': 0.5,
                        'good': 0.6,
                        'excellent': 0.8
                    },
                    'quality_level': 'EXCELLENT' if precision >= 0.8 else 'GOOD' if precision >= 0.6 else 'ACCEPTABLE' if precision >= 0.5 else 'NEEDS_IMPROVEMENT',
                    'description': f'Precision@K đo lường tỷ lệ documents liên quan trong top-{k} kết quả. Giá trị cao hơn là tốt hơn.'
                }
                
                results[f'recall_at_{k}'] = {
                    'score': round(recall, 4),
                    'thresholds': {
                        'minimum': 0.6,
                        'good': 0.7,
                        'excellent': 0.9
                    },
                    'quality_level': 'EXCELLENT' if recall >= 0.9 else 'GOOD' if recall >= 0.7 else 'ACCEPTABLE' if recall >= 0.6 else 'NEEDS_IMPROVEMENT',
                    'description': f'Recall@K đo lường tỷ lệ documents liên quan được lấy trong top-{k} kết quả. Giá trị cao hơn là tốt hơn.'
                }
                
                # NDCG@K if relevance scores provided
                if relevance_scores:
                    ndcg = RetrievalService.ndcg_at_k(retrieved_indices, relevance_scores, k)
                    results[f'ndcg_at_{k}'] = {
                        'score': round(ndcg, 4),
                        'thresholds': {
                            'minimum': 0.5,
                            'good': 0.7,
                            'excellent': 0.9
                        },
                        'quality_level': 'EXCELLENT' if ndcg >= 0.9 else 'GOOD' if ndcg >= 0.7 else 'ACCEPTABLE' if ndcg >= 0.5 else 'NEEDS_IMPROVEMENT',
                        'description': f'NDCG@K đo lường chất lượng ranking, xét cả mức độ liên quan và vị trí. Giá trị cao hơn là tốt hơn.'
                    }
            
            # Add retrieved results info
            results['retrieved_documents'] = [
                {
                    'index': idx,
                    'similarity': round(sim, 4),
                    'is_relevant': idx in relevant_doc_indices
                }
                for idx, sim in all_similarities[:max(k_values) if k_values else 10]
            ]
            
            return {
                'success': True,
                'results': results
            }
        except Exception as e:
            logger.error(f"Error evaluating retrieval quality: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': None
            }
    
    @staticmethod
    def evaluate_multiple_queries(test_queries: List[Dict[str, Any]],
                                  document_embeddings: List[List[float]],
                                  k_values: List[int] = [5, 10]) -> Dict[str, Any]:
        """
        Evaluate retrieval quality for multiple test queries
        
        Args:
            test_queries: List of dicts with:
                - 'query_embedding': List[float]
                - 'relevant_doc_indices': Set[int] or List[int]
                - 'relevance_scores': Optional[Dict[int, float]]
            document_embeddings: List of document embedding vectors
            k_values: List of K values to evaluate
        
        Returns:
            Dict with aggregated evaluation results
        """
        try:
            all_precisions = {k: [] for k in k_values}
            all_recalls = {k: [] for k in k_values}
            all_ndcgs = {k: [] for k in k_values}
            reciprocal_ranks = []
            
            for query_data in test_queries:
                query_emb = query_data.get('query_embedding')
                relevant_indices = query_data.get('relevant_doc_indices', [])
                relevance_scores = query_data.get('relevance_scores')
                
                if not query_emb or not relevant_indices:
                    continue
                
                # Convert to set if needed
                if isinstance(relevant_indices, list):
                    relevant_indices = set(relevant_indices)
                
                # Evaluate this query
                query_result = RetrievalService.evaluate_retrieval_quality(
                    query_emb,
                    document_embeddings,
                    relevant_indices,
                    k_values,
                    relevance_scores
                )
                
                if not query_result.get('success'):
                    continue
                
                results = query_result.get('results', {})
                
                # Collect metrics
                for k in k_values:
                    if f'precision_at_{k}' in results:
                        all_precisions[k].append(results[f'precision_at_{k}']['score'])
                    if f'recall_at_{k}' in results:
                        all_recalls[k].append(results[f'recall_at_{k}']['score'])
                    if f'ndcg_at_{k}' in results and results[f'ndcg_at_{k}']:
                        all_ndcgs[k].append(results[f'ndcg_at_{k}']['score'])
                
                # Calculate reciprocal rank for this query
                retrieved = [doc['index'] for doc in results.get('retrieved_documents', [])]
                for rank, idx in enumerate(retrieved, start=1):
                    if idx in relevant_indices:
                        reciprocal_ranks.append(1.0 / rank)
                        break
                else:
                    reciprocal_ranks.append(0.0)
            
            # Aggregate results
            aggregated = {
                'n_queries': len(test_queries),
                'metrics': {}
            }
            
            for k in k_values:
                if all_precisions[k]:
                    avg_precision = sum(all_precisions[k]) / len(all_precisions[k])
                    aggregated['metrics'][f'precision_at_{k}'] = {
                        'mean': round(avg_precision, 4),
                        'thresholds': {'minimum': 0.5, 'good': 0.6, 'excellent': 0.8},
                        'quality_level': 'EXCELLENT' if avg_precision >= 0.8 else 'GOOD' if avg_precision >= 0.6 else 'ACCEPTABLE' if avg_precision >= 0.5 else 'NEEDS_IMPROVEMENT',
                        'description': f'Average Precision@K across all queries. Higher is better.'
                    }
                
                if all_recalls[k]:
                    avg_recall = sum(all_recalls[k]) / len(all_recalls[k])
                    aggregated['metrics'][f'recall_at_{k}'] = {
                        'mean': round(avg_recall, 4),
                        'thresholds': {'minimum': 0.6, 'good': 0.7, 'excellent': 0.9},
                        'quality_level': 'EXCELLENT' if avg_recall >= 0.9 else 'GOOD' if avg_recall >= 0.7 else 'ACCEPTABLE' if avg_recall >= 0.6 else 'NEEDS_IMPROVEMENT',
                        'description': f'Average Recall@K across all queries. Higher is better.'
                    }
                
                if all_ndcgs[k]:
                    avg_ndcg = sum(all_ndcgs[k]) / len(all_ndcgs[k])
                    aggregated['metrics'][f'ndcg_at_{k}'] = {
                        'mean': round(avg_ndcg, 4),
                        'thresholds': {'minimum': 0.5, 'good': 0.7, 'excellent': 0.9},
                        'quality_level': 'EXCELLENT' if avg_ndcg >= 0.9 else 'GOOD' if avg_ndcg >= 0.7 else 'ACCEPTABLE' if avg_ndcg >= 0.5 else 'NEEDS_IMPROVEMENT',
                        'description': f'Average NDCG@K across all queries. Higher is better.'
                    }
            
            # MRR
            if reciprocal_ranks:
                mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
                aggregated['metrics']['mrr'] = {
                    'score': round(mrr, 4),
                    'thresholds': {'minimum': 0.4, 'good': 0.5, 'excellent': 0.7},
                    'quality_level': 'EXCELLENT' if mrr >= 0.7 else 'GOOD' if mrr >= 0.5 else 'ACCEPTABLE' if mrr >= 0.4 else 'NEEDS_IMPROVEMENT',
                    'description': 'Mean Reciprocal Rank đo lường vị trí trung bình của kết quả liên quan đầu tiên. Giá trị cao hơn là tốt hơn.'
                }
            
            return {
                'success': True,
                'results': aggregated
            }
        except Exception as e:
            logger.error(f"Error evaluating multiple queries: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': None
            }
