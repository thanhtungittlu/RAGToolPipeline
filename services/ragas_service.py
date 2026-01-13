"""
RAGAS Service - RAG Assessment Framework Integration
Based on guide: "Evaluating Embedding Quality Before Ingesting into Vector Database"

RAGAS Metrics:
- Faithfulness: Answer is faithful to the context (no hallucination)
- Answer Relevancy: Answer correctly addresses the question
- Context Precision: How many retrieved chunks are actually relevant
- Context Recall: Whether sufficient relevant information was retrieved
"""
import logging
from typing import List, Dict, Any, Optional
import re

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("requests not available, LLM-as-Judge will not work")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("numpy not available, similarity calculations will not work")


class RAGASService:
    """Service for RAGAS-style evaluation"""
    
    @staticmethod
    def cosine_similarity_custom(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not HAS_NUMPY:
            import math
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
    def extract_claims(text: str) -> List[str]:
        """
        Extract factual claims from text (simple heuristic-based)
        
        In production, this should use LLM or NLP model
        """
        # Simple heuristic: split by sentence and filter
        sentences = re.split(r'[.!?]+\s+', text)
        claims = [s.strip() for s in sentences if len(s.strip()) > 20 and not s.strip().startswith(('If', 'When', 'Should'))]
        return claims[:10]  # Limit to top 10
    
    @staticmethod
    def check_claim_in_context(claim: str, contexts: List[str]) -> bool:
        """
        Check if a claim is supported by contexts (simple keyword matching)
        
        In production, should use semantic similarity or LLM
        """
        claim_lower = claim.lower()
        claim_words = set(re.findall(r'\w+', claim_lower))
        
        for context in contexts:
            context_lower = context.lower()
            context_words = set(re.findall(r'\w+', context_lower))
            
            # Check if significant portion of claim words are in context
            overlap = len(claim_words & context_words)
            if overlap >= max(3, len(claim_words) * 0.5):
                return True
        
        return False
    
    @staticmethod
    def evaluate_faithfulness(answer: str, contexts: List[str]) -> Dict[str, Any]:
        """
        Evaluate Faithfulness: Answer is faithful to the context
        
        Faithfulness = (Claims supported by context) / (Total claims in answer)
        
        Target: ≥ 0.7 (Good), ≥ 0.85 (Excellent - compliance-grade)
        """
        try:
            # Extract claims from answer
            claims = RAGASService.extract_claims(answer)
            
            if not claims:
                return {
                    'success': True,
                    'score': 1.0,  # No claims to verify, assume faithful
                    'description': 'No claims found in answer. Faithfulness score set to 1.0.',
                    'claims_checked': 0
                }
            
            # Check each claim against contexts
            supported_claims = 0
            claim_details = []
            
            for claim in claims:
                is_supported = RAGASService.check_claim_in_context(claim, contexts)
                claim_details.append({
                    'claim': claim[:100],  # Truncate for display
                    'supported': is_supported
                })
                if is_supported:
                    supported_claims += 1
            
            faithfulness_score = supported_claims / len(claims) if claims else 1.0
            
            # Quality assessment
            if faithfulness_score >= 0.85:
                quality_level = 'EXCELLENT'
            elif faithfulness_score >= 0.7:
                quality_level = 'GOOD'
            else:
                quality_level = 'NEEDS_IMPROVEMENT'
            
            return {
                'success': True,
                'score': round(faithfulness_score, 4),
                'quality_level': quality_level,
                'thresholds': {
                    'minimum': 0.7,
                    'good': 0.7,
                    'excellent': 0.85
                },
                'claims_total': len(claims),
                'claims_supported': supported_claims,
                'claims_details': claim_details[:5],  # Limit for display
                    'description': 'Faithfulness đo lường mức độ answer trung thành với context được cung cấp. Điểm số cao hơn cho thấy ít nguy cơ hallucination hơn.'
            }
        except Exception as e:
            logger.error(f"Error evaluating faithfulness: {e}")
            return {
                'success': False,
                'error': str(e),
                'score': None
            }
    
    @staticmethod
    def evaluate_answer_relevancy(question: str, answer: str, answer_embedding: Optional[List[float]] = None, question_embedding: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Evaluate Answer Relevancy: Answer correctly addresses the question
        
        Simple heuristic-based approach. In production, use LLM to generate questions from answer
        and compare with original question.
        
        Target: ≥ 0.7 (Good), ≥ 0.85 (Excellent)
        """
        try:
            # Simple heuristic: keyword overlap and semantic similarity
            question_words = set(re.findall(r'\w+', question.lower()))
            answer_words = set(re.findall(r'\w+', answer.lower()))
            
            # Keyword overlap
            overlap = len(question_words & answer_words)
            keyword_score = overlap / len(question_words) if question_words else 0.0
            
            # If embeddings provided, use semantic similarity
            semantic_score = None
            if answer_embedding and question_embedding:
                semantic_score = RAGASService.cosine_similarity_custom(answer_embedding, question_embedding)
                # Normalize to 0-1 (cosine similarity is -1 to 1, but typically 0 to 1)
                semantic_score = max(0, semantic_score)
            
            # Combine scores (weighted average)
            if semantic_score is not None:
                relevancy_score = (keyword_score * 0.3 + semantic_score * 0.7)
            else:
                relevancy_score = keyword_score
            
            # Quality assessment
            if relevancy_score >= 0.85:
                quality_level = 'EXCELLENT'
            elif relevancy_score >= 0.7:
                quality_level = 'GOOD'
            else:
                quality_level = 'NEEDS_IMPROVEMENT'
            
            return {
                'success': True,
                'score': round(relevancy_score, 4),
                'quality_level': quality_level,
                'thresholds': {
                    'minimum': 0.7,
                    'good': 0.7,
                    'excellent': 0.85
                },
                'keyword_score': round(keyword_score, 4),
                'semantic_score': round(semantic_score, 4) if semantic_score else None,
                'description': 'Answer Relevancy đo lường mức độ answer trả lời đúng câu hỏi. Điểm số cao hơn cho thấy answer liên quan hơn.'
            }
        except Exception as e:
            logger.error(f"Error evaluating answer relevancy: {e}")
            return {
                'success': False,
                'error': str(e),
                'score': None
            }
    
    @staticmethod
    def evaluate_context_precision(retrieved_contexts: List[str], relevant_contexts: List[str]) -> Dict[str, Any]:
        """
        Evaluate Context Precision: How many retrieved chunks are actually relevant
        
        Precision = (Relevant contexts in retrieved) / (Total retrieved contexts)
        
        Target: ≥ 0.6 (Minimum), ≥ 0.7 (Good), ≥ 0.8 (Excellent)
        """
        try:
            if not retrieved_contexts:
                return {
                    'success': True,
                    'score': 0.0,
                    'description': 'No contexts retrieved.'
                }
            
            # Simple heuristic: check if retrieved contexts are in relevant list
            # In production, use semantic similarity or LLM judgment
            relevant_set = set(relevant_contexts)
            retrieved_set = set(retrieved_contexts)
            
            # Exact match
            relevant_retrieved = len(retrieved_set & relevant_set)
            precision = relevant_retrieved / len(retrieved_contexts)
            
            # Quality assessment
            if precision >= 0.8:
                quality_level = 'EXCELLENT'
            elif precision >= 0.7:
                quality_level = 'GOOD'
            elif precision >= 0.6:
                quality_level = 'ACCEPTABLE'
            else:
                quality_level = 'NEEDS_IMPROVEMENT'
            
            return {
                'success': True,
                'score': round(precision, 4),
                'quality_level': quality_level,
                'thresholds': {
                    'minimum': 0.6,
                    'good': 0.7,
                    'excellent': 0.8
                },
                'relevant_retrieved': relevant_retrieved,
                'total_retrieved': len(retrieved_contexts),
                'description': 'Context Precision đo lường tỷ lệ contexts liên quan trong các kết quả được lấy. Điểm số cao hơn cho thấy ít noise hơn.'
            }
        except Exception as e:
            logger.error(f"Error evaluating context precision: {e}")
            return {
                'success': False,
                'error': str(e),
                'score': None
            }
    
    @staticmethod
    def evaluate_context_recall(retrieved_contexts: List[str], all_relevant_contexts: List[str]) -> Dict[str, Any]:
        """
        Evaluate Context Recall: Whether sufficient relevant information was retrieved
        
        Recall = (Relevant contexts retrieved) / (Total relevant contexts)
        
        Target: ≥ 0.7 (Minimum), ≥ 0.8 (Good), ≥ 0.9 (Excellent)
        """
        try:
            if not all_relevant_contexts:
                return {
                    'success': True,
                    'score': 1.0,  # No relevant contexts means perfect recall
                    'description': 'No relevant contexts defined.'
                }
            
            # Simple heuristic: check how many relevant contexts were retrieved
            relevant_set = set(all_relevant_contexts)
            retrieved_set = set(retrieved_contexts)
            
            relevant_retrieved = len(retrieved_set & relevant_set)
            recall = relevant_retrieved / len(all_relevant_contexts)
            
            # Quality assessment
            if recall >= 0.9:
                quality_level = 'EXCELLENT'
            elif recall >= 0.8:
                quality_level = 'GOOD'
            elif recall >= 0.7:
                quality_level = 'ACCEPTABLE'
            else:
                quality_level = 'NEEDS_IMPROVEMENT'
            
            return {
                'success': True,
                'score': round(recall, 4),
                'quality_level': quality_level,
                'thresholds': {
                    'minimum': 0.7,
                    'good': 0.8,
                    'excellent': 0.9
                },
                'relevant_retrieved': relevant_retrieved,
                'total_relevant': len(all_relevant_contexts),
                'description': 'Context Recall đo lường tỷ lệ contexts liên quan đã được lấy. Điểm số cao hơn cho thấy thông tin đầy đủ hơn.'
            }
        except Exception as e:
            logger.error(f"Error evaluating context recall: {e}")
            return {
                'success': False,
                'error': str(e),
                'score': None
            }
    
    @staticmethod
    def comprehensive_ragas_evaluation(question: str,
                                      answer: str,
                                      contexts: List[str],
                                      relevant_contexts: Optional[List[str]] = None,
                                      answer_embedding: Optional[List[float]] = None,
                                      question_embedding: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Comprehensive RAGAS evaluation
        
        Evaluates all RAGAS metrics:
        - Faithfulness
        - Answer Relevancy
        - Context Precision (if relevant_contexts provided)
        - Context Recall (if relevant_contexts provided)
        """
        try:
            results = {}
            
            # Faithfulness
            faithfulness_result = RAGASService.evaluate_faithfulness(answer, contexts)
            results['faithfulness'] = faithfulness_result
            
            # Answer Relevancy
            relevancy_result = RAGASService.evaluate_answer_relevancy(
                question, answer, answer_embedding, question_embedding
            )
            results['answer_relevancy'] = relevancy_result
            
            # Context Precision & Recall (if relevant_contexts provided)
            if relevant_contexts is not None:
                precision_result = RAGASService.evaluate_context_precision(contexts, relevant_contexts)
                results['context_precision'] = precision_result
                
                recall_result = RAGASService.evaluate_context_recall(contexts, relevant_contexts)
                results['context_recall'] = recall_result
            
            # Overall quality assessment
            scores = [
                results.get('faithfulness', {}).get('score', 0),
                results.get('answer_relevancy', {}).get('score', 0)
            ]
            if 'context_precision' in results:
                scores.append(results['context_precision'].get('score', 0))
            if 'context_recall' in results:
                scores.append(results['context_recall'].get('score', 0))
            
            avg_score = sum(scores) / len(scores) if scores else 0.0
            
            if avg_score >= 0.85:
                overall_quality = 'EXCELLENT'
            elif avg_score >= 0.7:
                overall_quality = 'GOOD'
            else:
                overall_quality = 'NEEDS_IMPROVEMENT'
            
            return {
                'success': True,
                'results': results,
                'overall_quality': overall_quality,
                'average_score': round(avg_score, 4)
            }
        except Exception as e:
            logger.error(f"Error in comprehensive RAGAS evaluation: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': None
            }
