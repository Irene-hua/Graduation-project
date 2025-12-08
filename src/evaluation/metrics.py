"""
Metrics Module
Calculate evaluation metrics for RAG system
"""

from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import re


class Metrics:
    """Evaluation metrics for RAG system"""
    
    @staticmethod
    def calculate_retrieval_metrics(retrieved_ids: List[str],
                                   relevant_ids: List[str],
                                   k: int = None) -> Dict:
        """
        Calculate retrieval metrics
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
            k: Top-k cutoff (None for all retrieved)
            
        Returns:
            Dict with precision, recall, F1, and other metrics
        """
        if k:
            retrieved_ids = retrieved_ids[:k]
        
        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)
        
        # Calculate metrics
        true_positives = len(retrieved_set & relevant_set)
        false_positives = len(retrieved_set - relevant_set)
        false_negatives = len(relevant_set - retrieved_set)
        
        precision = true_positives / len(retrieved_set) if retrieved_set else 0
        recall = true_positives / len(relevant_set) if relevant_set else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Additional metrics
        map_score = Metrics._calculate_map([retrieved_ids], [relevant_ids])
        mrr = Metrics._calculate_mrr(retrieved_ids, relevant_ids)
        ndcg = Metrics._calculate_ndcg(retrieved_ids, relevant_ids, k)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'map': map_score,
            'mrr': mrr,
            'ndcg': ndcg
        }
    
    @staticmethod
    def _calculate_map(retrieved_lists: List[List[str]], 
                       relevant_lists: List[List[str]]) -> float:
        """Calculate Mean Average Precision"""
        if len(retrieved_lists) != len(relevant_lists):
            raise ValueError("Lists must have same length")
        
        average_precisions = []
        
        for retrieved, relevant in zip(retrieved_lists, relevant_lists):
            if not relevant:
                continue
            
            relevant_set = set(relevant)
            precisions = []
            num_relevant = 0
            
            for i, doc_id in enumerate(retrieved):
                if doc_id in relevant_set:
                    num_relevant += 1
                    precision_at_i = num_relevant / (i + 1)
                    precisions.append(precision_at_i)
            
            if precisions:
                average_precisions.append(np.mean(precisions))
        
        return np.mean(average_precisions) if average_precisions else 0
    
    @staticmethod
    def _calculate_mrr(retrieved: List[str], relevant: List[str]) -> float:
        """Calculate Mean Reciprocal Rank"""
        relevant_set = set(relevant)
        
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    @staticmethod
    def _calculate_ndcg(retrieved: List[str], relevant: List[str], k: int = None) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        if k:
            retrieved = retrieved[:k]
        
        relevant_set = set(relevant)
        
        # Calculate DCG
        dcg = 0
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant_set:
                dcg += 1 / np.log2(i + 2)  # i+2 because index starts at 0
        
        # Calculate Ideal DCG
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), len(retrieved))))
        
        return dcg / idcg if idcg > 0 else 0
    
    @staticmethod
    def calculate_answer_metrics(predicted_answers: List[str],
                                ground_truth_answers: List[str]) -> Dict:
        """
        Calculate answer quality metrics
        
        Args:
            predicted_answers: List of predicted answers
            ground_truth_answers: List of ground truth answers
            
        Returns:
            Dict with various answer quality metrics
        """
        if len(predicted_answers) != len(ground_truth_answers):
            raise ValueError("Number of predictions must match ground truth")
        
        exact_matches = 0
        token_f1_scores = []
        bleu_scores = []
        
        for pred, truth in zip(predicted_answers, ground_truth_answers):
            # Exact match
            if pred.strip().lower() == truth.strip().lower():
                exact_matches += 1
            
            # Token-level F1
            token_f1 = Metrics._calculate_token_f1(pred, truth)
            token_f1_scores.append(token_f1)
        
        return {
            'exact_match': exact_matches / len(predicted_answers),
            'avg_token_f1': np.mean(token_f1_scores),
            'num_samples': len(predicted_answers)
        }
    
    @staticmethod
    def _calculate_token_f1(prediction: str, ground_truth: str) -> float:
        """Calculate token-level F1 score"""
        pred_tokens = set(Metrics._normalize_text(prediction).split())
        truth_tokens = set(Metrics._normalize_text(ground_truth).split())
        
        if not pred_tokens or not truth_tokens:
            return 0
        
        common = pred_tokens & truth_tokens
        
        if not common:
            return 0
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for comparison"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def calculate_performance_metrics(inference_times: List[float],
                                    memory_usages: List[float] = None) -> Dict:
        """
        Calculate performance metrics
        
        Args:
            inference_times: List of inference times in seconds
            memory_usages: Optional list of memory usage in MB
            
        Returns:
            Dict with performance statistics
        """
        metrics = {
            'avg_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'median_inference_time': np.median(inference_times),
            'p95_inference_time': np.percentile(inference_times, 95),
            'p99_inference_time': np.percentile(inference_times, 99)
        }
        
        if memory_usages:
            metrics.update({
                'avg_memory_mb': np.mean(memory_usages),
                'max_memory_mb': np.max(memory_usages),
                'min_memory_mb': np.min(memory_usages)
            })
        
        return metrics
    
    @staticmethod
    def compare_systems(baseline_metrics: Dict,
                       comparison_metrics: Dict) -> Dict:
        """
        Compare two systems
        
        Args:
            baseline_metrics: Metrics from baseline system
            comparison_metrics: Metrics from comparison system
            
        Returns:
            Dict with comparison results
        """
        comparison = {}
        
        for key in baseline_metrics:
            if key in comparison_metrics:
                baseline_val = baseline_metrics[key]
                comparison_val = comparison_metrics[key]
                
                if isinstance(baseline_val, (int, float)):
                    diff = comparison_val - baseline_val
                    pct_change = (diff / baseline_val * 100) if baseline_val != 0 else 0
                    
                    comparison[key] = {
                        'baseline': baseline_val,
                        'comparison': comparison_val,
                        'difference': diff,
                        'percent_change': pct_change
                    }
        
        return comparison
