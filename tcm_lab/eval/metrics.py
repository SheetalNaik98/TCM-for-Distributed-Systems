"""Metrics calculation for experiments"""

from typing import Dict, List, Any
import numpy as np
from collections import defaultdict

from tcm_lab.eval.harness import QueryResult
from tcm_lab.memory.base import BaseMemory


class MetricsCalculator:
    """Calculate metrics from experiment results"""
    
    def calculate_all(self, results: List[QueryResult], memory: BaseMemory) -> Dict[str, Any]:
        """Calculate all metrics"""
        
        metrics = {}
        
        # Basic metrics
        metrics["num_queries"] = len(results)
        metrics["retrieval_efficiency"] = self.calculate_retrieval_efficiency(results)
        metrics["transfer_accuracy"] = self.calculate_transfer_accuracy(results, memory)
        metrics["task_success_rate"] = self.calculate_success_rate(results)
        
        # Performance metrics
        metrics["avg_execution_time"] = np.mean([r.execution_time for r in results])
        metrics["total_memory_writes"] = sum(r.memory_writes for r in results)
        
        # TCM-specific metrics
        if hasattr(memory, 'get_expertise_distribution'):
            metrics["trust_convergence"] = self.calculate_trust_convergence(memory)
            metrics["expertise_distribution"] = memory.get_expertise_distribution()
            
        # Memory backend metrics
        memory_metrics = memory.get_metrics()
        metrics["memory_stats"] = memory_metrics
        
        return metrics
    
    def calculate_retrieval_efficiency(self, results: List[QueryResult]) -> float:
        """Calculate average retrieval efficiency"""
        
        if not results:
            return 0.0
            
        scores = [r.retrieval_score for r in results]
        return np.mean(scores)
    
    def calculate_transfer_accuracy(self, results: List[QueryResult], memory: BaseMemory) -> float:
        """Calculate transfer accuracy (delegation rate for TCM)"""
        
        memory_metrics = memory.get_metrics()
        
        # For TCM, use delegation rate
        if "delegation_rate" in memory_metrics:
            return memory_metrics["delegation_rate"]
            
        # For Selective, use delegation rate
        if "delegation_rate" in memory_metrics:
            return memory_metrics["delegation_rate"]
            
        # For Shared, always 1.0 (all transfers to shared pool)
        if memory.__class__.__name__ == "SharedMemory":
            return 1.0
            
        # For Isolated, always 0.0 (no transfers)
        return 0.0
    
    def calculate_success_rate(self, results: List[QueryResult]) -> float:
        """Calculate task success rate"""
        
        if not results:
            return 0.0
            
        successes = sum(r.success for r in results)
        return successes / len(results)
    
    def calculate_trust_convergence(self, memory: BaseMemory) -> Dict[str, Any]:
        """Calculate trust convergence metrics for TCM"""
        
        if not hasattr(memory, 'trust_params'):
            return {}
            
        convergence_metrics = {
            "mean_confidence": 0.0,
            "converged_pairs": 0,
            "total_pairs": 0,
            "entropy": 0.0
        }
        
        confidences = []
        converged = 0
        
        for key, params in memory.trust_params.items():
            alpha = params["alpha"]
            beta = params["beta"]
            
            # Confidence is total observations
            confidence = alpha + beta - 2
            confidences.append(confidence)
            
            # Consider converged if confidence > threshold
            if confidence > 5:
                converged += 1
                
            convergence_metrics["total_pairs"] += 1
            
        if confidences:
            convergence_metrics["mean_confidence"] = np.mean(confidences)
            convergence_metrics["converged_pairs"] = converged
            
            # Calculate entropy of expertise distribution
            expertise_scores = []
            for key, params in memory.trust_params.items():
                score = params["alpha"] / (params["alpha"] + params["beta"])
                expertise_scores.append(score)
                
            # Normalize and calculate entropy
            expertise_scores = np.array(expertise_scores)
            expertise_scores = expertise_scores / expertise_scores.sum()
            entropy = -np.sum(expertise_scores * np.log(expertise_scores + 1e-10))
            convergence_metrics["entropy"] = float(entropy)
            
        return convergence_metrics
