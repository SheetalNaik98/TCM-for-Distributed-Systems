"""Experiment harness for running evaluations"""

import time
from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass, asdict

from tcm_lab.agents.base import BaseAgent
from tcm_lab.memory.base import BaseMemory
from tcm_lab.eval.tasks import BaseTask
from tcm_lab.infra.event_log import EventLogger


@dataclass
class QueryResult:
    """Result from single query execution"""
    query_id: int
    query: str
    topic: str
    task_type: str
    
    # Agent responses
    planner_response: Optional[Dict] = None
    researcher_response: Optional[Dict] = None
    verifier_response: Optional[Dict] = None
    
    # Metrics
    retrieval_score: float = 0.0
    success: bool = False
    execution_time: float = 0.0
    
    # Memory stats
    memory_writes: int = 0
    memory_reads: int = 0
    delegations: int = 0


class ExperimentHarness:
    """Harness for running experiments"""
    
    def __init__(
        self,
        agents: Dict[str, BaseAgent],
        memory: BaseMemory,
        task: BaseTask,
        logger: EventLogger,
        verbose: bool = False
    ):
        self.agents = agents
        self.memory = memory
        self.task = task
        self.logger = logger
        self.verbose = verbose
        
        # Attach memory to agents
        for agent in agents.values():
            agent.set_memory(memory)
            
    def run_single_query(self, query_index: int) -> QueryResult:
        """Run a single query through the system"""
        
        start_time = time.time()
        
        # Generate query from task
        query_data = self.task.generate_query(query_index)
        query = query_data["query"]
        topic = query_data["topic"]
        context = query_data.get("context", {})
        
        # Log query
        self.logger.log_event("query_start", {
            "query_id": query_index,
            "query": query,
            "topic": topic,
            "context": context
        })
        
        # Track initial memory state
        initial_entries = len(self.memory.entries)
        
        # Phase 1: Planning
        planner_response = None
        if context.get("requires_planning", True):
            planner_response = self.agents["planner"].process(query, context)
            self.logger.log_event("planner_response", planner_response)
            
        # Phase 2: Research  
        researcher_response = None
        if context.get("requires_research", True):
            research_context = context.copy()
            if planner_response:
                research_context["plan"] = planner_response.get("plan")
                
            researcher_response = self.agents["researcher"].process(query, research_context)
            self.logger.log_event("researcher_response", researcher_response)
            
        # Phase 3: Verification
        verifier_response = None
        if context.get("requires_verification", True):
            verify_context = context.copy()
            if researcher_response:
                verify_context["claim"] = researcher_response.get("synthesis", {}).get("content", query)
                verify_context["source_agent"] = "researcher"
                verify_context["topic"] = topic
                
            verifier_response = self.agents["verifier"].process(query, verify_context)
            self.logger.log_event("verifier_response", verifier_response)
            
        # Calculate metrics
        execution_time = time.time() - start_time
        final_entries = len(self.memory.entries)
        
        # Retrieval score (from researcher's search)
        retrieval_score = 0.0
        if researcher_response:
            # Simple heuristic: confidence * memories_used / 5
            confidence = researcher_response.get("synthesis", {}).get("confidence", 0.5)
            memories_used = researcher_response.get("memories_used", 0)
            retrieval_score = confidence * min(memories_used / 5, 1.0)
            
        # Success determination
        success = False
        if verifier_response:
            verdict = verifier_response.get("verification", {}).get("verdict")
            success = verdict == "SUPPORTED"
        elif researcher_response:
            success = researcher_response.get("synthesis", {}).get("confidence", 0) > 0.6
            
        # Count delegations (for TCM)
        delegations = 0
        if hasattr(self.memory, 'delegation_matrix'):
            delegations = sum(self.memory.delegation_matrix.values())
            
        result = QueryResult(
            query_id=query_index,
            query=query,
            topic=topic,
            task_type=query_data["type"],
            planner_response=planner_response,
            researcher_response=researcher_response,
            verifier_response=verifier_response,
            retrieval_score=retrieval_score,
            success=success,
            execution_time=execution_time,
            memory_writes=final_entries - initial_entries,
            memory_reads=0,  # Would need to track this in memory backend
            delegations=delegations
        )
        
        # Log result
        self.logger.log_event("query_complete", asdict(result))
        
        if self.verbose:
            print(f"Query {query_index}: {query[:50]}...")
            print(f"  Success: {success}, Retrieval: {retrieval_score:.2f}, Time: {execution_time:.2f}s")
            
        return result
    
    def run_experiment(self, num_queries: int = 20) -> List[QueryResult]:
        """Run full experiment"""
        
        results = []
        
        self.logger.log_event("experiment_start", {
            "task": self.task.name,
            "memory_backend": self.memory.__class__.__name__,
            "num_queries": num_queries
        })
        
        for i in range(num_queries):
            result = self.run_single_query(i)
            results.append(result)
            
        self.logger.log_event("experiment_complete", {
            "num_queries": len(results),
            "success_rate": sum(r.success for r in results) / len(results),
            "avg_time": sum(r.execution_time for r in results) / len(results)
        })
        
        return results
