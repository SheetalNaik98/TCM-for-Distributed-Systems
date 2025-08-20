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
        if context.get
