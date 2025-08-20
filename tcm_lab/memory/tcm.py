"""Transactive Cognitive Memory implementation with Beta distribution trust model"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict
import networkx as nx

from .base import BaseMemory, MemoryEntry
from .vector_store import VectorStore


class TransactiveCognitiveMemory(BaseMemory):
    """TCM with dynamic trust-based delegation"""
    
    def __init__(self, agents: List[str], topics: List[str]):
        super().__init__()
        self.agents = agents
        self.topics = topics
        
        # Vector stores for each agent
        self.agent_stores = {agent: VectorStore() for agent in agents}
        
        # Trust model: Beta distribution parameters (alpha, beta) for each (agent, topic) pair
        self.trust_params = defaultdict(lambda: {"alpha": 1.0, "beta": 1.0})
        
        # Meta-memory: knowledge graph
        self.knowledge_graph = nx.DiGraph()
        for agent in agents:
            self.knowledge_graph.add_node(agent, type="agent")
        for topic in topics:
            self.knowledge_graph.add_node(topic, type="topic")
            
        # Tracking
        self.write_counts = {agent: 0 for agent in agents}
        self.delegation_matrix = defaultdict(int)  # (from_agent, to_agent, topic) -> count
        self.pointer_refs = {}  # entry_id -> actual_owner
        
    def write(self, content: str, topic: str, agent_id: str, metadata: Dict[str, Any] = None) -> str:
        """Write with trust-based delegation"""
        
        # Determine best owner for this topic
        best_owner = self.get_best_owner(topic, agent_id)
        
        # Track delegation
        if best_owner != agent_id:
            self.delegation_matrix[(agent_id, best_owner, topic)] += 1
            
        # Create entry
        entry = MemoryEntry(
            content=content,
            topic=topic,
            owner=best_owner,
            metadata=metadata or {}
        )
        
        # Store in best owner's memory
        entry_id = self.agent_stores[best_owner].add(entry)
        self.entries.append(entry)
        
        # Store pointer in other agents
        for agent in self.agents:
            if agent != best_owner:
                pointer_entry = MemoryEntry(
                    content=f"[POINTER] See {best_owner}:{entry_id}",
                    topic=topic,
                    owner=agent,
                    metadata={"pointer": True, "target": best_owner, "target_id": entry_id}
                )
                self.agent_stores[agent].add(pointer_entry)
                
        self.pointer_refs[entry_id] = best_owner
        self.write_counts[best_owner] += 1
        
        # Update knowledge graph
        self.knowledge_graph.add_edge(best_owner, topic, weight=self.get_expertise_score(best_owner, topic))
        
        return entry_id
    
    def search(self, query: str, agent_id: str, topic: Optional[str] = None, k: int = 5) -> List[Dict]:
        """Search with fallback to expert agents"""
        
        # First search local memory
        local_results = self.agent_stores[agent_id].search(query, k=k)
        
        # If not enough results and topic specified, check expert
        if len(local_results) < k and topic:
            expert = self.get_best_owner(topic, agent_id)
            if expert != agent_id:
                expert_results = self.agent_stores[expert].search(query, k=k)
                local_results.extend(expert_results)
                
        # Filter by topic if specified
        if topic:
            local_results = [r for r in local_results if r.get("topic") == topic]
            
        # Resolve pointers
        resolved_results = []
        for result in local_results[:k]:
            if result.get("metadata", {}).get("pointer"):
                # Fetch actual content from target
                target = result["metadata"]["target"]
                target_id = result["metadata"]["target_id"]
                actual_results = self.agent_stores[target].search_by_id(target_id)
                if actual_results:
                    resolved_results.append(actual_results[0])
            else:
                resolved_results.append(result)
                
        return resolved_results[:k]
    
    def get_best_owner(self, topic: str, requesting_agent: str) -> str:
        """Determine best owner using Thompson sampling on Beta distributions"""
        
        scores = {}
        for agent in self.agents:
            # Get trust parameters
            key = f"{agent}_{topic}"
            alpha = self.trust_params[key]["alpha"]
            beta = self.trust_params[key]["beta"]
            
            # Thompson sampling: sample from Beta distribution
            score = np.random.beta(alpha, beta)
            
            # Add small bonus for requesting agent (locality preference)
            if agent == requesting_agent:
                score += 0.1
                
            scores[agent] = score
            
        # Return agent with highest sampled score
        return max(scores, key=scores.get)
    
    def get_expertise_score(self, agent: str, topic: str) -> float:
        """Get expertise score (mean of Beta distribution)"""
        key = f"{agent}_{topic}"
        alpha = self.trust_params[key]["alpha"]
        beta = self.trust_params[key]["beta"]
        return alpha / (alpha + beta)
    
    def update_trust(self, agent: str, topic: str, success: bool):
        """Update trust parameters based on verification outcome"""
        key = f"{agent}_{topic}"
        
        if success:
            # Success: increment alpha
            self.trust_params[key]["alpha"] += 1
        else:
            # Failure: increment beta
            self.trust_params[key]["beta"] += 1
            
        # Update knowledge graph weight
        new_score = self.get_expertise_score(agent, topic)
        if self.knowledge_graph.has_edge(agent, topic):
            self.knowledge_graph[agent][topic]["weight"] = new_score
        else:
            self.knowledge_graph.add_edge(agent, topic, weight=new_score)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get TCM-specific metrics"""
        
        # Calculate trust convergence
        trust_scores = {}
        for agent in self.agents:
            for topic in self.topics:
                key = f"{agent}_{topic}"
                score = self.get_expertise_score(agent, topic)
                confidence = (self.trust_params[key]["alpha"] + 
                            self.trust_params[key]["beta"] - 2)  # Total observations
                trust_scores[key] = {"score": score, "confidence": confidence}
                
        # Calculate delegation statistics
        total_delegations = sum(self.delegation_matrix.values())
        total_writes = sum(self.write_counts.values())
        
        return {
            "total_entries": len(self.entries),
            "write_counts": self.write_counts,
            "trust_scores": trust_scores,
            "delegation_matrix": dict(self.delegation_matrix),
            "total_delegations": total_delegations,
            "delegation_rate": total_delegations / total_writes if total_writes > 0 else 0,
            "graph_density": nx.density(self.knowledge_graph),
            "expertise_distribution": self.get_expertise_distribution()
        }
    
    def get_expertise_distribution(self) -> Dict[str, Dict[str, float]]:
        """Get expertise distribution across agents and topics"""
        distribution = {}
        for agent in self.agents:
            distribution[agent] = {}
            for topic in self.topics:
                distribution[agent][topic] = self.get_expertise_score(agent, topic)
        return distribution
