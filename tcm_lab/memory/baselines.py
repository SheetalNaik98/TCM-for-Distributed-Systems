"""Baseline memory implementations"""

from typing import Any, Dict, List, Optional
import random

from .base import BaseMemory, MemoryEntry
from .vector_store import VectorStore


class IsolatedMemory(BaseMemory):
    """Each agent has completely isolated memory"""
    
    def __init__(self, agents: List[str]):
        super().__init__()
        self.agents = agents
        self.agent_stores = {agent: VectorStore() for agent in agents}
        self.write_counts = {agent: 0 for agent in agents}
        
    def write(self, content: str, topic: str, agent_id: str, metadata: Dict[str, Any] = None) -> str:
        """Write to agent's isolated memory"""
        if agent_id not in self.agent_stores:
            self.agent_stores[agent_id] = VectorStore()
            
        entry = MemoryEntry(
            content=content,
            topic=topic,
            owner=agent_id,
            metadata=metadata or {}
        )
        
        entry_id = self.agent_stores[agent_id].add(entry)
        self.entries.append(entry)
        self.write_counts[agent_id] += 1
        
        return entry_id
    
    def search(self, query: str, agent_id: str, topic: Optional[str] = None, k: int = 5) -> List[Dict]:
        """Search only in agent's own memory"""
        if agent_id not in self.agent_stores:
            return []
            
        results = self.agent_stores[agent_id].search(query, k=k)
        
        # Filter by topic if specified
        if topic:
            results = [r for r in results if r.get("topic") == topic]
            
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get isolation metrics"""
        return {
            "total_entries": len(self.entries),
            "entries_per_agent": {
                agent: len(store.entries) 
                for agent, store in self.agent_stores.items()
            },
            "write_counts": self.write_counts,
            "isolation_score": 1.0  # Perfect isolation
        }


class SharedMemory(BaseMemory):
    """Single shared memory pool for all agents"""
    
    def __init__(self):
        super().__init__()
        self.store = VectorStore()
        self.write_counts = {}
        
    def write(self, content: str, topic: str, agent_id: str, metadata: Dict[str, Any] = None) -> str:
        """Write to shared memory"""
        entry = MemoryEntry(
            content=content,
            topic=topic,
            owner=agent_id,
            metadata=metadata or {}
        )
        
        entry_id = self.store.add(entry)
        self.entries.append(entry)
        
        if agent_id not in self.write_counts:
            self.write_counts[agent_id] = 0
        self.write_counts[agent_id] += 1
        
        return entry_id
    
    def search(self, query: str, agent_id: str, topic: Optional[str] = None, k: int = 5) -> List[Dict]:
        """Search in shared memory"""
        results = self.store.search(query, k=k)
        
        # Filter by topic if specified
        if topic:
            results = [r for r in results if r.get("topic") == topic]
            
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get sharing metrics"""
        return {
            "total_entries": len(self.entries),
            "write_counts": self.write_counts,
            "unique_owners": len(set(e.owner for e in self.entries)),
            "sharing_score": 1.0  # Perfect sharing
        }


class SelectiveMemory(BaseMemory):
    """Rule-based selective memory sharing"""
    
    def __init__(self, rules: Dict[str, str] = None):
        super().__init__()
        self.store = VectorStore()
        self.rules = rules or {}  # topic -> owner mapping
        self.write_counts = {}
        self.delegated_writes = 0
        
    def write(self, content: str, topic: str, agent_id: str, metadata: Dict[str, Any] = None) -> str:
        """Write based on rules"""
        # Determine owner based on rules
        owner = self.rules.get(topic, agent_id)
        
        if owner != agent_id:
            self.delegated_writes += 1
            
        entry = MemoryEntry(
            content=content,
            topic=topic,
            owner=owner,
            metadata=metadata or {}
        )
        
        entry_id = self.store.add(entry)
        self.entries.append(entry)
        
        if owner not in self.write_counts:
            self.write_counts[owner] = 0
        self.write_counts[owner] += 1
        
        return entry_id
    
    def search(self, query: str, agent_id: str, topic: Optional[str] = None, k: int = 5) -> List[Dict]:
        """Search all memory (rules don't restrict reading)"""
        results = self.store.search(query, k=k)
        
        # Filter by topic if specified
        if topic:
            results = [r for r in results if r.get("topic") == topic]
            
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get selective sharing metrics"""
        total_writes = sum(self.write_counts.values())
        
        return {
            "total_entries": len(self.entries),
            "write_counts": self.write_counts,
            "rules": self.rules,
            "delegated_writes": self.delegated_writes,
            "delegation_rate": self.delegated_writes / total_writes if total_writes > 0 else 0
        }
