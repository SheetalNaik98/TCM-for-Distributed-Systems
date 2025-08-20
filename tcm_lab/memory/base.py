"""Base memory interface for all memory backends"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import uuid
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MemoryEntry:
    """Single memory entry"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    topic: str = "general"
    owner: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    vector: Optional[List[float]] = None


class BaseMemory(ABC):
    """Abstract base class for memory backends"""
    
    def __init__(self):
        self.entries: List[MemoryEntry] = []
        self.metadata = {}
        
    @abstractmethod
    def write(self, content: str, topic: str, agent_id: str, metadata: Dict[str, Any] = None) -> str:
        """Write to memory and return entry ID"""
        pass
    
    @abstractmethod
    def search(self, query: str, agent_id: str, topic: Optional[str] = None, k: int = 5) -> List[Dict]:
        """Search memory and return top k results"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get memory-specific metrics"""
        pass
    
    def clear(self):
        """Clear all memory entries"""
        self.entries = []
        self.metadata = {}
