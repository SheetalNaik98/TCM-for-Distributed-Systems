"""Base agent class with common functionality"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import uuid

from tcm_lab.llm.provider import LLMProvider
from tcm_lab.memory.base import BaseMemory


@dataclass
class AgentMessage:
    """Message structure for agent communication"""
    agent_id: str
    content: str
    metadata: Dict[str, Any]
    timestamp: float


class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, agent_id: str, role: str = None):
        self.agent_id = agent_id
        self.role = role or self.__class__.__name__.replace("Agent", "").lower()
        self.llm = LLMProvider()
        self.memory: Optional[BaseMemory] = None
        self.message_history: List[AgentMessage] = []
        
    def set_memory(self, memory: BaseMemory):
        """Attach memory backend to agent"""
        self.memory = memory
        
    @abstractmethod
    def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a query and return results"""
        pass
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate LLM response with agent-specific context"""
        system_prompt = f"You are a {self.role} agent in a multi-agent system."
        return self.llm.generate(prompt, system_prompt=system_prompt, **kwargs)
    
    def store_memory(self, content: str, topic: str, metadata: Dict[str, Any] = None):
        """Store information in memory"""
        if self.memory:
            metadata = metadata or {}
            metadata["agent_id"] = self.agent_id
            metadata["role"] = self.role
            return self.memory.write(content, topic, self.agent_id, metadata)
        
    def retrieve_memory(self, query: str, topic: str = None, k: int = 5) -> List[Dict]:
        """Retrieve information from memory"""
        if self.memory:
            return self.memory.search(query, self.agent_id, topic, k)
        return []
    
    def log_message(self, content: str, metadata: Dict[str, Any] = None):
        """Log agent message"""
        import time
        message = AgentMessage(
            agent_id=self.agent_id,
            content=content,
            metadata=metadata or {},
            timestamp=time.time()
        )
        self.message_history.append(message)
