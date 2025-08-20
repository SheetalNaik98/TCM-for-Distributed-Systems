"""Researcher agent - responsible for information gathering and synthesis"""

import json
from typing import Any, Dict, List

from .base import BaseAgent


class ResearcherAgent(BaseAgent):
    """Agent specialized in research and information synthesis"""
    
    def __init__(self, agent_id: str = "researcher"):
        super().__init__(agent_id, role="researcher")
        self.research_topics = ["nlp", "cv", "robotics", "ml", "systems", "theory"]
        
    def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process research request"""
        context = context or {}
        
        # Determine research topic
        topic = self.identify_topic(query)
        
        # Retrieve relevant research
        memories = self.retrieve_memory(query, topic=topic, k=5)
        
        # Synthesize research
        synthesis = self.synthesize_information(query, memories, context)
        
        # Store research findings
        self.store_memory(
            synthesis["content"],
            topic=topic,
            metadata={
                "type": "research",
                "query": query,
                "sources": synthesis.get("sources", [])
            }
        )
        
        return {
            "agent": self.agent_id,
            "topic": topic,
            "synthesis": synthesis,
            "memories_used": len(memories),
            "confidence": synthesis.get("confidence", 0.7)
        }
    
    def identify_topic(self, query: str) -> str:
        """Identify the primary research topic"""
        
        prompt = f"""
        Identify the primary topic for this research query:
        Query: {query}
        
        Choose from: {', '.join(self.research_topics)}
        
        Return only the topic name.
        """
        
        response = self.generate_response(prompt, temperature=0.3, max_tokens=50)
        
        # Extract topic from response
        for topic in self.research_topics:
            if topic.lower() in response.lower():
                return topic
        
        # Default to ML if no match
        return "ml"
    
    def synthesize_information(self, query: str, memories: List[Dict], context: Dict) -> Dict:
        """Synthesize information from memories and context"""
        
        memory_context = "\n".join([
            f"- {m.get('content', '')}" for m in memories[:5]
        ])
        
        prompt = f"""
        Synthesize research findings for the following query:
        Query: {query}
        
        Existing knowledge:
        {memory_context if memory_context else "No prior research found"}
        
        Additional context: {json.dumps(context)}
        
        Provide:
        1. Key findings
        2. Relevant facts
        3. Connections between concepts
        4. Confidence level (0-1)
        
        Format as structured text.
        """
        
        response = self.generate_response(prompt, temperature=0.6)
        
        # Parse and structure the synthesis
        synthesis = {
            "content": response,
            "sources": [m.get("id", "") for m in memories if m.get("id")],
            "confidence": self.estimate_confidence(response, memories)
        }
        
        return synthesis
    
    def estimate_confidence(self, synthesis: str, memories: List[Dict]) -> float:
        """Estimate confidence in research synthesis"""
        # Based on synthesis length and memory support
        base_confidence = 0.5
        
        if len(synthesis) > 100:
            base_confidence += 0.2
        
        if len(memories) > 0:
            base_confidence += min(len(memories) * 0.1, 0.3)
        
        return min(base_confidence, 1.0)
