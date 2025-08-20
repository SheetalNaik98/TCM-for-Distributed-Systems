"""Planner agent - responsible for task decomposition and planning"""

import json
from typing import Any, Dict, List

from .base import BaseAgent


class PlannerAgent(BaseAgent):
    """Agent specialized in planning and task decomposition"""
    
    def __init__(self, agent_id: str = "planner"):
        super().__init__(agent_id, role="planner")
        
    def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process planning request"""
        context = context or {}
        
        # Retrieve relevant planning information
        memories = self.retrieve_memory(query, topic="planning", k=3)
        
        # Generate plan
        plan = self.create_plan(query, memories, context)
        
        # Store the plan in memory
        self.store_memory(
            json.dumps(plan),
            topic="planning",
            metadata={"type": "plan", "query": query}
        )
        
        return {
            "agent": self.agent_id,
            "plan": plan,
            "memories_used": len(memories),
            "confidence": self.estimate_confidence(plan)
        }
    
    def create_plan(self, query: str, memories: List[Dict], context: Dict) -> Dict:
        """Create a structured plan"""
        
        memory_context = "\n".join([m.get("content", "") for m in memories[:3]])
        
        prompt = f"""
        Create a structured plan for the following task:
        Task: {query}
        
        Context from memory:
        {memory_context if memory_context else "No relevant memories found"}
        
        Additional context: {json.dumps(context)}
        
        Generate a plan with the following structure:
        1. Main objective
        2. Steps (list of action items)
        3. Required resources
        4. Success criteria
        
        Return as JSON.
        """
        
        response = self.generate_response(prompt, temperature=0.7)
        
        try:
            # Parse JSON response
            plan = json.loads(response)
        except:
            # Fallback to structured dict
            plan = {
                "objective": query,
                "steps": ["Analyze requirements", "Gather information", "Execute", "Verify"],
                "resources": ["memory", "computation"],
                "success_criteria": ["Task completed", "Objectives met"]
            }
        
        return plan
    
    def estimate_confidence(self, plan: Dict) -> float:
        """Estimate confidence in the plan"""
        # Simple heuristic based on plan completeness
        score = 0.0
        if plan.get("objective"):
            score += 0.25
        if plan.get("steps") and len(plan["steps"]) > 2:
            score += 0.25
        if plan.get("resources"):
            score += 0.25
        if plan.get("success_criteria"):
            score += 0.25
        return min(score, 1.0)
