"""Task definitions for experiments"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import random


class BaseTask(ABC):
    """Base class for experimental tasks"""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.queries = []
        self.setup()
        
    @abstractmethod
    def setup(self):
        """Setup task-specific queries and contexts"""
        pass
    
    @abstractmethod
    def generate_query(self, index: int) -> Dict[str, Any]:
        """Generate a single query with context"""
        pass
    
    def get_metrics_config(self) -> Dict[str, Any]:
        """Get task-specific metrics configuration"""
        return {
            "primary_metric": "task_success_rate",
            "secondary_metrics": ["retrieval_efficiency", "transfer_accuracy"]
        }


class ExploratorySynthesisTask(BaseTask):
    """Agents co-write research synthesis by dividing domain expertise"""
    
    def setup(self):
        """Setup exploratory synthesis queries"""
        
        self.topics = [
            "transformer architectures in NLP",
            "computer vision for autonomous vehicles",
            "reinforcement learning in robotics",
            "graph neural networks",
            "federated learning systems",
            "quantum machine learning",
            "neuromorphic computing",
            "explainable AI methods",
            "multimodal learning",
            "continual learning strategies"
        ]
        
        self.synthesis_templates = [
            "Synthesize recent advances in {topic}",
            "Compare approaches for {topic}",
            "Identify key challenges in {topic}",
            "Propose future directions for {topic}",
            "Analyze the state-of-the-art in {topic}"
        ]
        
    def generate_query(self, index: int) -> Dict[str, Any]:
        """Generate synthesis query"""
        
        topic = self.topics[index % len(self.topics)]
        template = random.choice(self.synthesis_templates)
        
        return {
            "query": template.format(topic=topic),
            "topic": self._extract_topic_category(topic),
            "type": "synthesis",
            "context": {
                "requires_research": True,
                "requires_planning": True,
                "requires_verification": True,
                "complexity": "high"
            }
        }
    
    def _extract_topic_category(self, topic: str) -> str:
        """Map topic to category"""
        if "nlp" in topic.lower() or "transformer" in topic.lower():
            return "nlp"
        elif "vision" in topic.lower() or "image" in topic.lower():
            return "cv"
        elif "robot" in topic.lower():
            return "robotics"
        elif "quantum" in topic.lower() or "neuromorphic" in topic.lower():
            return "systems"
        else:
            return "ml"


class DynamicProblemSolvingTask(BaseTask):
    """Agents adaptively solve problems with changing conditions"""
    
    def setup(self):
        """Setup dynamic problem-solving scenarios"""
        
        self.problems = [
            {
                "initial": "Design a recommendation system for e-commerce",
                "constraint": "Must handle 1M users with real-time updates",
                "change": "Scale to 10M users with multi-region deployment"
            },
            {
                "initial": "Optimize supply chain routing",
                "constraint": "Minimize cost while meeting delivery deadlines",
                "change": "Add carbon footprint minimization constraint"
            },
            {
                "initial": "Develop anomaly detection for network security",
                "constraint": "Process 1GB/s of traffic with <10ms latency",
                "change": "Adapt to new attack patterns without retraining"
            },
            {
                "initial": "Create personalized learning curriculum",
                "constraint": "Adapt to individual learning speeds",
                "change": "Support collaborative group learning"
            },
            {
                "initial": "Build automated trading strategy",
                "constraint": "Maintain risk below 5% volatility",
                "change": "Adapt to cryptocurrency markets"
            }
        ]
        
    def generate_query(self, index: int) -> Dict[str, Any]:
        """Generate dynamic problem query"""
        
        problem = self.problems[index % len(self.problems)]
        
        # Simulate progression through problem stages
        stage = index // len(self.problems) % 3
        
        if stage == 0:
            query = problem["initial"]
            context_info = {"stage": "initial", "constraints": [problem["constraint"]]}
        elif stage == 1:
            query = f"{problem['initial']} with constraint: {problem['constraint']}"
            context_info = {"stage": "constrained", "constraints": [problem["constraint"]]}
        else:
            query = f"{problem['initial']} - UPDATE: {problem['change']}"
            context_info = {
                "stage": "adapted",
                "constraints": [problem["constraint"], problem["change"]]
            }
            
        return {
            "query": query,
            "topic": "planning",
            "type": "problem_solving",
            "context": {
                "requires_planning": True,
                "requires_adaptation": stage > 0,
                "complexity": "medium" if stage == 0 else "high",
                **context_info
            }
        }


class DistributedReasoningTask(BaseTask):
    """Critical information fragmented across agents requiring collaboration"""
    
    def setup(self):
        """Setup distributed reasoning scenarios"""
        
        self.scenarios = [
            {
                "goal": "Diagnose system failure in distributed application",
                "fragments": [
                    "Frontend shows 500 errors starting at 14:32 UTC",
                    "Database CPU spike detected at 14:30 UTC",
                    "Deployment of new caching layer at 14:28 UTC",
                    "Memory leak in cache invalidation logic"
                ]
            },
            {
                "goal": "Identify security breach vector",
                "fragments": [
                    "Unusual login patterns from IP range 192.168.x.x",
                    "Firewall logs show port scanning at 03:00",
                    "Employee reported phishing email at 02:45",
                    "VPN credentials used from two locations simultaneously"
                ]
            },
            {
                "goal": "Optimize machine learning pipeline",
                "fragments": [
                    "Model accuracy dropped 5% last week",
                    "Training data distribution shift detected",
                    "Feature engineering pipeline modified Tuesday",
                    "New data source integrated with schema mismatch"
                ]
            },
            {
                "goal": "Resolve production incident",
                "fragments": [
                    "Customer complaints about slow response times",
                    "CDN cache hit ratio decreased to 40%",
                    "Origin server received 10x normal traffic",
                    "Bot detection rules were accidentally disabled"
                ]
            }
        ]
        
    def generate_query(self, index: int) -> Dict[str, Any]:
        """Generate distributed reasoning query"""
        
        scenario = self.scenarios[index % len(self.scenarios)]
        
        # Determine which fragment(s) this query focuses on
        fragment_index = index % len(scenario["fragments"])
        
        # Create query that requires piecing together information
        if index < len(self.scenarios):
            # Initial queries - provide fragments
            query = f"Information: {scenario['fragments'][fragment_index]}"
            query_type = "fragment"
        else:
            # Later queries - require reasoning
            query = scenario["goal"]
            query_type = "reasoning"
            
        return {
            "query": query,
            "topic": "verification",
            "type": "distributed_reasoning",
            "context": {
                "scenario_id": index % len(self.scenarios),
                "fragment_index": fragment_index if query_type == "fragment" else None,
                "query_type": query_type,
                "requires_verification": True,
                "requires_correlation": query_type == "reasoning",
                "complexity": "high"
            }
        }
