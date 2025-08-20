"""Verifier agent - responsible for fact-checking and validation"""

import json
from typing import Any, Dict, List, Optional

from .base import BaseAgent


class VerifierAgent(BaseAgent):
    """Agent specialized in verification and fact-checking"""
    
    def __init__(self, agent_id: str = "verifier"):
        super().__init__(agent_id, role="verifier")
        self.verification_threshold = 0.7
        
    def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process verification request"""
        context = context or {}
        claim = context.get("claim", query)
        
        # Retrieve evidence
        evidence = self.retrieve_memory(claim, topic="verification", k=5)
        
        # Verify the claim
        verification = self.verify_claim(claim, evidence, context)
        
        # Store verification result
        self.store_memory(
            json.dumps(verification),
            topic="verification",
            metadata={
                "type": "verification",
                "claim": claim,
                "verdict": verification["verdict"]
            }
        )
        
        # Update trust if TCM backend
        self.update_trust(verification, context)
        
        return {
            "agent": self.agent_id,
            "verification": verification,
            "evidence_count": len(evidence),
            "trust_updated": verification.get("trust_updated", False)
        }
    
    def verify_claim(self, claim: str, evidence: List[Dict], context: Dict) -> Dict:
        """Verify a claim against evidence"""
        
        evidence_text = "\n".join([
            f"- {e.get('content', '')}" for e in evidence[:5]
        ])
        
        prompt = f"""
        Verify the following claim based on available evidence:
        Claim: {claim}
        
        Evidence:
        {evidence_text if evidence_text else "No direct evidence found"}
        
        Context: {json.dumps(context)}
        
        Provide:
        1. Verdict: SUPPORTED, REFUTED, or UNCERTAIN
        2. Confidence: 0-1
        3. Reasoning: Brief explanation
        4. Key evidence: Most relevant evidence pieces
        
        Return as JSON.
        """
        
        response = self.generate_response(prompt, temperature=0.3)
        
        try:
            verification = json.loads(response)
        except:
            # Fallback structure
            verification = {
                "verdict": "UNCERTAIN",
                "confidence": 0.5,
                "reasoning": "Unable to verify claim with available evidence",
                "key_evidence": []
            }
        
        # Ensure verdict is valid
        if verification.get("verdict") not in ["SUPPORTED", "REFUTED", "UNCERTAIN"]:
            verification["verdict"] = "UNCERTAIN"
        
        # Ensure confidence is float between 0 and 1
        try:
            verification["confidence"] = float(verification.get("confidence", 0.5))
            verification["confidence"] = max(0, min(1, verification["confidence"]))
        except:
            verification["confidence"] = 0.5
        
        return verification
    
    def update_trust(self, verification: Dict, context: Dict):
        """Update trust scores in TCM backend based on verification"""
        
        if not self.memory or not hasattr(self.memory, 'update_trust'):
            return
        
        source_agent = context.get("source_agent")
        topic = context.get("topic", "general")
        
        if not source_agent:
            return
        
        # Update trust based on verification outcome
        verdict = verification.get("verdict")
        confidence = verification.get("confidence", 0.5)
        
        if verdict == "SUPPORTED" and confidence > self.verification_threshold:
            # Success - increase trust
            self.memory.update_trust(source_agent, topic, success=True)
            verification["trust_updated"] = True
            
        elif verdict == "REFUTED" and confidence > self.verification_threshold:
            # Failure - decrease trust
            self.memory.update_trust(source_agent, topic, success=False)
            verification["trust_updated"] = True
            
        # For UNCERTAIN, don't update trust
