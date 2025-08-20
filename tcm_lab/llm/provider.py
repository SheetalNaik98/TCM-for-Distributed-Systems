"""Unified LLM provider for OpenAI and Anthropic APIs"""

import os
import json
import hashlib
from typing import Optional, Dict, Any, List
from functools import lru_cache
import logging

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class LLMProvider:
    """Unified interface for multiple LLM providers"""
    
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.model = os.getenv("LLM_MODEL", "gpt-4")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2000"))
        self.cache_enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
        
        # Initialize providers
        self._init_openai()
        self._init_anthropic()
        
        # Cache for responses
        self.cache = {} if self.cache_enabled else None
        
    def _init_openai(self):
        """Initialize OpenAI client"""
        self.openai_client = None
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if self.openai_api_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI client initialized")
            except ImportError:
                logger.warning("OpenAI library not installed")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
                
    def _init_anthropic(self):
        """Initialize Anthropic client"""
        self.anthropic_client = None
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if self.anthropic_api_key:
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
                logger.info("Anthropic client initialized")
            except ImportError:
                logger.warning("Anthropic library not installed")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic: {e}")
                
    def _get_cache_key(self, prompt: str, **kwargs) -> str:
        """Generate cache key for prompt and parameters"""
        cache_data = {
            "prompt": prompt,
            "provider": self.provider,
            "model": kwargs.get("model", self.model),
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "system_prompt": kwargs.get("system_prompt", "")
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate response from LLM"""
        
        # Check cache first
        if self.cache_enabled:
            cache_key = self._get_cache_key(prompt, system_prompt=system_prompt, **kwargs)
            if cache_key in self.cache:
                logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
                return self.cache[cache_key]
        
        # Set defaults
        model = model or self.model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        try:
            # Try primary provider
            if self.provider == "openai":
                response = self._generate_openai(
                    prompt, system_prompt, model, temperature, max_tokens, **kwargs
                )
            elif self.provider == "anthropic":
                response = self._generate_anthropic(
                    prompt, system_prompt, model, temperature, max_tokens, **kwargs
                )
            else:
                # Fallback to mock for testing
                response = self._generate_mock(prompt, system_prompt)
                
        except Exception as e:
            logger.error(f"Primary provider failed: {e}")
            # Try fallback provider
            response = self._fallback_generate(
                prompt, system_prompt, model, temperature, max_tokens, **kwargs
            )
            
        # Cache response
        if self.cache_enabled and response:
            self.cache[cache_key] = response
            
        return response
    
    def _generate_openai(
        self, 
        prompt: str, 
        system_prompt: Optional[str],
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using OpenAI API"""
        
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
            
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Map model names if needed
        if model == "gpt-4":
            model = "gpt-4-turbo-preview"
        elif model == "gpt-3.5":
            model = "gpt-3.5-turbo"
            
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return response.choices[0].message.content
    
    def _generate_anthropic(
        self, 
        prompt: str, 
        system_prompt: Optional[str],
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using Anthropic API"""
        
        if not self.anthropic_client:
            raise ValueError("Anthropic client not initialized")
            
        # Map model names if needed
        if model in ["claude-3", "claude"]:
            model = "claude-3-opus-20240229"
        elif model == "claude-instant":
            model = "claude-instant-1.2"
            
        # Combine system prompt with user prompt for Claude
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nHuman: {prompt}\n\nAssistant:"
        else:
            full_prompt = f"Human: {prompt}\n\nAssistant:"
            
        response = self.anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": full_prompt}],
            **kwargs
        )
        
        return response.content[0].text
    
    def _fallback_generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str],
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Fallback to alternative provider"""
        
        # Try opposite provider
        if self.provider == "openai" and self.anthropic_client:
            logger.info("Falling back to Anthropic")
            return self._generate_anthropic(
                prompt, system_prompt, model, temperature, max_tokens, **kwargs
            )
        elif self.provider == "anthropic" and self.openai_client:
            logger.info("Falling back to OpenAI")
            return self._generate_openai(
                prompt, system_prompt, model, temperature, max_tokens, **kwargs
            )
        else:
            # Last resort: mock response
            logger.warning("All providers failed, using mock response")
            return self._generate_mock(prompt, system_prompt)
    
    def _generate_mock(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Generate mock response for testing"""
        
        # Detect what type of response is expected
        prompt_lower = prompt.lower()
        
        if "json" in prompt_lower or "structure" in prompt_lower:
            # Return structured response
            if "plan" in prompt_lower:
                return json.dumps({
                    "objective": "Complete the requested task",
                    "steps": ["Analyze", "Process", "Execute", "Verify"],
                    "resources": ["memory", "computation"],
                    "success_criteria": ["Task completed successfully"]
                })
            elif "verif" in prompt_lower:
                return json.dumps({
                    "verdict": "SUPPORTED",
                    "confidence": 0.8,
                    "reasoning": "Based on available evidence",
                    "key_evidence": ["Evidence 1", "Evidence 2"]
                })
            else:
                return json.dumps({"result": "Mock structured response"})
                
        elif "topic" in prompt_lower and any(t in prompt_lower for t in ["nlp", "cv", "ml"]):
            # Return a topic
            for topic in ["nlp", "cv", "robotics", "ml", "systems", "theory"]:
                if topic in prompt_lower:
                    return topic
            return "ml"
            
        else:
            # Return text response
            role = "an AI assistant"
            if system_prompt and "planner" in system_prompt:
                role = "a planning agent"
            elif system_prompt and "researcher" in system_prompt:
                role = "a research agent"
            elif system_prompt and "verifier" in system_prompt:
                role = "a verification agent"
                
            return f"As {role}, I would approach this task by analyzing the requirements and providing a comprehensive response based on the available information."
    
    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """Generate responses for multiple prompts"""
        
        responses = []
        for prompt in prompts:
            response = self.generate(prompt, system_prompt, **kwargs)
            responses.append(response)
            
        return responses
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        
        try:
            import tiktoken
            
            # Get encoder for model
            if "gpt" in self.model:
                encoding = tiktoken.encoding_for_model(self.model)
            else:
                encoding = tiktoken.get_encoding("cl100k_base")
                
            return len(encoding.encode(text))
            
        except:
            # Rough estimate: 1 token ≈ 4 characters
            return len(text) // 4
