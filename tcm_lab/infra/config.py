"""Configuration management"""

import os
from typing import List, Optional
from pathlib import Path
import toml
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings"""
    
    # System settings
    app_name: str = "TCM Lab"
    version: str = "0.1.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # LLM settings
    llm_provider: str = Field(default="openai", env="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=2000, env="LLM_MAX_TOKENS")
    
    # API keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # Memory settings
    default_memory_backend: str = "tcm"
    vector_dim: int = 768
    similarity_threshold: float = 0.7
    use_embeddings: bool = False
    
    # TCM settings
    tcm_initial_alpha: float = 1.0
    tcm_initial_beta: float = 1.0
    tcm_exploration_weight: float = 0.2
    tcm_trust_decay: float = 0.95
    tcm_min_confidence: int = 3
    
    # Agent settings
    agent_roles: List[str] = ["planner", "researcher", "verifier"]
    agent_max_iterations: int = 10
    agent_timeout: int = 30
    
    # Topics
    topics: List[str] = [
        "planning",
        "research",
        "verification", 
        "nlp",
        "cv",
        "robotics",
        "ml",
        "systems",
        "theory"
    ]
    
    # Evaluation settings
    eval_num_queries: int = 20
    eval_metrics: List[str] = [
        "retrieval_efficiency",
        "transfer_accuracy",
        "task_success_rate",
        "trust_convergence"
    ]
    
    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = "json"
    output_dir: str = "runs"
    
    # Cache settings
    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")
    cache_ttl: int = 3600
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        
    @classmethod
    def from_toml(cls, config_file: str = "config.toml") -> "Settings":
        """Load settings from TOML file"""
        
        if Path(config_file).exists():
            config_data = toml.load(config_file)
            
            # Flatten nested config
            flat_config = {}
            for section, values in config_data.items():
                if isinstance(values, dict):
                    for key, value in values.items():
                        flat_key = f"{section}_{key}" if section != "system" else key
                        flat_config[flat_key] = value
                        
            return cls(**flat_config)
        else:
            return cls()
            
    def to_dict(self) -> dict:
        """Convert settings to dictionary"""
        return self.dict(exclude_none=True)
    
    def validate_api_keys(self) -> bool:
        """Validate that at least one API key is configured"""
        
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        elif self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("Anthropic API key not configured")
            
        return True
