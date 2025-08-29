"""
Simple LLM utilities using LiteLLM and Instructor.
This replaces the complex custom LLM system with battle-tested libraries.
"""

import os
import random
import instructor
from litellm import completion
from typing import Optional, Type, Any, Dict, List, Tuple
from pydantic import BaseModel
from .logger import setup_logger

logger = setup_logger("llm_simple")

# Setup instructor client - this handles all the complexity
client = instructor.from_litellm(completion)

def get_llm_client():
    """Get the instructor client configured with LiteLLM"""
    return client

def structured_completion(
    model: str,
    messages: List[Dict[str, str]], 
    response_model: Type[BaseModel],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    max_retries: int = 3
) -> BaseModel:
    """
    Get structured completion using instructor + litellm
    
    Args:
        model: Model name in format "provider/model" (e.g., "openai/gpt-4", "anthropic/claude-3-sonnet-20240229")
        messages: List of message dicts with 'role' and 'content'
        response_model: Pydantic model class for structured output
        temperature: Model temperature
        max_tokens: Max tokens to generate
        max_retries: Number of retries on failure
        
    Returns:
        Parsed response as the specified Pydantic model
    """
    return client.chat.completions.create(
        model=model,
        messages=messages,
        response_model=response_model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries
    )

def text_completion(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> str:
    """
    Get simple text completion using litellm
    
    Args:
        model: Model name in format "provider/model" 
        messages: List of message dicts with 'role' and 'content'
        temperature: Model temperature
        max_tokens: Max tokens to generate
        
    Returns:
        Response text content
    """
    response = completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

# Convenience functions for common providers
def openai_structured(
    messages: List[Dict[str, str]],
    response_model: Type[BaseModel], 
    model: str = "gpt-4",
    **kwargs
) -> BaseModel:
    """OpenAI structured completion"""
    return structured_completion(f"openai/{model}", messages, response_model, **kwargs)

def openai_text(
    messages: List[Dict[str, str]],
    model: str = "gpt-4", 
    **kwargs
) -> str:
    """OpenAI text completion"""
    return text_completion(f"openai/{model}", messages, **kwargs)

def anthropic_structured(
    messages: List[Dict[str, str]],
    response_model: Type[BaseModel],
    model: str = "claude-3-sonnet-20240229",
    **kwargs
) -> BaseModel:
    """Anthropic structured completion"""
    return structured_completion(f"anthropic/{model}", messages, response_model, **kwargs)

def anthropic_text(
    messages: List[Dict[str, str]], 
    model: str = "claude-3-sonnet-20240229",
    **kwargs
) -> str:
    """Anthropic text completion"""
    return text_completion(f"anthropic/{model}", messages, **kwargs)

# Migration helpers - these match the old API patterns
class LLMResult:
    """Simple result wrapper to match old API"""
    def __init__(self, content: Any):
        self.text = content

def infer_structured(
    provider: str,
    prompt: str,
    response_model: Type[BaseModel],
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7
) -> LLMResult:
    """
    Compatibility function that matches old LLM.infer() API for structured outputs
    """
    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    # Default models
    if not model:
        if provider == "openai":
            model = "gpt-4"
        elif provider == "anthropic": 
            model = "claude-3-sonnet-20240229"
        else:
            model = "gpt-4"  # fallback
    
    # Make call
    result = structured_completion(
        model=f"{provider}/{model}",
        messages=messages,
        response_model=response_model,
        temperature=temperature
    )
    
    return LLMResult(result)

def infer_text(
    provider: str,
    prompt: str, 
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7
) -> LLMResult:
    """
    Compatibility function that matches old LLM.infer() API for text outputs
    """
    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    # Default models
    if not model:
        if provider == "openai":
            model = "gpt-4"
        elif provider == "anthropic":
            model = "claude-3-sonnet-20240229" 
        else:
            model = "gpt-4"  # fallback
    
    # Make call
    result = text_completion(
        model=f"{provider}/{model}",
        messages=messages,
        temperature=temperature
    )
    
    return LLMResult(result)

# Utility functions for migration
def get_random_llm_config(level: int = 0) -> Tuple[str, str]:
    """
    Get a random LLM provider and model configuration
    
    Args:
        level: Processing level (0-3) - affects model selection strategy
        
    Returns:
        Tuple of (provider, model) strings
    """
    # Provider selection - prefer OpenAI for reliability during migration
    providers = ["openai", "anthropic"]
    weights = [0.7, 0.3]  # Favor OpenAI initially
    provider = random.choices(providers, weights=weights)[0]
    
    # Model selection based on level complexity
    if level == 0:  # College level - simpler
        models = ["gpt-4o-mini", "gpt-4"]
        model = random.choices(models, weights=[0.7, 0.3])[0]
    elif level == 1:  # Department level - medium complexity
        models = ["gpt-4", "gpt-4o-mini"]
        model = random.choices(models, weights=[0.6, 0.4])[0]
    elif level == 2:  # Research areas - higher complexity
        models = ["gpt-4", "gpt-4o"]
        model = random.choices(models, weights=[0.7, 0.3])[0]
    elif level == 3:  # Conference topics - highest complexity
        models = ["gpt-4o-mini", "gpt-4o"]  # Use faster models for conference processing
        model = random.choices(models, weights=[0.8, 0.2])[0]
    else:
        model = "gpt-4"  # Default fallback
    
    return provider, model