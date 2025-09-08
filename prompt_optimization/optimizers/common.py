"""
Common helper functions for prompt optimizers.

Environment Variables:
    GEPA_GEN_MODEL: Generation model (default: gpt-4o-mini)
    GEPA_REFLECTION_MODEL: Reflection model (default: same as GEPA_GEN_MODEL)
"""

import json
import os
from typing import List, Dict, Any, Tuple, Optional
import dspy


def load_json_training(path: str) -> List[Dict[str, Any]]:
    """
    Load training examples from a JSON file.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        List of training examples
        
    Raises:
        ValueError: If file doesn't exist or no examples found
    """
    if not os.path.exists(path):
        raise ValueError(f"Training data file not found: {path}")
        
    with open(path, 'r') as f:
        examples = json.load(f)
        
    if len(examples) == 0:
        raise ValueError(f"No training examples found at {path}")
        
    return examples


def split_train_val(examples: List[Dict[str, Any]], ratio: float = 0.8) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split examples into training and validation sets.
    
    Args:
        examples: List of examples to split
        ratio: Fraction for training set (default: 0.8)
        
    Returns:
        Tuple of (trainset, valset)
    """
    if len(examples) == 0:
        raise ValueError("Cannot split empty dataset")
        
    if len(examples) == 1:
        # With only one example, use it for both train and val
        return examples, examples
        
    train_size = max(1, int(ratio * len(examples)))
    trainset = examples[:train_size]
    valset = examples[train_size:]
    
    return trainset, valset


def configure_openai_lms(api_key: Optional[str] = None) -> Tuple[dspy.LM, dspy.LM]:
    """
    Configure OpenAI language models for generation and reflection.
    
    Args:
        api_key: OpenAI API key (if not provided, uses OPENAI_API_KEY env var)
        
    Returns:
        Tuple of (generation_lm, reflection_lm)
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY env var not set")
    
    # Read model names from environment variables with defaults
    gen_model = os.getenv("GEPA_GEN_MODEL", "gpt-4o-mini")
    ref_model = os.getenv("GEPA_REFLECTION_MODEL", gen_model)
    
    # Configure DSPy with OpenAI
    lm = dspy.OpenAI(model=gen_model, api_key=api_key, max_tokens=2000)
    reflection_lm = dspy.OpenAI(model=ref_model, api_key=api_key, max_tokens=2000)
    
    return lm, reflection_lm


def get_gepa_params(trainset_size: int) -> Dict[str, int]:
    """
    Get GEPA optimizer parameters based on dataset size.
    
    Args:
        trainset_size: Number of training examples
        
    Returns:
        Dict with max_labeled_demos and max_bootstrapped_demos
    """
    return {
        "max_labeled_demos": min(5, trainset_size),
        "max_bootstrapped_demos": min(3, trainset_size)
    }


def extract_optimized_instruction(optimized: Any, default_fallback: str) -> str:
    """
    Extract the optimized instruction from a GEPA-optimized object.
    
    Args:
        optimized: The optimized object from GEPA
        default_fallback: Default template to use if extraction fails
        
    Returns:
        The extracted instruction or default fallback
    """
    # Try to navigate the optimized object structure
    raw_instruction = getattr(getattr(optimized, 'prog', None), 'signature', None)
    raw_instruction = getattr(raw_instruction, 'instruction', None)
    
    # Check if we got a valid string instruction
    extracted = raw_instruction if isinstance(raw_instruction, str) else None
    
    # Avoid saving InputField reprs
    if not extracted or 'InputField(' in str(raw_instruction):
        return default_fallback
    
    # If we have a valid extraction, prepend it to the default template
    return f"{extracted}\n\n{default_fallback}"