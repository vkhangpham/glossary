"""
Simplified prompt optimization core utilities.

This module provides a compatibility layer for the old prompt optimization interface,
wrapping the simplified save mechanism from simple_save.py.

The complex versioning, hashing, and metadata handling have been removed in favor of
a simpler approach that just saves prompts with basic metadata.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime

from prompt_optimization.save import save_prompt as save_prompt_func


def save_prompt(
    prompt_key: str,
    prompt_content: str,
    metadata: Optional[Dict[str, Any]] = None,
    output_dir: str = "data/prompts"
) -> str:
    """
    Save a prompt using the save mechanism.
    
    Args:
        prompt_key: The key identifier for the prompt (e.g., "lv0_s1_system")
        prompt_content: The actual prompt content
        metadata: Optional metadata (ignored - save handles metadata internally)
        output_dir: Output directory
    
    Returns:
        The path to the saved prompt file
    """
    return save_prompt_func(prompt_key, prompt_content, output_dir)


def load_prompt(prompt_key: str, version: str = "latest", output_dir: str = "data/prompts") -> Dict[str, Any]:
    """
    Load a prompt from file using the simplified format.
    
    This is a compatibility function for code that still uses the old interface.
    The simplified format only has 'content' and basic metadata fields.
    
    Args:
        prompt_key: The prompt key identifier
        version: Version to load (only 'latest' is supported in simplified format)
        output_dir: Output directory where prompts are stored
    
    Returns:
        Dictionary containing the prompt data with 'content' field
    """
    # Validate version parameter
    if version != "latest":
        raise ValueError(f"Only 'latest' is supported; got version={version!r}")
    
    # In the simplified approach, we only have latest files
    # No versioning with hashes
    file_path = Path(output_dir) / f"{prompt_key}_latest.json"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        return json.load(f)


def get_prompt_content(prompt_key: str, version: str = "latest") -> str:
    """
    Get just the content of a prompt.
    
    Args:
        prompt_key: The prompt key identifier
        version: Version to load (only 'latest' is supported)
    
    Returns:
        The prompt content string
    """
    prompt_data = load_prompt(prompt_key, version)
    content = prompt_data.get("content")
    if content is None:
        raise KeyError(f"Missing 'content' in prompt file for key {prompt_key}")
    return content


__all__ = [
    'save_prompt',
    'load_prompt', 
    'get_prompt_content'
]


