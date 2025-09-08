"""Utility module for saving optimized prompts."""

import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Set up logging
logger = logging.getLogger(__name__)


def save_prompt(
    prompt_key: str, 
    prompt_content: str, 
    output_dir: str = "data/prompts",
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save an optimized prompt in simple JSON format compatible with load_prompt_from_file().
    
    This function ensures exact compatibility with the generation scripts by:
    1. Using the exact file naming convention expected (_latest.json suffix)
    2. Creating the exact directory structure expected (data/prompts/)
    3. Using the JSON format that load_prompt_from_file() expects
    4. Validating inputs to prevent empty or malformed prompts
    
    Args:
        prompt_key: Key for the prompt (e.g., "lv0_s1_system", "lv0_s3_user")
        prompt_content: The actual prompt text content
        output_dir: Directory to save prompts (default: "data/prompts")
        metadata: Optional additional metadata to include
    
    Returns:
        str: Absolute path to the saved file
        
    Raises:
        ValueError: If prompt_key or prompt_content is empty/invalid
        IOError: If file cannot be written
    """
    # Validate inputs
    if not prompt_key or not isinstance(prompt_key, str):
        raise ValueError(f"Invalid prompt_key: {prompt_key}")
    
    if not prompt_content or not isinstance(prompt_content, str):
        raise ValueError(f"Invalid prompt_content for key '{prompt_key}': content cannot be empty")
    
    prompt_content = prompt_content.strip()
    if not prompt_content:
        raise ValueError(f"Prompt content for key '{prompt_key}' is empty after stripping whitespace")
    
    # Determine prompt type from key
    prompt_type = "system" if "system" in prompt_key.lower() else "user"
    
    # Ensure output directory exists with full path
    output_path = Path(output_dir).resolve()
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {output_path}")
    except Exception as e:
        raise IOError(f"Failed to create directory {output_path}: {e}")
    
    # Create filename with _latest suffix (exact format expected by generation scripts)
    filename = f"{prompt_key}_latest.json"
    filepath = output_path / filename
    
    # Create simple JSON structure compatible with load_prompt_from_file()
    prompt_data = {
        "content": prompt_content,
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "optimization_method": "GEPA",
            "prompt_key": prompt_key,
            "prompt_type": prompt_type,
            "content_length": len(prompt_content),
            "version": "1.0"
        }
    }
    
    # Add any additional metadata provided
    if metadata:
        prompt_data["metadata"].update(metadata)
    
    # Log what we're about to save
    logger.info(f"Saving prompt '{prompt_key}' to: {filepath}")
    logger.debug(f"Prompt content preview (first 100 chars): {prompt_content[:100]}...")
    
    # Save to file with proper error handling
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(prompt_data, f, indent=2, ensure_ascii=False)
        
        # Verify the file was written correctly
        if not filepath.exists():
            raise IOError(f"File was not created at {filepath}")
            
        file_size = filepath.stat().st_size
        if file_size == 0:
            raise IOError(f"File at {filepath} is empty")
            
        logger.info(f"Successfully saved prompt to {filepath} ({file_size} bytes)")
        
    except Exception as e:
        logger.error(f"Failed to save prompt to {filepath}: {e}")
        raise IOError(f"Failed to save prompt to {filepath}: {e}")
    
    # Return absolute path for clarity
    return str(filepath.resolve())


def save_prompt_batch(
    prompts: Dict[str, str],
    output_dir: str = "data/prompts",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Save multiple prompts at once.
    
    Args:
        prompts: Dictionary mapping prompt_key to prompt_content
        output_dir: Directory to save prompts
        metadata: Optional metadata to include with all prompts
        
    Returns:
        Dictionary mapping prompt_key to saved file path
        
    Raises:
        ValueError: If prompts is empty or contains invalid data
    """
    if not prompts:
        raise ValueError("No prompts provided to save")
    
    saved_paths = {}
    errors = []
    
    for prompt_key, prompt_content in prompts.items():
        try:
            path = save_prompt(prompt_key, prompt_content, output_dir, metadata)
            saved_paths[prompt_key] = path
        except Exception as e:
            errors.append(f"{prompt_key}: {e}")
            logger.error(f"Failed to save prompt '{prompt_key}': {e}")
    
    if errors:
        error_msg = f"Failed to save some prompts:\n" + "\n".join(errors)
        logger.error(error_msg)
        if not saved_paths:
            raise ValueError(error_msg)
    
    return saved_paths