"""Simple utility module for saving optimized prompts without complex versioning."""

import json
import os
from datetime import datetime
from pathlib import Path


def save_simple_prompt(prompt_key: str, prompt_content: str, output_dir: str = "data/prompts") -> str:
    """
    Save an optimized prompt in simple JSON format compatible with load_prompt_from_file().
    
    Args:
        prompt_key: Key for the prompt (e.g., "lv0_s1_system")
        prompt_content: The actual prompt text content
        output_dir: Directory to save prompts (default: "data/prompts")
    
    Returns:
        str: Path to the saved file
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create filename with _latest suffix
    filename = f"{prompt_key}_latest.json"
    filepath = output_path / filename
    
    # Create simple JSON structure compatible with load_prompt_from_file()
    prompt_data = {
        "content": prompt_content,
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "optimization_method": "GEPA",
            "prompt_key": prompt_key
        }
    }
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(prompt_data, f, indent=2, ensure_ascii=False)
    
    return str(filepath)