"""Core utilities for prompt optimization - minimal abstraction"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import dspy


def save_prompt(
    prompt_key: str,
    prompt_content: str,
    metadata: Dict[str, Any],
    output_dir: Path = Path("data/prompts")
) -> Path:
    """Save optimized prompt with versioning and metadata"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    version_hash = hashlib.sha256(prompt_content.encode()).hexdigest()[:8]
    prompt_data = {
        "key": prompt_key,
        "content": prompt_content,
        "version": version_hash,
        "metadata": {
            **metadata,
            "created_at": datetime.now().isoformat(),
            "optimization_method": "GEPA"
        }
    }
    
    filename = f"{prompt_key}_{version_hash}.json"
    filepath = output_dir / filename
    
    with open(filepath, "w") as f:
        json.dump(prompt_data, f, indent=2)
    latest_path = output_dir / f"{prompt_key}_latest.json"
    with open(latest_path, "w") as f:
        json.dump(prompt_data, f, indent=2)
    
    return filepath


def load_prompt(prompt_key: str, version: Optional[str] = None) -> Dict[str, Any]:
    """Load prompt by key and optionally version"""
    prompts_dir = Path("data/prompts")
    
    if version:
        filepath = prompts_dir / f"{prompt_key}_{version}.json"
    else:
        filepath = prompts_dir / f"{prompt_key}_latest.json"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Prompt not found: {filepath}")
    
    with open(filepath) as f:
        return json.load(f)


