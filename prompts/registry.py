"""
Pure functional prompt registry for managing prompts with versioning.
No classes, following functional programming paradigm.
"""

from typing import Dict, Optional, Any, List
from functools import lru_cache
import re

from .storage import (
    load_prompt_file,
    save_prompt_file, 
    track_usage,
    list_all_prompts,
    compute_hash
)
from generate_glossary.utils.logger import setup_logger

logger = setup_logger("prompt_registry")

@lru_cache(maxsize=128)
def _load_cached_prompt(key: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Internal function to load and cache prompts.
    
    Args:
        key: Prompt key
        version: Optional version hash or "latest"
        
    Returns:
        Dict with prompt data or None
    """
    data = load_prompt_file(key)
    if not data:
        return None
        
    if version is None or version == "latest":
        target_hash = data.get("latest")
    else:
        target_hash = version if len(version) > 8 else None
        if not target_hash:
            for v in data.get("versions", []):
                if v["hash"].startswith(version):
                    target_hash = v["hash"]
                    break
    
    if not target_hash:
        logger.error(f"Version not found for {key}: {version}")
        return None
        
    for v in data.get("versions", []):
        if v["hash"] == target_hash:
            return {
                "content": v["content"],
                "version": v["hash"],
                "metadata": v.get("metadata", {})
            }
    
    return None

def get_prompt(key: str, version: Optional[str] = None, **kwargs) -> str:
    """
    Get a prompt by key with optional version and template variables.
    
    Args:
        key: Dot-notation key (e.g., "extraction.level0.system")
        version: Specific version hash or "latest" (default)
        **kwargs: Template variables to substitute
        
    Returns:
        The prompt string with variables substituted
        
    Examples:
        >>> prompt = get_prompt("extraction.level0.system")
        >>> prompt = get_prompt("validation.template", level=1, context="departments")
    """
    prompt_data = _load_cached_prompt(key, version)
    
    if not prompt_data:
        raise ValueError(f"Prompt not found: {key}")
    
    prompt_text = prompt_data["content"]
    
    for var_name, var_value in kwargs.items():
        placeholder = f"{{{var_name}}}"
        prompt_text = prompt_text.replace(placeholder, str(var_value))
    
    remaining = re.findall(r'\{(\w+)\}', prompt_text)
    if remaining:
        logger.warning(f"Unsubstituted variables in prompt {key}: {remaining}")
    
    track_usage(key, prompt_data["version"])
    
    return prompt_text

def register_prompt(key: str, content: str, metadata: Optional[Dict] = None) -> str:
    """
    Register a new prompt or version.
    
    Args:
        key: Dot-notation key for the prompt
        content: The prompt content
        metadata: Optional metadata (author, description, etc.)
        
    Returns:
        Version hash for the registered prompt
        
    Example:
        >>> version = register_prompt(
        ...     "extraction.level0.system",
        ...     "You are an expert in academic classification...",
        ...     {"author": "system", "description": "Level 0 extraction prompt"}
        ... )
    """
    version_hash = compute_hash(content)
    
    # Load existing data or create new
    existing_data = load_prompt_file(key)
    
    if existing_data:
        data = existing_data
        
        # Check if version already exists
        for version in data.get("versions", []):
            if version["hash"] == version_hash:
                logger.info(f"Prompt {key} version {version_hash[:8]} already exists")
                return version_hash
    else:
        data = {
            "key": key,
            "versions": []
        }
    
    from datetime import datetime
    new_version = {
        "hash": version_hash,
        "content": content,
        "created_at": datetime.now().isoformat(),
        "metadata": metadata or {}
    }
    
    data["versions"].append(new_version)
    data["latest"] = version_hash
    
    # Save to disk
    if save_prompt_file(key, data):
        logger.info(f"Registered new prompt {key} version {version_hash[:8]}")
        
        # Clear cache for this key
        _load_cached_prompt.cache_clear()
        
        return version_hash
    else:
        raise RuntimeError(f"Failed to save prompt {key}")

def list_prompts() -> List[str]:
    """
    List all available prompt keys.
    
    Returns:
        List of dot-notation keys
        
    Example:
        >>> prompts = list_prompts()
        >>> print(prompts)
        ['extraction.level0.system', 'validation.level1.user', ...]
    """
    return list_all_prompts()

def get_prompt_versions(key: str) -> List[Dict[str, Any]]:
    """
    Get all versions of a prompt with metadata.
    
    Args:
        key: Prompt key
        
    Returns:
        List of version info dictionaries
        
    Example:
        >>> versions = get_prompt_versions("extraction.level0.system")
        >>> for v in versions:
        ...     print(f"Version {v['version']}: created {v['created_at']}")
    """
    data = load_prompt_file(key)
    
    if not data:
        return []
    
    return [
        {
            "version": v["hash"][:8],
            "full_hash": v["hash"],
            "created_at": v["created_at"],
            "metadata": v.get("metadata", {}),
            "is_latest": v["hash"] == data.get("latest")
        }
        for v in data.get("versions", [])
    ]

def get_prompt_with_fallback(key: str, default: str, **kwargs) -> str:
    """
    Get a prompt with a fallback default value.
    
    Args:
        key: Prompt key to look up
        default: Default prompt text if key not found
        **kwargs: Template variables
        
    Returns:
        Prompt text or default
        
    Example:
        >>> prompt = get_prompt_with_fallback(
        ...     "extraction.level0.system",
        ...     "You are a helpful assistant.",
        ...     level=0
        ... )
    """
    try:
        return get_prompt(key, **kwargs)
    except ValueError:
        logger.debug(f"Using fallback for prompt {key}")
        # Apply template substitution to default as well
        for var_name, var_value in kwargs.items():
            placeholder = f"{{{var_name}}}"
            default = default.replace(placeholder, str(var_value))
        return default