"""
Simplified prompt optimization core utilities.

This module provides save/load functionality for optimized prompts.
The complex versioning, hashing, and metadata handling have been removed in favor of
a simpler approach that just saves prompts with basic metadata.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def save_prompt(
    prompt_key: str,
    prompt_content: str,
    output_dir: str = "data/prompts",
    metadata: Optional[Dict[str, Any]] = None,
    signature_metadata: Optional[Dict[str, Any]] = None,
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
        signature_metadata: Optional DSPy signature metadata for declarative programming

    Returns:
        str: Absolute path to the saved file

    Raises:
        ValueError: If prompt_key or prompt_content is empty/invalid
        IOError: If file cannot be written
    """
    if not prompt_key or not isinstance(prompt_key, str):
        raise ValueError(f"Invalid prompt_key: {prompt_key}")

    if not prompt_content or not isinstance(prompt_content, str):
        raise ValueError(
            f"Invalid prompt_content for key '{prompt_key}': content cannot be empty"
        )

    prompt_content = prompt_content.strip()
    if not prompt_content:
        raise ValueError(
            f"Prompt content for key '{prompt_key}' is empty after stripping whitespace"
        )

    prompt_type = "system" if "system" in prompt_key.lower() else "user"

    output_path = Path(output_dir).resolve()
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {output_path}")
    except Exception as e:
        raise IOError(f"Failed to create directory {output_path}: {e}")

    filename = f"{prompt_key}_latest.json"
    filepath = output_path / filename

    # Determine version based on signature metadata presence
    version = "2.0" if signature_metadata else "1.0"
    
    prompt_data: Dict[str, Any] = {
        "content": prompt_content,
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "optimization_method": "GEPA",
            "prompt_key": prompt_key,
            "prompt_type": prompt_type,
            "content_length": len(prompt_content),
            "version": version,
        },
    }

    # Add signature metadata if provided
    if signature_metadata and isinstance(signature_metadata, dict):
        prompt_data["signature_metadata"] = signature_metadata
        logger.debug(f"Added signature metadata for prompt '{prompt_key}'")

    if metadata and isinstance(metadata, dict):
        prompt_data["metadata"].update(metadata)

    logger.info(f"Saving prompt '{prompt_key}' to: {filepath}")
    logger.debug(f"Prompt content preview (first 100 chars): {prompt_content[:100]}...")

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(prompt_data, f, indent=2, ensure_ascii=False)

        if not filepath.exists():
            raise IOError(f"File was not created at {filepath}")

        file_size = filepath.stat().st_size
        if file_size == 0:
            raise IOError(f"File at {filepath} is empty")

        logger.info(f"Successfully saved prompt to {filepath} ({file_size} bytes)")

    except Exception as e:
        logger.error(f"Failed to save prompt to {filepath}: {e}")
        raise IOError(f"Failed to save prompt to {filepath}: {e}")

    return str(filepath.resolve())


def save_prompt_batch(
    prompts: Dict[str, str],
    output_dir: str = "data/prompts",
    metadata: Optional[Dict[str, Any]] = None,
    signature_metadata: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]] = None,
) -> Dict[str, str]:
    """
    Save multiple prompts at once with flexible signature metadata handling.

    Args:
        prompts: Dictionary mapping prompt_key to prompt_content
        output_dir: Directory to save prompts
        metadata: Optional metadata to include with all prompts
        signature_metadata: Optional DSPy signature metadata. Can be:
            - Dict: Applied to all prompts
            - Dict[str, Dict]: Mapping from prompt_key to specific metadata

    Returns:
        Dictionary mapping prompt_key to saved file path. In case of partial failures,
        returns paths for successfully saved prompts only.

    Raises:
        ValueError: If prompts is empty or contains invalid data
        IOError: On complete batch write failures (when no prompts could be saved)
        
    Note:
        Partial failures (some prompts saved, some failed) return successful paths
        and log errors, but do not raise exceptions. Only complete failures raise IOError.
    """
    if not prompts:
        raise ValueError("No prompts provided to save")

    saved_paths = {}
    errors = []

    # Determine if signature_metadata is per-key mapping or global
    is_per_key_metadata = False
    if signature_metadata:
        # Check if it looks like a per-key mapping by checking if values are dicts
        # and keys match some of the prompt keys
        if all(isinstance(v, dict) for v in signature_metadata.values()):
            # Check if any keys match prompt keys
            if any(k in prompts for k in signature_metadata.keys()):
                is_per_key_metadata = True
                logger.debug("Using per-key signature metadata mapping")
            else:
                logger.debug("Signature metadata appears to be a global dict (no key matches)")
        else:
            logger.debug("Signature metadata is a global dict (values are not all dicts)")

    for prompt_key, prompt_content in prompts.items():
        try:
            # Determine metadata for this specific prompt
            prompt_signature_metadata = None
            if signature_metadata:
                if is_per_key_metadata:
                    prompt_signature_metadata = signature_metadata.get(prompt_key)
                    if prompt_signature_metadata:
                        logger.debug(f"Using specific signature metadata for '{prompt_key}'")
                    else:
                        logger.debug(f"No specific signature metadata found for '{prompt_key}'")
                else:
                    prompt_signature_metadata = signature_metadata
                    logger.debug(f"Using global signature metadata for '{prompt_key}'")

            path = save_prompt(prompt_key, prompt_content, output_dir, metadata, prompt_signature_metadata)
            saved_paths[prompt_key] = path
        except Exception as e:
            errors.append(f"{prompt_key}: {e}")
            logger.error(f"Failed to save prompt '{prompt_key}': {e}")

    if errors:
        error_msg = "Failed to save some prompts:\n" + "\n".join(errors)
        logger.error(error_msg)
        if not saved_paths:
            raise IOError(error_msg)

    return saved_paths


def load_prompt(
    prompt_key: str, version: str = "latest", output_dir: str = "data/prompts"
) -> Dict[str, Any]:
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
    if version != "latest":
        raise ValueError(f"Only 'latest' is supported; got version={version!r}")

    file_path = Path(output_dir) / f"{prompt_key}_latest.json"

    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")

    with open(file_path, "r") as f:
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


__all__ = ["save_prompt", "save_prompt_batch", "load_prompt", "get_prompt_content"]
