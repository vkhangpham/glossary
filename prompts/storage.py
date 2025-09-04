"""
Storage layer for prompt management.
Handles JSON file I/O with version tracking.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime

from generate_glossary.utils.logger import setup_logger

logger = setup_logger("prompt_storage")


def get_library_path() -> Path:
    """Get the path to the prompt library directory."""
    return Path(__file__).parent / "data" / "library"


def compute_hash(content: str) -> str:
    """Compute SHA256 hash of content for versioning."""
    return hashlib.sha256(content.encode()).hexdigest()


def load_prompt_file(key: str) -> Optional[Dict[str, Any]]:
    """
    Load a prompt file from disk.

    Args:
        key: Dot-notation key (e.g., "extraction.level0.system")

    Returns:
        Dict with prompt data or None if not found
    """
    parts = key.split(".")
    file_path = get_library_path()

    for part in parts[:-1]:
        file_path = file_path / part

    file_path = file_path / f"{parts[-1]}.json"

    if not file_path.exists():
        logger.debug(f"Prompt file not found: {file_path}")
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load prompt file {file_path}: {e}")
        return None


def save_prompt_file(key: str, data: Dict[str, Any]) -> bool:
    """
    Save a prompt file to disk.

    Args:
        key: Dot-notation key
        data: Prompt data to save

    Returns:
        True if successful, False otherwise
    """
    parts = key.split(".")
    file_path = get_library_path()

    for part in parts[:-1]:
        file_path = file_path / part

    file_path.mkdir(parents=True, exist_ok=True)

    file_path = file_path / f"{parts[-1]}.json"

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved prompt file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save prompt file {file_path}: {e}")
        return False


def load_metrics() -> Dict[str, Any]:
    """Load usage metrics from disk."""
    metrics_file = get_library_path() / "metrics.json"

    if not metrics_file.exists():
        return {}

    try:
        with open(metrics_file, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metrics: {e}")
        return {}


def save_metrics(metrics: Dict[str, Any]) -> bool:
    """Save usage metrics to disk."""
    metrics_file = get_library_path() / "metrics.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")
        return False


def track_usage(key: str, version: str):
    """
    Track prompt usage for metrics.

    Args:
        key: Prompt key
        version: Version hash (first 8 chars)
    """
    metrics = load_metrics()
    usage_key = f"{key}:{version[:8]}"

    if usage_key not in metrics:
        metrics[usage_key] = {"count": 0, "first_used": datetime.now().isoformat()}

    metrics[usage_key]["count"] += 1
    metrics[usage_key]["last_used"] = datetime.now().isoformat()

    save_metrics(metrics)


def list_all_prompts() -> List[str]:
    """
    List all available prompt keys.

    Returns:
        List of dot-notation keys
    """
    library_path = get_library_path()
    if not library_path.exists():
        return []

    prompts = []

    for json_file in library_path.rglob("*.json"):
        if json_file.name == "metrics.json":
            continue

        # Convert path to dot notation
        relative_path = json_file.relative_to(library_path)
        parts = list(relative_path.parts)
        parts[-1] = parts[-1].replace(".json", "")
        key = ".".join(parts)
        prompts.append(key)

    return sorted(prompts)
