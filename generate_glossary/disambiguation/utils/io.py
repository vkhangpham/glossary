"""I/O utilities for disambiguation."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


def load_hierarchy(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load hierarchy data from JSON file.

    Args:
        path: Path to hierarchy.json

    Returns:
        Hierarchy dictionary
    """
    path = Path(path)

    if not path.exists():
        logging.error(f"Hierarchy file not found: {path}")
        return {"terms": {}}

    try:
        with open(path, 'r') as f:
            hierarchy = json.load(f)

        # Validate structure
        if "terms" not in hierarchy:
            logging.warning("Hierarchy missing 'terms' key")
            hierarchy["terms"] = {}

        return hierarchy

    except json.JSONDecodeError as e:
        logging.error(f"Error decoding hierarchy JSON: {e}")
        return {"terms": {}}
    except Exception as e:
        logging.error(f"Error loading hierarchy: {e}")
        return {"terms": {}}


def load_web_content(path: Optional[Union[str, Path]]) -> Dict[str, Any]:
    """
    Load web content from JSON file.

    Args:
        path: Path to web content JSON

    Returns:
        Web content dictionary
    """
    if not path:
        return {}

    path = Path(path)

    if not path.exists():
        logging.warning(f"Web content file not found: {path}")
        return {}

    try:
        with open(path, 'r') as f:
            content = json.load(f)

        # Handle different formats
        if isinstance(content, dict):
            return content
        elif isinstance(content, list):
            # Convert list format to dict
            result = {}
            for item in content:
                if isinstance(item, dict) and "term" in item:
                    result[item["term"]] = item
            return result
        else:
            logging.warning(f"Unexpected web content format: {type(content)}")
            return {}

    except json.JSONDecodeError as e:
        logging.error(f"Error decoding web content JSON: {e}")
        return {}
    except Exception as e:
        logging.error(f"Error loading web content: {e}")
        return {}


def save_results(
    results: Any,
    output_dir: Union[str, Path],
    prefix: str
) -> Path:
    """
    Save results to JSON file with timestamp.

    Args:
        results: Results to save
        output_dir: Output directory
        prefix: Filename prefix

    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.json"
    output_path = output_dir / filename

    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logging.info(f"Results saved to {output_path}")
        return output_path

    except Exception as e:
        logging.error(f"Error saving results: {e}")
        # Try simpler filename
        fallback_path = output_dir / f"{prefix}.json"
        with open(fallback_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        return fallback_path