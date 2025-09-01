"""
Utility functions for sense disambiguation.

Pure functions for data loading, parameter management, and calculations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import re
from collections import Counter


# Level-specific parameters
LEVEL_PARAMS = {
    0: {
        "eps": 0.6,
        "min_samples": 3,
        "description": "college or broad academic domain",
        "separation_threshold": 0.7,
        "examples": "Arts and Sciences, Engineering, Medicine, Business, Law"
    },
    1: {
        "eps": 0.5,
        "min_samples": 2,
        "description": "academic department or field",
        "separation_threshold": 0.6,
        "examples": "Computer Science, Psychology, Economics, Biology"
    },
    2: {
        "eps": 0.4,
        "min_samples": 2,
        "description": "research area or specialized topic",
        "separation_threshold": 0.5,
        "examples": "Machine Learning, Cognitive Psychology, Econometrics"
    },
    3: {
        "eps": 0.3,
        "min_samples": 2,
        "description": "conference or journal topic",
        "separation_threshold": 0.25,
        "examples": "Natural Language Processing, Computer Vision, Reinforcement Learning"
    }
}


def get_level_params(level: int) -> Dict[str, Any]:
    """
    Get parameters for a specific hierarchy level.
    
    Args:
        level: Hierarchy level (0-3)
        
    Returns:
        Dictionary of level-specific parameters
    """
    return LEVEL_PARAMS.get(level, LEVEL_PARAMS[2])  # Default to level 2


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


def extract_informative_content(
    resource: Union[str, Dict[str, Any]],
    max_length: int = 1000
) -> Optional[str]:
    """
    Extract informative text content from a resource.
    
    This function extracts the most informative portions from:
    - Beginning (definitions, introductions)
    - Middle (core concepts)
    - End (conclusions, summaries)
    
    Args:
        resource: Resource data (string or dict)
        max_length: Maximum length of extracted content
        
    Returns:
        Extracted text or None
    """
    # Handle string input
    if isinstance(resource, str):
        content = resource
    elif isinstance(resource, dict):
        # Try different fields
        content = (
            resource.get("content") or
            resource.get("text") or
            resource.get("snippet") or
            resource.get("description") or
            resource.get("abstract") or
            ""
        )
        
        # Handle list content
        if isinstance(content, list):
            content = " ".join(str(item) for item in content)
    else:
        return None
    
    # Convert to string
    content = str(content).strip()
    
    if not content:
        return None
    
    # Clean content
    content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
    content = re.sub(r'<[^>]+>', '', content)  # Remove HTML tags
    
    # Extract informative portions
    if len(content) <= max_length:
        return content
    
    # Take portions from beginning, middle, and end
    portion_size = max_length // 3
    
    beginning = content[:portion_size]
    middle_start = len(content) // 2 - portion_size // 2
    middle = content[middle_start:middle_start + portion_size]
    end = content[-portion_size:]
    
    return f"{beginning} ... {middle} ... {end}"


def calculate_confidence_score(
    num_clusters: int,
    silhouette: float,
    num_resources: int
) -> float:
    """
    Calculate confidence score for ambiguity detection.
    
    Args:
        num_clusters: Number of clusters found
        silhouette: Silhouette score (-1 to 1)
        num_resources: Number of resources analyzed
        
    Returns:
        Confidence score between 0 and 1
    """
    # Base confidence from number of clusters
    if num_clusters < 2:
        return 0.0
    elif num_clusters == 2:
        base_confidence = 0.6
    elif num_clusters == 3:
        base_confidence = 0.7
    else:
        base_confidence = 0.8
    
    # Adjust for silhouette score (good separation)
    if silhouette > 0:
        silhouette_boost = silhouette * 0.2  # Max 0.2 boost
    else:
        silhouette_boost = silhouette * 0.1  # Penalty for poor separation
    
    # Adjust for resource count (more resources = more confidence)
    if num_resources >= 10:
        resource_boost = 0.1
    elif num_resources >= 5:
        resource_boost = 0.05
    else:
        resource_boost = -0.1  # Penalty for few resources
    
    # Combine scores
    confidence = base_confidence + silhouette_boost + resource_boost
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, confidence))


def extract_keywords(
    resources: List[Dict[str, Any]],
    top_n: int = 10
) -> List[str]:
    """
    Extract keywords from resources using simple frequency analysis.
    
    Args:
        resources: List of resource dictionaries
        top_n: Number of top keywords to return
        
    Returns:
        List of keywords
    """
    # Common stopwords
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "up", "about", "into", "through", "during",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might",
        "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they"
    }
    
    # Collect all words
    words = []
    for resource in resources:
        # Extract text
        if isinstance(resource, dict):
            text = (
                resource.get("content") or
                resource.get("text") or
                resource.get("snippet") or
                ""
            )
        else:
            text = str(resource)
        
        # Tokenize and clean
        text_words = re.findall(r'\b[a-z]+\b', text.lower())
        words.extend(text_words)
    
    # Count frequencies (excluding stopwords)
    word_counts = Counter(
        word for word in words
        if word not in stopwords and len(word) > 2
    )
    
    # Return top keywords
    return [word for word, _ in word_counts.most_common(top_n)]


def calculate_separation_score(
    clusters: List[List[float]],
    embeddings: Optional[List[List[float]]] = None
) -> float:
    """
    Calculate separation score between clusters.
    
    Args:
        clusters: List of cluster centroids or cluster assignments
        embeddings: Optional embeddings for detailed calculation
        
    Returns:
        Separation score between 0 and 1
    """
    if len(clusters) < 2:
        return 0.0
    
    # Simple implementation: use cluster count ratio
    # More clusters with even distribution = higher separation
    cluster_sizes = [len(c) if isinstance(c, list) else 1 for c in clusters]
    total = sum(cluster_sizes)
    
    if total == 0:
        return 0.0
    
    # Calculate distribution evenness
    max_size = max(cluster_sizes)
    evenness = 1.0 - (max_size / total - 1.0 / len(clusters))
    
    # More clusters = higher base score
    cluster_score = min(1.0, len(clusters) / 4.0)
    
    # Combine scores
    separation = (evenness + cluster_score) / 2.0
    
    return max(0.0, min(1.0, separation))


def get_term_level(
    term: str,
    hierarchy: Dict[str, Any]
) -> Optional[int]:
    """
    Get the hierarchy level of a term.
    
    Args:
        term: Term to look up
        hierarchy: Hierarchy data
        
    Returns:
        Level (0-3) or None if not found
    """
    terms_dict = hierarchy.get("terms", {})
    
    if term in terms_dict:
        return terms_dict[term].get("level")
    
    return None


def filter_terms_by_level(
    terms: List[str],
    hierarchy: Dict[str, Any],
    level: int
) -> List[str]:
    """
    Filter terms to only include those at specified level.
    
    Args:
        terms: List of terms
        hierarchy: Hierarchy data
        level: Target level (0-3)
        
    Returns:
        Filtered list of terms
    """
    filtered = []
    terms_dict = hierarchy.get("terms", {})
    
    for term in terms:
        if term in terms_dict:
            term_level = terms_dict[term].get("level")
            if term_level == level:
                filtered.append(term)
    
    return filtered