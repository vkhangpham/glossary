"""Text processing utilities for disambiguation."""

import re
from typing import Dict, List, Any, Union, Optional
from collections import Counter


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


def extract_keywords(
    text: str,
    num_keywords: int = 10,
    min_word_length: int = 3
) -> List[str]:
    """
    Extract keywords from text using frequency analysis.

    Args:
        text: Input text to analyze
        num_keywords: Number of top keywords to return
        min_word_length: Minimum length for valid keywords

    Returns:
        List of keywords sorted by frequency
    """
    if not text:
        return []

    # Clean and tokenize text
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()

    # Filter words
    words = [
        word for word in words
        if len(word) >= min_word_length and word.isalpha()
    ]

    # Common stop words to filter out
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'among', 'within', 'without',
        'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we',
        'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
        'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
        'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
        'themselves', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when',
        'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now'
    }

    # Filter stop words
    words = [word for word in words if word not in stop_words]

    # Count word frequencies
    word_counts = Counter(words)

    # Return top keywords
    return [word for word, count in word_counts.most_common(num_keywords)]