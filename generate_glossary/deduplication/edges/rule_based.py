"""
Rule-based deduplication for adding edges to the graph.

Includes text similarity rules and compound term rules.
"""

import logging
from typing import Dict, List, Optional
import networkx as nx
from difflib import SequenceMatcher

from ..types import Edge, RuleConfig, create_default_synonym_patterns

def create_rule_edges(terms: List[str], config: Optional[RuleConfig] = None) -> List[Edge]:
    """
    Create rule-based edges using pure functional approach.
    
    Args:
        terms: List of terms to process
        config: Optional RuleConfig object (uses defaults if None)
        
    Returns:
        List of Edge objects representing rule-based relationships
    """
    if config is None:
        config = RuleConfig()
    
    edges = []
    
    # Add text similarity edges
    edges.extend(_find_text_similarity_edges(terms, config.min_similarity))
    
    # Add compound term edges
    edges.extend(_find_compound_term_edges(terms))
    
    # Add acronym edges (if enabled)
    if config.enable_acronym_detection:
        edges.extend(_find_acronym_edges(terms))
    
    # Add synonym edges (if enabled)
    if config.enable_synonym_detection:
        edges.extend(_find_synonym_edges(terms, config.synonym_patterns))
    
    return edges


def _find_text_similarity_edges(terms: List[str], min_similarity: float) -> List[Edge]:
    """Find text similarity edges between terms."""
    edges = []
    
    for i, term1 in enumerate(terms):
        for term2 in terms[i+1:]:
            similarity = calculate_text_similarity(term1, term2)
            if similarity >= min_similarity:
                edges.append(create_text_similarity_edge(term1, term2, similarity))
    
    return edges


def _find_compound_term_edges(terms: List[str]) -> List[Edge]:
    """Find compound term relationship edges."""
    edges = []
    
    for i, term1 in enumerate(terms):
        for term2 in terms[i+1:]:
            if is_compound_relationship(term1, term2):
                # Use higher weight for compound relationships
                similarity = calculate_text_similarity(term1, term2)
                weight = max(0.9, similarity) if similarity > 0 else 0.9
                edges.append(create_compound_term_edge(term1, term2, weight))
    
    return edges


def _find_acronym_edges(terms: List[str]) -> List[Edge]:
    """Find acronym expansion edges."""
    edges = []
    seen = set()  # Track seen edges to avoid duplicates
    
    for term1 in terms:
        for term2 in terms:
            if term1 != term2 and is_acronym_expansion(term1, term2):
                # Create normalized pair to avoid duplicates
                pair = tuple(sorted((term1, term2)))
                edge_key = ("acronym", *pair)
                
                if edge_key not in seen:
                    seen.add(edge_key)
                    edges.append(create_acronym_edge(term1, term2))
    
    return edges


def _find_synonym_edges(terms: List[str], synonym_patterns: Dict[str, str]) -> List[Edge]:
    """Find synonym edges based on regex patterns."""
    import re
    
    edges = []
    seen = set()  # Track seen edges to avoid duplicates
    
    # Use provided patterns or create defaults
    if not synonym_patterns:
        synonym_patterns = create_default_synonym_patterns()
    
    for term in terms:
        # Check if term matches any synonym pattern (regex)
        for pattern, long_form in synonym_patterns.items():
            if re.search(pattern, term, flags=re.I):
                # Find other terms with the long form
                for other_term in terms:
                    if other_term == term:
                        continue
                    
                    # Check if other term contains the long form (case insensitive)
                    if long_form.lower() in other_term.lower():
                        # Create normalized pair to avoid duplicates
                        pair = tuple(sorted((term, other_term)))
                        edge_key = ("synonym", *pair)
                        
                        if edge_key not in seen:
                            seen.add(edge_key)
                            edges.append(create_synonym_edge(term, other_term))
            
            # Also check the reverse - if term contains long form, find terms with pattern
            elif long_form.lower() in term.lower():
                for other_term in terms:
                    if other_term == term:
                        continue
                    
                    # Check if other term matches the pattern (regex)
                    if re.search(pattern, other_term, flags=re.I):
                        # Create normalized pair to avoid duplicates
                        pair = tuple(sorted((term, other_term)))
                        edge_key = ("synonym", *pair)
                        
                        if edge_key not in seen:
                            seen.add(edge_key)
                            edges.append(create_synonym_edge(term, other_term))
    
    return edges

def add_rule_based_edges(
    graph: nx.Graph,
    terms: Optional[List[str]] = None,
    min_similarity: float = 0.85,
    *,
    config: Optional[RuleConfig] = None
) -> nx.Graph:
    """
    Add edges based on rule-based similarity.

    DEPRECATED: Use create_rule_edges() for pure functional approach.
    This function is kept for backward compatibility.

    Args:
        graph: NetworkX graph to add edges to
        terms: Optional list of terms to process (if None, process all)
        min_similarity: Minimum text similarity threshold (fallback for backward compatibility)
        config: Optional RuleConfig object

    Returns:
        Updated graph with rule-based edges
    """
    import warnings
    warnings.warn(
        "add_rule_based_edges is deprecated. Use create_rule_edges() for pure functional approach.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if terms is None:
        terms = list(graph.nodes())
    
    # Use config values if provided, otherwise create fallback config
    if config is None:
        config = RuleConfig(min_similarity=min_similarity)
    
    # Get edges from pure function
    edges = create_rule_edges(terms, config)
    
    # Add edges to graph
    edges_added = 0
    for edge in edges:
        # Only add if both terms exist in graph and edge doesn't exist
        if (edge.source in graph and edge.target in graph and 
            not graph.has_edge(edge.source, edge.target)):
            graph.add_edge(
                edge.source, edge.target,
                weight=edge.weight,
                edge_type=edge.edge_type,
                method=edge.method
            )
            edges_added += 1
    
    logging.info(f"Added {edges_added} rule-based edges")
    return graph


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate text similarity using SequenceMatcher.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    # Normalize texts
    text1_norm = normalize_text(text1)
    text2_norm = normalize_text(text2)
    
    # Calculate basic similarity
    similarity = SequenceMatcher(None, text1_norm, text2_norm).ratio()
    
    # Check for substring relationships
    if text1_norm in text2_norm or text2_norm in text1_norm:
        # Boost similarity for substring matches
        len_ratio = min(len(text1_norm), len(text2_norm)) / max(len(text1_norm), len(text2_norm))
        similarity = max(similarity, 0.7 + 0.3 * len_ratio)
    
    return similarity


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Convert to lowercase and strip
    text = text.lower().strip()
    
    # Remove common punctuation
    text = text.replace("-", " ").replace("_", " ")
    text = text.replace("'s", "").replace("'", "")
    
    # Normalize whitespace
    text = " ".join(text.split())
    
    return text


def is_compound_relationship(term1: str, term2: str) -> bool:
    """
    Check if two terms have a compound relationship.
    
    Examples:
        - "Machine Learning" and "Learning"
        - "Deep Learning" and "Machine Learning"
        - "Computer Science" and "Science"
    
    Args:
        term1: First term
        term2: Second term
        
    Returns:
        True if compound relationship exists
    """
    # Normalize terms
    term1_norm = normalize_text(term1)
    term2_norm = normalize_text(term2)
    
    # Split into words
    words1 = set(term1_norm.split())
    words2 = set(term2_norm.split())
    
    # Check if one is a compound of the other
    if len(words1) > 1 and len(words2) == 1:
        # term1 is compound, term2 is simple
        if words2.issubset(words1):
            return True
    elif len(words2) > 1 and len(words1) == 1:
        # term2 is compound, term1 is simple
        if words1.issubset(words2):
            return True
    elif len(words1) > 1 and len(words2) > 1:
        # Both are compound terms
        # Check for significant overlap
        intersection = words1.intersection(words2)
        if len(intersection) >= 2:
            return True
        # Check if one is subset of the other
        if words1.issubset(words2) or words2.issubset(words1):
            return True
    
    return False


def add_acronym_edges(
    graph: nx.Graph,
    terms: Optional[List[str]] = None,
    *,
    config: Optional[RuleConfig] = None
) -> nx.Graph:
    """
    Add edges between acronyms and their expansions.

    DEPRECATED: Use create_rule_edges() for pure functional approach.
    This function is kept for backward compatibility.

    Args:
        graph: NetworkX graph
        terms: Optional list of terms to process
        config: Optional RuleConfig object

    Returns:
        Updated graph with acronym edges
    """
    import warnings
    warnings.warn(
        "add_acronym_edges is deprecated. Use create_rule_edges() for pure functional approach.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if terms is None:
        terms = list(graph.nodes())

    # Check if acronym detection is enabled
    if config is not None and not config.enable_acronym_detection:
        logging.debug("Acronym detection disabled in config, skipping")
        return graph
    
    if config is None:
        config = RuleConfig()

    # Get acronym edges from pure function
    edges = _find_acronym_edges(terms)
    
    # Add edges to graph
    edges_added = 0
    for edge in edges:
        # Only add if both terms exist in graph and edge doesn't exist
        if (edge.source in graph and edge.target in graph and 
            not graph.has_edge(edge.source, edge.target)):
            graph.add_edge(
                edge.source, edge.target,
                weight=edge.weight,
                edge_type=edge.edge_type,
                method=edge.method
            )
            edges_added += 1
            logging.debug(f"Added acronym edge: {edge.source} <-> {edge.target}")
    
    if edges_added > 0:
        logging.info(f"Added {edges_added} acronym edges")
    
    return graph


def is_acronym_expansion(short_term: str, long_term: str) -> bool:
    """
    Check if short_term is an acronym of long_term.
    
    Args:
        short_term: Potential acronym
        long_term: Potential expansion
        
    Returns:
        True if acronym relationship exists
    """
    # Check if short_term could be an acronym
    if not (2 <= len(short_term) <= 10 and short_term.isupper()):
        return False
    
    # Get words from long_term
    words = long_term.split()
    if len(words) < 2:
        return False
    
    # Check if initials match
    initials = "".join(word[0].upper() for word in words if word)
    
    # Exact match
    if initials == short_term:
        return True
    
    # Check with common words removed
    significant_words = [w for w in words if w.lower() not in {"of", "and", "the", "for", "in", "on", "at"}]
    if len(significant_words) >= 2:
        significant_initials = "".join(word[0].upper() for word in significant_words)
        if significant_initials == short_term:
            return True
    
    return False


def add_synonym_edges(
    graph: nx.Graph,
    terms: Optional[List[str]] = None,
    synonym_patterns: Optional[dict] = None,
    *,
    config: Optional[RuleConfig] = None
) -> nx.Graph:
    """
    Add edges for known synonyms based on patterns.

    DEPRECATED: Use create_rule_edges() for pure functional approach.
    This function is kept for backward compatibility.

    Args:
        graph: NetworkX graph
        terms: Optional list of terms to process
        synonym_patterns: Optional dictionary of synonym patterns (fallback for backward compatibility)
        config: Optional RuleConfig object

    Returns:
        Updated graph with synonym edges
    """
    import warnings
    warnings.warn(
        "add_synonym_edges is deprecated. Use create_rule_edges() for pure functional approach.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if terms is None:
        terms = list(graph.nodes())

    # Check if synonym detection is enabled
    if config is not None and not config.enable_synonym_detection:
        logging.debug("Synonym detection disabled in config, skipping")
        return graph

    # Use config patterns if provided, otherwise use individual parameter or create config
    if config is not None and config.synonym_patterns:
        patterns = config.synonym_patterns
    elif synonym_patterns is not None:
        patterns = synonym_patterns
    else:
        patterns = {}
    
    if config is None:
        config = RuleConfig(synonym_patterns=patterns)

    # Get synonym edges from pure function
    edges = _find_synonym_edges(terms, config.synonym_patterns)
    
    # Add edges to graph
    edges_added = 0
    for edge in edges:
        # Only add if both terms exist in graph and edge doesn't exist
        if (edge.source in graph and edge.target in graph and 
            not graph.has_edge(edge.source, edge.target)):
            graph.add_edge(
                edge.source, edge.target,
                weight=edge.weight,
                edge_type=edge.edge_type,
                method=edge.method
            )
            edges_added += 1
            logging.debug(f"Added synonym edge: {edge.source} <-> {edge.target}")

    if edges_added > 0:
        logging.info(f"Added {edges_added} synonym edges")

    return graph


# Edge creation helper functions (for future functional conversion)

def create_text_similarity_edge(term1: str, term2: str, similarity: float) -> Edge:
    """Create a text similarity edge object."""
    return Edge(
        source=term1,
        target=term2,
        weight=similarity,
        edge_type="text_similarity",
        method="rule_based"
    )


def create_compound_term_edge(term1: str, term2: str, weight: float) -> Edge:
    """Create a compound term edge object."""
    return Edge(
        source=term1,
        target=term2,
        weight=weight,
        edge_type="compound_term",
        method="rule_based"
    )


def create_acronym_edge(term1: str, term2: str) -> Edge:
    """Create an acronym edge object."""
    return Edge(
        source=term1,
        target=term2,
        weight=0.95,
        edge_type="acronym",
        method="rule_based"
    )


def create_synonym_edge(term1: str, term2: str) -> Edge:
    """Create a synonym edge object."""
    return Edge(
        source=term1,
        target=term2,
        weight=0.9,
        edge_type="synonym",
        method="rule_based"
    )