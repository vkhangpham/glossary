"""
Rule-based deduplication for adding edges to the graph.

Includes text similarity rules and compound term rules.
"""

import logging
from typing import List, Optional
import networkx as nx
from difflib import SequenceMatcher

def add_rule_based_edges(
    graph: nx.Graph,
    terms: Optional[List[str]] = None,
    min_similarity: float = 0.85
) -> nx.Graph:
    """
    Add edges based on rule-based similarity.
    
    Combines text similarity and compound term analysis.
    
    Args:
        graph: NetworkX graph to add edges to
        terms: Optional list of terms to process (if None, process all)
        min_similarity: Minimum text similarity threshold
        
    Returns:
        Updated graph with rule-based edges
    """
    if terms is None:
        terms = list(graph.nodes())
    
    edges_added = 0
    
    # Add text similarity edges
    for i, term1 in enumerate(terms):
        if term1 not in graph:
            continue
            
        for term2 in terms[i+1:]:
            if term2 not in graph:
                continue
                
            # Skip if edge already exists
            if graph.has_edge(term1, term2):
                continue
            
            # Check text similarity
            similarity = calculate_text_similarity(term1, term2)
            if similarity >= min_similarity:
                graph.add_edge(
                    term1, term2,
                    weight=similarity,
                    edge_type="text_similarity",
                    method="rule_based"
                )
                edges_added += 1
                logging.debug(f"Added text similarity edge: {term1} <-> {term2} (sim={similarity:.3f})")
            
            # Check compound term relationship
            if is_compound_relationship(term1, term2):
                # Use higher weight for compound relationships
                weight = max(0.9, similarity) if similarity > 0 else 0.9
                graph.add_edge(
                    term1, term2,
                    weight=weight,
                    edge_type="compound_term",
                    method="rule_based"
                )
                edges_added += 1
                logging.debug(f"Added compound term edge: {term1} <-> {term2}")
    
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
    terms: Optional[List[str]] = None
) -> nx.Graph:
    """
    Add edges between acronyms and their expansions.
    
    Args:
        graph: NetworkX graph
        terms: Optional list of terms to process
        
    Returns:
        Updated graph with acronym edges
    """
    if terms is None:
        terms = list(graph.nodes())
    
    edges_added = 0
    
    for term1 in terms:
        if term1 not in graph:
            continue
            
        for term2 in terms:
            if term2 == term1 or term2 not in graph:
                continue
                
            if graph.has_edge(term1, term2):
                continue
            
            if is_acronym_expansion(term1, term2):
                graph.add_edge(
                    term1, term2,
                    weight=0.95,
                    edge_type="acronym",
                    method="rule_based"
                )
                edges_added += 1
                logging.debug(f"Added acronym edge: {term1} <-> {term2}")
    
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
    synonym_patterns: Optional[dict] = None
) -> nx.Graph:
    """
    Add edges for known synonyms based on patterns.
    
    Args:
        graph: NetworkX graph
        terms: Optional list of terms to process
        synonym_patterns: Optional dictionary of synonym patterns
        
    Returns:
        Updated graph with synonym edges
    """
    if synonym_patterns is None:
        synonym_patterns = {
            "ml": "machine learning",
            "ai": "artificial intelligence",
            "cs": "computer science",
            "nlp": "natural language processing",
            "cv": "computer vision",
            "rl": "reinforcement learning",
            "dl": "deep learning",
            "nn": "neural network",
            "cnn": "convolutional neural network",
            "rnn": "recurrent neural network",
            "gan": "generative adversarial network"
        }
    
    if terms is None:
        terms = list(graph.nodes())
    
    edges_added = 0
    
    for term in terms:
        if term not in graph:
            continue
            
        term_lower = term.lower()
        
        # Check if term matches any synonym pattern
        for short_form, long_form in synonym_patterns.items():
            if short_form in term_lower or long_form in term_lower:
                # Find other terms with the alternate form
                for other_term in terms:
                    if other_term == term or other_term not in graph:
                        continue
                        
                    if graph.has_edge(term, other_term):
                        continue
                    
                    other_lower = other_term.lower()
                    
                    # Check if other term has the alternate form
                    if (short_form in term_lower and long_form in other_lower) or \
                       (long_form in term_lower and short_form in other_lower):
                        graph.add_edge(
                            term, other_term,
                            weight=0.9,
                            edge_type="synonym",
                            method="rule_based"
                        )
                        edges_added += 1
                        logging.debug(f"Added synonym edge: {term} <-> {other_term}")
    
    if edges_added > 0:
        logging.info(f"Added {edges_added} synonym edges")
    
    return graph