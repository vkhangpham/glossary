"""
Canonical term selection from deduplication graph.

This module handles the selection of canonical terms from connected components
in the deduplication graph.
"""

import logging
from typing import Dict, List, Any, Set, Tuple
import networkx as nx


def select_canonical_terms(
    graph: nx.Graph,
    prefer_higher_level: bool = False,
    use_web_quality: bool = True
) -> Dict[str, str]:
    """
    Select canonical terms from the deduplication graph.
    
    Args:
        graph: Deduplication graph with connected components
        prefer_higher_level: Whether to prefer terms from higher levels
        use_web_quality: Whether to use web presence as a quality signal
        
    Returns:
        Dictionary mapping all terms to their canonical forms
    """
    canonical_mapping = {}
    components = list(nx.connected_components(graph))
    
    logging.info(f"Selecting canonical terms from {len(components)} components")
    
    for component in components:
        if len(component) == 1:
            # Singleton component - term is its own canonical
            term = next(iter(component))
            canonical_mapping[term] = term
        else:
            # Multi-term component - need to select canonical
            canonical = select_canonical_from_component(
                graph, component, prefer_higher_level, use_web_quality
            )
            
            # Map all terms in component to the canonical
            for term in component:
                canonical_mapping[term] = canonical
    
    # Log statistics
    unique_canonicals = set(canonical_mapping.values())
    reduction_rate = 1 - (len(unique_canonicals) / len(canonical_mapping))
    logging.info(
        f"Selected {len(unique_canonicals)} canonical terms from {len(canonical_mapping)} total "
        f"(reduction: {reduction_rate:.1%})"
    )
    
    return canonical_mapping


def select_canonical_from_component(
    graph: nx.Graph,
    component: Set[str],
    prefer_higher_level: bool = False,
    use_web_quality: bool = True
) -> str:
    """
    Select the best canonical term from a connected component.
    
    Selection criteria (in order of priority):
    1. Level preference (if configured)
    2. Centrality in the graph (degree)
    3. Web presence quality
    4. Term length (shorter preferred)
    5. Alphabetical (for stability)
    
    Args:
        graph: Deduplication graph
        component: Set of terms in the component
        prefer_higher_level: Whether to prefer higher level terms
        use_web_quality: Whether to consider web presence
        
    Returns:
        Selected canonical term
    """
    if len(component) == 1:
        return next(iter(component))
    
    candidates = []
    
    for term in component:
        score = calculate_canonical_score(
            graph, term, prefer_higher_level, use_web_quality
        )
        candidates.append((term, score))
    
    # Sort by score (descending), then by term (ascending for stability)
    candidates.sort(key=lambda x: (-x[1], x[0]))
    
    canonical = candidates[0][0]
    
    # Log decision for large components
    if len(component) > 5:
        logging.debug(
            f"Selected '{canonical}' as canonical from {len(component)} terms "
            f"(score: {candidates[0][1]:.2f})"
        )
    
    return canonical


def calculate_canonical_score(
    graph: nx.Graph,
    term: str,
    prefer_higher_level: bool,
    use_web_quality: bool
) -> float:
    """
    Calculate a score for a term to determine its suitability as canonical.
    
    Args:
        graph: Deduplication graph
        term: Term to score
        prefer_higher_level: Whether to prefer higher level terms
        use_web_quality: Whether to consider web presence
        
    Returns:
        Canonical suitability score
    """
    node_data = graph.nodes[term]
    score = 0.0
    
    # Level-based scoring (most important)
    level = node_data.get("level", 0)
    if prefer_higher_level:
        # Higher levels (larger numbers) get higher scores
        score += level * 100
    else:
        # Lower levels (smaller numbers) get higher scores
        score += (10 - level) * 100
    
    # Centrality scoring (degree in the component)
    degree = graph.degree(term)
    score += degree * 10
    
    # Web presence scoring
    if use_web_quality:
        # URL count
        url_count = len(node_data.get("urls", []))
        score += min(url_count * 2, 20)  # Cap at 20 points
        
        # Web content availability
        if node_data.get("web_content_count", 0) > 0:
            score += 10
        
        # Average relevance score
        avg_relevance = node_data.get("avg_relevance", 0)
        score += avg_relevance * 10
    
    # Edge weight scoring (sum of edge weights)
    edge_weights = sum(
        data.get("weight", 0)
        for _, _, data in graph.edges(term, data=True)
    )
    score += edge_weights * 5
    
    # Length penalty (prefer shorter terms)
    score -= len(term) * 0.5
    
    # Special term patterns
    if term.isupper() and len(term) <= 5:
        # Likely an acronym - slight penalty
        score -= 5
    
    # Compound term handling
    word_count = len(term.split())
    if word_count == 1:
        # Single word - slight bonus
        score += 3
    elif word_count > 3:
        # Very long compound - penalty
        score -= (word_count - 3) * 2
    
    return score


def get_canonical_groups(canonical_mapping: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Group terms by their canonical forms.
    
    Args:
        canonical_mapping: Mapping from terms to canonical forms
        
    Returns:
        Dictionary mapping canonical terms to their variations
    """
    groups = {}
    
    for term, canonical in canonical_mapping.items():
        if canonical not in groups:
            groups[canonical] = []
        groups[canonical].append(term)
    
    # Sort variations within each group
    for canonical in groups:
        groups[canonical].sort()
    
    return groups


def apply_level_priority(
    canonical_mapping: Dict[str, str],
    graph: nx.Graph,
    strict: bool = True
) -> Dict[str, str]:
    """
    Apply level-based priority rules to canonical selection.
    
    Ensures that terms from higher priority levels (lower numbers)
    are always selected as canonical when present.
    
    Args:
        canonical_mapping: Initial canonical mapping
        graph: Deduplication graph
        strict: If True, always prefer lower level terms
        
    Returns:
        Updated canonical mapping
    """
    # Group terms by canonical
    groups = get_canonical_groups(canonical_mapping)
    updated_mapping = {}
    
    for canonical, variations in groups.items():
        if len(variations) == 1:
            # No variations, keep as is
            updated_mapping[canonical] = canonical
            continue
        
        # Find the term with the lowest level
        best_term = canonical
        best_level = graph.nodes[canonical].get("level", float('inf'))
        
        for term in variations:
            term_level = graph.nodes[term].get("level", float('inf'))
            
            if term_level < best_level:
                best_level = term_level
                best_term = term
            elif term_level == best_level and strict:
                # Same level - use other criteria
                if len(term) < len(best_term):
                    best_term = term
        
        # Update mapping for all variations
        for term in variations:
            updated_mapping[term] = best_term
    
    # Log changes
    changes = sum(
        1 for term in canonical_mapping
        if canonical_mapping[term] != updated_mapping.get(term, canonical_mapping[term])
    )
    
    if changes > 0:
        logging.info(f"Level priority adjustment changed {changes} canonical selections")
    
    return updated_mapping


def filter_canonical_by_level(
    canonical_mapping: Dict[str, str],
    graph: nx.Graph,
    target_level: int
) -> List[str]:
    """
    Get canonical terms that belong to a specific level.
    
    Args:
        canonical_mapping: Mapping from terms to canonical forms
        graph: Deduplication graph
        target_level: Level to filter for
        
    Returns:
        List of canonical terms at the target level
    """
    canonical_at_level = set()
    
    for term, canonical in canonical_mapping.items():
        # Check if the original term is at the target level
        if graph.nodes[term].get("level", -1) == target_level:
            # Only include if the canonical is also at the same or lower level
            canonical_level = graph.nodes[canonical].get("level", -1)
            if canonical_level <= target_level:
                canonical_at_level.add(canonical)
    
    return sorted(canonical_at_level)


def resolve_cross_level_duplicates(
    canonical_mapping: Dict[str, str],
    graph: nx.Graph
) -> Tuple[Dict[str, str], Dict[int, Set[str]]]:
    """
    Resolve duplicates across levels, ensuring each term appears only once.
    
    Args:
        canonical_mapping: Initial canonical mapping
        graph: Deduplication graph
        
    Returns:
        Tuple of (updated_mapping, excluded_by_level)
    """
    # Track which terms are claimed by which level
    term_owners = {}  # canonical -> owning_level
    excluded_by_level = {}  # level -> set of excluded terms
    
    # Process levels in priority order (lower levels have priority)
    levels = set()
    for term in canonical_mapping:
        level = graph.nodes[term].get("level", -1)
        if level >= 0:
            levels.add(level)
    
    for level in sorted(levels):
        excluded_by_level[level] = set()
        
        # Get terms at this level
        level_terms = [
            term for term in canonical_mapping
            if graph.nodes[term].get("level", -1) == level
        ]
        
        for term in level_terms:
            canonical = canonical_mapping[term]
            
            # Check if this canonical is already claimed
            if canonical in term_owners:
                owner_level = term_owners[canonical]
                if owner_level < level:
                    # Higher priority level owns it
                    excluded_by_level[level].add(term)
                    logging.debug(
                        f"Excluding '{term}' from level {level} "
                        f"(owned by level {owner_level})"
                    )
            else:
                # Claim this canonical for this level
                term_owners[canonical] = level
    
    return canonical_mapping, excluded_by_level


def format_canonical_result(
    canonical_mapping: Dict[str, str],
    graph: nx.Graph,
    include_stats: bool = True
) -> Dict[str, Any]:
    """
    Format the canonical selection result for output.
    
    Args:
        canonical_mapping: Mapping from terms to canonical forms
        graph: Deduplication graph
        include_stats: Whether to include statistics
        
    Returns:
        Formatted result dictionary
    """
    # Get canonical groups
    groups = get_canonical_groups(canonical_mapping)
    canonical_terms = sorted(groups.keys())
    
    # Build variations dictionary
    variations = {}
    for canonical, terms in groups.items():
        # Exclude the canonical itself from variations
        vars = [t for t in terms if t != canonical]
        if vars:
            variations[canonical] = vars
    
    result = {
        "canonical_terms": canonical_terms,
        "canonical_mapping": canonical_mapping,
        "variations": variations,
        "num_canonical": len(canonical_terms),
        "num_total": len(canonical_mapping)
    }
    
    if include_stats:
        stats = {
            "reduction_rate": 1 - (len(canonical_terms) / len(canonical_mapping)),
            "avg_variations": sum(len(v) for v in variations.values()) / len(canonical_terms) if canonical_terms else 0,
            "max_variations": max(len(v) for v in variations.values()) if variations else 0,
            "singleton_count": sum(1 for g in groups.values() if len(g) == 1)
        }
        
        # Add component size distribution
        component_sizes = [len(c) for c in nx.connected_components(graph)]
        stats["component_sizes"] = {
            "min": min(component_sizes) if component_sizes else 0,
            "max": max(component_sizes) if component_sizes else 0,
            "mean": sum(component_sizes) / len(component_sizes) if component_sizes else 0,
            "count": len(component_sizes)
        }
        
        result["stats"] = stats
    
    return result