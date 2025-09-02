"""
Hierarchy-based disambiguation using parent context analysis.

Detects ambiguous terms by analyzing their relationships with parent
terms in the hierarchy to identify divergent contexts.
"""

import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict

from .utils import calculate_confidence_score


def detect(
    terms: List[str],
    web_content: Optional[Dict[str, Any]],
    hierarchy: Dict[str, Any],
    min_parent_overlap: float = 0.3,
    max_parent_similarity: float = 0.7
) -> Dict[str, Dict[str, Any]]:
    """
    Detect ambiguous terms by analyzing parent relationships.
    
    A term is considered ambiguous if:
    1. It appears under multiple parents
    2. Those parents have low contextual overlap
    3. The term's usage differs across parent contexts
    
    Args:
        terms: List of terms to analyze
        web_content: Optional web resources for enhanced analysis
        hierarchy: Hierarchy data with parent-child relationships
        min_parent_overlap: Minimum keyword overlap to consider parents related
        max_parent_similarity: Maximum similarity to consider contexts different
        
    Returns:
        Dictionary mapping ambiguous terms to their detection evidence
    """
    logging.info(f"Detecting ambiguity using hierarchy for {len(terms)} terms")
    
    results = {}
    
    # Build parent-child mappings
    parent_map = defaultdict(set)
    level_map = {}
    
    for level_idx, level_data in enumerate(hierarchy.get("levels", [])):
        for term_data in level_data.get("terms", []):
            term = term_data.get("term")
            if not term:
                continue
                
            level_map[term] = level_idx
            
            # Track parents
            parents = term_data.get("parents", [])
            if isinstance(parents, str):
                parents = [parents]
            for parent in parents:
                if parent:
                    parent_map[term].add(parent)
    
    # Analyze each term
    for term in terms:
        if term not in parent_map:
            continue
            
        parents = parent_map[term]
        if len(parents) < 2:
            continue
        
        # Analyze parent contexts
        parent_contexts = {}
        for parent in parents:
            context = extract_parent_context(parent, hierarchy, web_content)
            if context:
                parent_contexts[parent] = context
        
        if len(parent_contexts) < 2:
            continue
        
        # Check for divergent contexts
        divergence_evidence = analyze_context_divergence(
            parent_contexts,
            min_parent_overlap,
            max_parent_similarity
        )
        
        if divergence_evidence["is_divergent"]:
            # Calculate confidence
            confidence = calculate_hierarchy_confidence(
                num_parents=len(parents),
                divergence_score=divergence_evidence["divergence_score"],
                level=level_map.get(term, -1)
            )
            
            results[term] = {
                "term": term,
                "method": "hierarchy",
                "parents": list(parents),
                "level": level_map.get(term, -1),
                "divergence_evidence": divergence_evidence,
                "confidence": confidence,
                "evidence": {
                    "num_parents": len(parents),
                    "parent_contexts": parent_contexts,
                    "min_overlap": min_parent_overlap,
                    "max_similarity": max_parent_similarity
                }
            }
    
    logging.info(f"Found {len(results)} ambiguous terms via hierarchy")
    return results


def extract_parent_context(
    parent: str,
    hierarchy: Dict[str, Any],
    web_content: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Extract context information for a parent term.
    
    Args:
        parent: Parent term to analyze
        hierarchy: Hierarchy data
        web_content: Optional web resources
        
    Returns:
        Context dictionary with keywords and relationships
    """
    context = {
        "term": parent,
        "keywords": set(),
        "children": set(),
        "level": -1
    }
    
    # Find parent in hierarchy
    for level_idx, level_data in enumerate(hierarchy.get("levels", [])):
        for term_data in level_data.get("terms", []):
            if term_data.get("term") == parent:
                context["level"] = level_idx
                
                # Extract keywords from metadata
                metadata = term_data.get("metadata", {})
                if metadata:
                    # Add keywords from various metadata fields
                    for field in ["keywords", "topics", "areas", "domains"]:
                        if field in metadata:
                            values = metadata[field]
                            if isinstance(values, list):
                                context["keywords"].update(values)
                            elif isinstance(values, str):
                                context["keywords"].add(values)
                
                # Track children
                children = term_data.get("children", [])
                if isinstance(children, list):
                    context["children"].update(children)
                
                break
    
    # Enhance with web content if available
    if web_content and parent in web_content:
        resources = web_content[parent]
        if isinstance(resources, dict):
            resources = resources.get("resources", [])
        
        # Extract keywords from resource titles/descriptions
        for resource in resources[:5]:  # Sample first 5
            if isinstance(resource, dict):
                title = resource.get("title", "")
                description = resource.get("description", "")
                
                # Simple keyword extraction (could be enhanced)
                text = f"{title} {description}".lower()
                words = text.split()
                # Filter to meaningful words (length > 3, not stopwords)
                keywords = {w for w in words if len(w) > 3}
                context["keywords"].update(keywords)
    
    return context if context["keywords"] else None


def analyze_context_divergence(
    parent_contexts: Dict[str, Dict],
    min_overlap: float,
    max_similarity: float
) -> Dict[str, Any]:
    """
    Analyze divergence between parent contexts.
    
    Args:
        parent_contexts: Context data for each parent
        min_overlap: Minimum keyword overlap threshold
        max_similarity: Maximum similarity threshold
        
    Returns:
        Evidence of context divergence
    """
    parents = list(parent_contexts.keys())
    
    if len(parents) < 2:
        return {"is_divergent": False, "divergence_score": 0.0}
    
    # Calculate pairwise similarities
    similarities = []
    divergent_pairs = []
    
    for i in range(len(parents)):
        for j in range(i + 1, len(parents)):
            p1, p2 = parents[i], parents[j]
            ctx1, ctx2 = parent_contexts[p1], parent_contexts[p2]
            
            # Calculate keyword overlap
            keywords1 = ctx1["keywords"]
            keywords2 = ctx2["keywords"]
            
            if not keywords1 or not keywords2:
                continue
            
            intersection = keywords1 & keywords2
            union = keywords1 | keywords2
            
            overlap = len(intersection) / len(union) if union else 0
            similarities.append(overlap)
            
            # Check if this pair shows divergence
            if overlap < min_overlap:
                divergent_pairs.append({
                    "parent1": p1,
                    "parent2": p2,
                    "overlap": overlap,
                    "unique_to_p1": keywords1 - keywords2,
                    "unique_to_p2": keywords2 - keywords1
                })
    
    if not similarities:
        return {"is_divergent": False, "divergence_score": 0.0}
    
    # Calculate divergence score
    avg_similarity = sum(similarities) / len(similarities)
    divergence_score = 1.0 - avg_similarity
    
    # Determine if contexts are divergent
    is_divergent = (
        avg_similarity < max_similarity and
        len(divergent_pairs) > 0
    )
    
    return {
        "is_divergent": is_divergent,
        "divergence_score": divergence_score,
        "avg_similarity": avg_similarity,
        "divergent_pairs": divergent_pairs,
        "num_comparisons": len(similarities)
    }


def calculate_hierarchy_confidence(
    num_parents: int,
    divergence_score: float,
    level: int
) -> float:
    """
    Calculate confidence score for hierarchy-based detection.
    
    Args:
        num_parents: Number of parent terms
        divergence_score: Context divergence score (0-1)
        level: Hierarchy level
        
    Returns:
        Confidence score between 0 and 1
    """
    # Base confidence from divergence
    base_confidence = divergence_score
    
    # Boost for multiple parents
    parent_boost = min(0.2, (num_parents - 2) * 0.1)
    
    # Level-based adjustment (higher levels more likely to be ambiguous)
    level_factor = 1.0
    if level >= 0:
        level_factor = 1.0 + (level * 0.05)
    
    confidence = min(1.0, base_confidence * level_factor + parent_boost)
    
    return confidence