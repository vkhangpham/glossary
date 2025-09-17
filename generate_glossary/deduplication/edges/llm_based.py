"""
LLM-based deduplication for adding edges to the graph.

Uses language models to analyze terms that have SOME web overlap (not enough
for web-based edges) by comparing their web content to determine if they're duplicates.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple, Literal
import networkx as nx
import json
from pydantic import BaseModel

from generate_glossary.llm import completion, structured_completion

from ..types import Edge, LLMConfig


class DedupDecision(BaseModel):
    """Pydantic model for structured deduplication decisions."""
    verdict: Literal["duplicate", "different"]
    confidence: float
    reason: str


def create_llm_edges(
    terms: List[str], 
    web_content: Dict[str, Dict[str, Any]], 
    config: Optional[LLMConfig] = None
) -> List[Edge]:
    """
    Create LLM-based edges using pure functional approach.
    
    Args:
        terms: List of terms to process
        web_content: Dictionary mapping terms to their web resources.
                    Expected structure for each term:
                    {
                        "term_name": {
                            "results": [
                                {
                                    "url": str,
                                    "title": Optional[str],
                                    "content": Optional[str],
                                    "relevance_score": Optional[float],
                                    "domain": Optional[str]
                                },
                                ...
                            ],
                            "metadata": Optional[Dict[str, Any]]
                        }
                    }
        config: Optional LLMConfig object (uses defaults if None)
        
    Returns:
        List of Edge objects representing LLM-based relationships
        
    Raises:
        ValueError: If web_content structure is invalid or missing required keys
    """
    if config is None:
        config = LLMConfig()
    
    # Validate web_content structure before processing
    for term in terms:
        if term in web_content:
            term_data = web_content[term]
            if not isinstance(term_data, dict):
                raise ValueError(f"Invalid web_content structure for term '{term}': expected dict, got {type(term_data)}")
            
            if "results" not in term_data:
                raise ValueError(f"Missing 'results' key in web_content for term '{term}'")
            
            if not isinstance(term_data["results"], list):
                raise ValueError(f"Invalid 'results' structure for term '{term}': expected list, got {type(term_data['results'])}")
    
    edges = []
    
    # Find candidate pairs with URL overlap
    candidate_pairs = _find_candidate_pairs(terms, web_content, config.min_url_overlap, config.max_url_overlap)
    
    if not candidate_pairs:
        return edges
    
    # Evaluate pairs in batches using LLM
    edge_decisions = _evaluate_pairs_with_llm(candidate_pairs, web_content, config)
    
    # Create edges from decisions that meet confidence threshold
    edges = _create_edges_from_decisions(edge_decisions, config.confidence_threshold)
    
    return edges


def _find_candidate_pairs(
    terms: List[str], 
    web_content: Dict[str, Any], 
    min_overlap: int, 
    max_overlap: int
) -> List[Tuple[str, str]]:
    """Find candidate pairs with URL overlap in specified range."""
    candidate_pairs = []

    for i, term1 in enumerate(terms):
        if term1 not in web_content:
            continue

        urls1 = set()
        for result in web_content[term1].get("results", []):
            urls1.add(result.get("url", ""))

        if not urls1:
            continue

        for term2 in terms[i + 1:]:
            if term2 not in web_content:
                continue

            urls2 = set()
            for result in web_content[term2].get("results", []):
                urls2.add(result.get("url", ""))

            if not urls2:
                continue

            overlap = len(urls1 & urls2)
            if min_overlap <= overlap < max_overlap:
                candidate_pairs.append((term1, term2))

    return candidate_pairs


def _evaluate_pairs_with_llm(
    term_pairs: List[Tuple[str, str]], 
    web_content: Dict[str, Any], 
    config: LLMConfig
) -> Dict[Tuple[str, str], Tuple[bool, float, str]]:
    """Evaluate term pairs using LLM and return decisions."""
    results = {}
    
    # Process pairs in batches
    for i in range(0, len(term_pairs), config.batch_size):
        batch = term_pairs[i:i + config.batch_size]
        
        for term1, term2 in batch:
            try:
                content1 = extract_relevant_content(term1, web_content)
                content2 = extract_relevant_content(term2, web_content)

                prompt = build_content_comparison_prompt(term1, term2, content1, content2)
                
                messages = [{"role": "user", "content": prompt}]
                
                # Determine use_case - prefer explicit use_case over level-based default
                use_case_final = config.use_case or (f"lv{config.level}_s1" if config.level is not None else "deduplication")
                
                try:
                    if config.provider == "gemini":
                        obj = structured_completion(
                            messages=messages,
                            response_model=DedupDecision,
                            model="gemini/gemini-pro",
                            temperature=config.temperature,
                            use_case=use_case_final
                        )
                    elif config.provider == "openai":
                        obj = structured_completion(
                            messages=messages,
                            response_model=DedupDecision,
                            model="openai/gpt-4",
                            temperature=config.temperature,
                            use_case=use_case_final
                        )
                    else:
                        try:
                            # Prefer per-use-case model resolution without forcing a tier
                            obj = structured_completion(
                                messages=messages,
                                response_model=DedupDecision,
                                use_case=use_case_final,
                                temperature=config.temperature,
                            )
                        except Exception:
                            logging.warning("Unknown provider '%s'. Falling back to gemini/gemini-pro.", config.provider)
                            obj = structured_completion(
                                messages=messages,
                                response_model=DedupDecision,
                                model="gemini/gemini-pro",
                                temperature=config.temperature,
                                use_case=use_case_final,
                            )
                    
                    results[(term1, term2)] = (obj.verdict == "duplicate", float(obj.confidence), obj.reason)
                except Exception as inner_e:
                    logging.error(f"Structured completion failed for {term1} vs {term2}: {inner_e}")
                    # Fallback to old parsing method if structured completion fails
                    try:
                        response = completion(
                            messages=messages,
                            model="gemini/gemini-pro",
                            temperature=config.temperature,
                            use_case=use_case_final
                        )
                        is_duplicate, confidence, reason = parse_llm_response(response)
                        results[(term1, term2)] = (is_duplicate, confidence, reason)
                    except Exception as fallback_e:
                        logging.error(f"Fallback completion also failed for {term1} vs {term2}: {fallback_e}")
                        results[(term1, term2)] = (False, 0.0, "Error in evaluation")

            except Exception as e:
                logging.error(f"Error evaluating {term1} vs {term2}: {e}")
                results[(term1, term2)] = (False, 0.0, "Error in evaluation")

    return results


def _create_edges_from_decisions(
    edge_decisions: Dict[Tuple[str, str], Tuple[bool, float, str]], 
    confidence_threshold: float
) -> List[Edge]:
    """Create edges from LLM decisions that meet confidence threshold."""
    edges = []
    
    for (term1, term2), (is_duplicate, confidence, reason) in edge_decisions.items():
        if is_duplicate and confidence >= confidence_threshold:
            edges.append(create_llm_analysis_edge(term1, term2, confidence, reason))
    
    return edges

def add_llm_based_edges(
    graph: nx.Graph,
    web_content: Dict[str, Any],
    terms: Optional[List[str]] = None,
    provider: str = "gemini",
    min_url_overlap: int = 1,
    max_url_overlap: int = 2,
    batch_size: int = 5,
    confidence_threshold: float = 0.8,
    level: Optional[int] = None,
    use_case: Optional[str] = None,
    temperature: float = 0.3,
    *,
    config: Optional[LLMConfig] = None
) -> nx.Graph:
    """
    Add edges based on LLM analysis of shared web content using DSPy framework.

    DEPRECATED: Use create_llm_edges() for pure functional approach.
    This function is kept for backward compatibility.

    This method focuses on term pairs that have SOME URL overlap (not enough
    for web-based edges) and uses LLM with optimized prompts to analyze their
    web content to determine if they're duplicates.

    Args:
        graph: NetworkX graph to add edges to
        web_content: Web content dictionary for all terms
        terms: Optional list of terms to process (if None, process all)
        provider: LLM provider to use (fallback for backward compatibility)
        min_url_overlap: Minimum URL overlap to consider (fallback for backward compatibility)
        max_url_overlap: Maximum URL overlap (fallback for backward compatibility)
        batch_size: Number of term pairs to evaluate per LLM call (fallback for backward compatibility)
        confidence_threshold: Minimum confidence for creating edge (fallback for backward compatibility)
        level: Optional level for use_case mapping (fallback for backward compatibility)
        use_case: Optional use_case override (fallback for backward compatibility)
        temperature: LLM temperature setting (fallback for backward compatibility)
        config: Optional LLMConfig object

    Returns:
        Updated graph with LLM-based edges
    """
    import warnings
    warnings.warn(
        "add_llm_based_edges is deprecated. Use create_llm_edges() for pure functional approach.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if terms is None:
        terms = list(graph.nodes())

    # Use config values if provided, otherwise create fallback config
    if config is None:
        config = LLMConfig(
            provider=provider,
            min_url_overlap=min_url_overlap,
            max_url_overlap=max_url_overlap,
            batch_size=batch_size,
            confidence_threshold=confidence_threshold,
            level=level,
            use_case=use_case,
            temperature=temperature
        )

    # Get edges from pure function
    edges = create_llm_edges(terms, web_content, config)
    
    # Add edges to graph
    edges_added = 0
    for edge in edges:
        # Only add if both terms exist in graph and edge doesn't exist
        if (edge.source in graph and edge.target in graph and 
            not graph.has_edge(edge.source, edge.target)):
            # Add edge with all metadata
            edge_data = {
                "weight": edge.weight,
                "edge_type": edge.edge_type,
                "method": edge.method
            }
            # Add any metadata from the edge
            if edge.metadata:
                edge_data.update(edge.metadata)
                # Also add reason directly for backward compatibility
                if "reason" in edge.metadata:
                    edge_data["reason"] = edge.metadata["reason"]
                
            graph.add_edge(edge.source, edge.target, **edge_data)
            edges_added += 1
            logging.debug(
                f"Added LLM edge: {edge.source} <-> {edge.target} "
                f"(confidence={edge.weight:.2f}, reason={edge.metadata.get('reason', 'N/A')})"
            )

    logging.info(f"Added {edges_added} LLM-based edges")
    return graph


def find_candidate_pairs_with_overlap(
    terms: List[str],
    web_content: Dict[str, Any],
    min_overlap: int = 1,
    max_overlap: int = 2,
) -> List[Tuple[str, str]]:
    """
    Find term pairs that have some URL overlap but not enough for web-based edges.

    Args:
        terms: List of terms to check
        web_content: Web content dictionary
        min_overlap: Minimum URL overlap required
        max_overlap: Maximum URL overlap (exclusive)

    Returns:
        List of (term1, term2) tuples with URL overlap in range
    """
    candidate_pairs = []

    for i, term1 in enumerate(terms):
        if term1 not in web_content:
            continue

        urls1 = set()
        for result in web_content[term1].get("results", []):
            urls1.add(result.get("url", ""))

        if not urls1:
            continue

        for term2 in terms[i + 1 :]:
            if term2 not in web_content:
                continue

            urls2 = set()
            for result in web_content[term2].get("results", []):
                urls2.add(result.get("url", ""))

            if not urls2:
                continue

            overlap = len(urls1 & urls2)
            if min_overlap <= overlap < max_overlap:
                candidate_pairs.append((term1, term2))

    return candidate_pairs


def evaluate_pairs_with_web_content(
    term_pairs: List[Tuple[str, str]],
    web_content: Dict[str, Any],
    provider: str = "gemini",
    level: Optional[int] = None,
    use_case: Optional[str] = None,
    temperature: float = 0.3,
) -> Dict[Tuple[str, str], Tuple[bool, float, str]]:
    """
    Evaluate term pairs using their web content with DSPy-based LLM interface.

    Args:
        term_pairs: List of (term1, term2) tuples
        web_content: Web content dictionary
        provider: LLM provider to use ("gemini", "openai", or tier-based)
        level: Optional level for use_case mapping (enables optimized prompts)
        use_case: Optional use_case override (e.g., 'deduplication', 'lv0_s1', 'lv0_s3', 'lv1_s1')
        temperature: LLM temperature setting

    Returns:
        Dictionary mapping term pairs to (is_duplicate, confidence, reason)
    """
    results = {}

    for term1, term2 in term_pairs:
        try:
            content1 = extract_relevant_content(term1, web_content)
            content2 = extract_relevant_content(term2, web_content)

            prompt = build_content_comparison_prompt(term1, term2, content1, content2)
            
            messages = [{"role": "user", "content": prompt}]
            
            # Determine use_case - prefer explicit use_case over level-based default
            use_case_final = use_case or (f"lv{level}_s1" if level is not None else "deduplication")
            
            try:
                if provider == "gemini":
                    obj = structured_completion(
                        messages=messages,
                        response_model=DedupDecision,
                        model="gemini/gemini-pro",
                        temperature=temperature,
                        use_case=use_case_final
                    )
                elif provider == "openai":
                    obj = structured_completion(
                        messages=messages,
                        response_model=DedupDecision,
                        model="openai/gpt-4",
                        temperature=temperature,
                        use_case=use_case_final
                    )
                else:
                    try:
                        # Prefer per-use-case model resolution without forcing a tier
                        obj = structured_completion(
                            messages=messages,
                            response_model=DedupDecision,
                            use_case=use_case_final,
                            temperature=temperature,
                        )
                    except Exception:
                        logging.warning("Unknown provider '%s'. Falling back to gemini/gemini-pro.", provider)
                        obj = structured_completion(
                            messages=messages,
                            response_model=DedupDecision,
                            model="gemini/gemini-pro",
                            temperature=temperature,
                            use_case=use_case_final,
                        )
                
                results[(term1, term2)] = (obj.verdict == "duplicate", float(obj.confidence), obj.reason)
            except Exception as inner_e:
                logging.error(f"Structured completion failed for {term1} vs {term2}: {inner_e}")
                # Fallback to old parsing method if structured completion fails
                try:
                    if provider == "gemini":
                        response = completion(
                            messages=messages,
                            model="gemini/gemini-pro",
                            temperature=temperature,
                            use_case=use_case_final
                        )
                    else:
                        response = completion(
                            messages=messages,
                            model="gemini/gemini-pro",
                            temperature=temperature,
                            use_case=use_case_final
                        )
                    is_duplicate, confidence, reason = parse_llm_response(response)
                    results[(term1, term2)] = (is_duplicate, confidence, reason)
                except Exception as fallback_e:
                    logging.error(f"Fallback completion also failed for {term1} vs {term2}: {fallback_e}")
                    results[(term1, term2)] = (False, 0.0, "Error in evaluation")

        except Exception as e:
            logging.error(f"Error evaluating {term1} vs {term2}: {e}")
            results[(term1, term2)] = (False, 0.0, "Error in evaluation")

    return results


def extract_relevant_content(term: str, web_content: Dict[str, Any]) -> str:
    """
    Extract relevant web content for a term.

    Args:
        term: Term to get content for
        web_content: Web content dictionary

    Returns:
        Concatenated relevant content
    """
    if term not in web_content:
        return ""

    content_parts = []

    results = web_content[term].get("results", [])[:3]

    for result in results:
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        url = result.get("url", "")

        if title or snippet:
            content_parts.append(f"URL: {url}\nTitle: {title}\nSnippet: {snippet}\n")

    return "\n".join(content_parts)


def build_content_comparison_prompt(
    term1: str, term2: str, content1: str, content2: str
) -> str:
    """
    Build prompt for comparing terms based on their web content.

    Args:
        term1: First term
        term2: Second term
        content1: Web content for term1
        content2: Web content for term2

    Returns:
        Prompt string
    """
    prompt = f"""You are an expert in academic and technical terminology. 
Analyze whether these two terms refer to the same concept based on their web content.

Term 1: "{term1}"
Web content for Term 1:
{content1[:1000]}

Term 2: "{term2}"
Web content for Term 2:
{content2[:1000]}

Based on the web content, determine if these terms are duplicates (refer to the same concept).

Consider:
- Are they describing the same academic field, department, or concept?
- Are they acronyms/abbreviations of each other?
- Do they have overlapping definitions but different names?

Respond in JSON format:
```json
{{
  "verdict": "duplicate" or "different",
  "confidence": 0.0-1.0,
  "reason": "brief explanation based on the web content"
}}
```"""

    return prompt


def parse_llm_response(response: str) -> Tuple[bool, float, str]:
    """
    Parse LLM response for single evaluation.

    Args:
        response: LLM response string

    Returns:
        Tuple of (is_duplicate, confidence, reason)
    """
    try:
        json_str = response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0]

        parsed = json.loads(json_str)

        verdict = parsed.get("verdict", "different")
        confidence = float(parsed.get("confidence", 0.0))
        reason = parsed.get("reason", "")

        is_duplicate = verdict.lower() == "duplicate"
        return is_duplicate, confidence, reason

    except Exception as e:
        logging.error(f"Error parsing LLM response: {e}")
        return False, 0.0, "Parse error"


# Edge creation helper functions (for future functional conversion)

def create_llm_analysis_edge(term1: str, term2: str, confidence: float, reason: str) -> Edge:
    """Create an LLM analysis edge object."""
    return Edge(
        source=term1,
        target=term2,
        weight=confidence,
        edge_type="llm_web_analysis",
        method="llm_based",
        metadata={"reason": reason}
    )


def extract_llm_params(config: LLMConfig) -> Dict[str, Any]:
    """Extract LLM parameters from config for function calls."""
    return {
        "provider": config.provider,
        "batch_size": config.batch_size,
        "confidence_threshold": config.confidence_threshold,
        "min_url_overlap": config.min_url_overlap,
        "max_url_overlap": config.max_url_overlap,
        "level": config.level,
        "use_case": config.use_case,
        "temperature": config.temperature
    }
