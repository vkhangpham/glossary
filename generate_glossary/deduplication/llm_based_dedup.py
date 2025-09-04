"""
LLM-based deduplication for adding edges to the graph.

Uses language models to analyze terms that have SOME web overlap (not enough
for web-based edges) by comparing their web content to determine if they're duplicates.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
import networkx as nx
import json

from generate_glossary.utils.llm import completion


def add_llm_based_edges(
    graph: nx.Graph,
    web_content: Dict[str, Any],
    terms: Optional[List[str]] = None,
    provider: str = "gemini",
    min_url_overlap: int = 1,
    max_url_overlap: int = 2,
    batch_size: int = 5,
    confidence_threshold: float = 0.8,
) -> nx.Graph:
    """
    Add edges based on LLM analysis of shared web content.

    This method focuses on term pairs that have SOME URL overlap (not enough
    for web-based edges) and uses LLM to analyze their web content to determine
    if they're duplicates.

    Args:
        graph: NetworkX graph to add edges to
        web_content: Web content dictionary for all terms
        terms: Optional list of terms to process (if None, process all)
        provider: LLM provider to use ("gemini" or "openai")
        min_url_overlap: Minimum URL overlap to consider (default: 1)
        max_url_overlap: Maximum URL overlap (below web-based threshold, default: 2)
        batch_size: Number of term pairs to evaluate per LLM call
        confidence_threshold: Minimum confidence for creating edge

    Returns:
        Updated graph with LLM-based edges
    """
    if terms is None:
        terms = list(graph.nodes())

    edges_added = 0

    candidate_pairs = find_candidate_pairs_with_overlap(
        terms, web_content, min_url_overlap, max_url_overlap
    )

    logging.info(
        f"Found {len(candidate_pairs)} candidate pairs with URL overlap {min_url_overlap}-{max_url_overlap}"
    )

    for i in range(0, len(candidate_pairs), batch_size):
        batch = candidate_pairs[i : i + batch_size]

        results = evaluate_pairs_with_web_content(batch, web_content, provider)

        for (term1, term2), (is_duplicate, confidence, reason) in results.items():
            if is_duplicate and confidence >= confidence_threshold:
                if (
                    term1 in graph
                    and term2 in graph
                    and not graph.has_edge(term1, term2)
                ):
                    graph.add_edge(
                        term1,
                        term2,
                        weight=confidence,
                        edge_type="llm_web_analysis",
                        method="llm_based",
                        reason=reason,
                    )
                    edges_added += 1
                    logging.debug(
                        f"Added LLM edge: {term1} <-> {term2} "
                        f"(confidence={confidence:.2f}, reason={reason})"
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
) -> Dict[Tuple[str, str], Tuple[bool, float, str]]:
    """
    Evaluate term pairs using their web content.

    Args:
        term_pairs: List of (term1, term2) tuples
        web_content: Web content dictionary
        provider: LLM provider to use

    Returns:
        Dictionary mapping term pairs to (is_duplicate, confidence, reason)
    """
    results = {}

    for term1, term2 in term_pairs:
        try:
            content1 = extract_relevant_content(term1, web_content)
            content2 = extract_relevant_content(term2, web_content)

            prompt = build_content_comparison_prompt(term1, term2, content1, content2)

            model = "gemini/gemini-pro" if provider == "gemini" else "openai/gpt-4"
            messages = [{"role": "user", "content": prompt}]

            response = completion(model=model, messages=messages, temperature=0.3)

            is_duplicate, confidence, reason = parse_llm_response(response)
            results[(term1, term2)] = (is_duplicate, confidence, reason)

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
