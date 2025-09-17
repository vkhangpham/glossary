"""
Main entry point for building the deduplication graph.

The graph is the core - everything else is just querying it.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import time

# Keep only the imports that are actually used in the current functional approach
from .graph.builder import (
    get_graph_stats,
    validate_graph
)
from .graph.io import save_graph, load_graph
from .graph.query import get_canonical_terms

from .types import DeduplicationConfig, RuleConfig, WebConfig, LLMConfig
from .core import build_deduplication_graph, is_failure, get_value, get_error


def build_graph(
    terms_by_level: Optional[Dict[int, List[str]]] = None,
    web_content: Optional[Dict[str, Any]] = None,
    existing_graph_path: Optional[str] = None,
    config: Optional[Union[Dict[str, Any], DeduplicationConfig]] = None,
    use_llm: bool = False,
    llm_provider: str = "gemini",
    # Legacy parameters for backward compatibility
    terms: Optional[List[str]] = None,
    level: Optional[int] = None
) -> Any:
    """
    Build or extend the deduplication graph using functional core orchestration.

    This is the ONLY function needed to build the graph.
    Everything else is just querying this graph.

    Args:
        terms_by_level: Dictionary mapping level to terms
        web_content: Optional web content for all terms
        existing_graph_path: Optional path to existing graph to extend
        config: Configuration parameters (DeduplicationConfig or legacy dict)
        use_llm: Whether to use LLM-based deduplication (fallback for backward compatibility)
        llm_provider: LLM provider to use (fallback for backward compatibility)
        terms: Legacy parameter - list of terms (for backward compatibility)
        level: Legacy parameter - level for the terms (for backward compatibility)

    Returns:
        The built/extended graph
    """
    start_time = time.time()

    # Handle legacy parameters
    if terms is not None and terms_by_level is None:
        # Construct terms_by_level from legacy parameters
        use_level = level if level is not None else 0
        terms_by_level = {use_level: terms}
        logging.info(f"Using legacy parameters: {len(terms)} terms at level {use_level}")
    elif terms_by_level is None:
        raise ValueError("Either terms_by_level or terms must be provided")

    # Convert configuration to DeduplicationConfig object
    if config is None:
        config = create_default_config(use_llm=use_llm, llm_provider=llm_provider)
    elif isinstance(config, dict):
        # Convert legacy dict config to DeduplicationConfig
        config = convert_legacy_config(config)
        # Override LLM settings if provided as function parameters
        if use_llm and config.llm_config is None:
            config = DeduplicationConfig(
                rule_config=config.rule_config,
                web_config=config.web_config,
                llm_config=LLMConfig(provider=llm_provider),
                remove_weak_edges=config.remove_weak_edges,
                weak_edge_threshold=config.weak_edge_threshold,
                parallel_processing=config.parallel_processing
            )
    
    # Load existing graph or prepare base graph
    base_graph = None
    if existing_graph_path:
        logging.info(f"Loading existing graph from {existing_graph_path}")
        base_graph = load_graph(existing_graph_path)
    else:
        logging.info("Creating new deduplication graph")
    
    # Use functional core for graph building
    logging.info("Using functional core orchestration for graph building")
    result = build_deduplication_graph(
        terms_by_level=terms_by_level,
        web_content=web_content,
        config=config,
        base_graph=base_graph
    )
    
    # Handle functional result
    if is_failure(result):
        error_msg = get_error(result)
        logging.error(f"Functional graph building failed: {error_msg}")
        raise RuntimeError(f"Graph building failed: {error_msg}")
    
    graph = get_value(result)
    
    # Integration test: Verify graph compatibility with validation and stats functions
    # This ensures the functional graph works with existing graph utilities
    try:
        # Test validate_graph function
        is_valid, issues = validate_graph(graph)
        if not is_valid:
            logging.warning(f"Graph validation issues: {issues}")
        
        # Test get_graph_stats function  
        stats = get_graph_stats(graph)
        logging.info(f"Graph stats integration test passed. Nodes: {stats.get('node_count', 'unknown')}, Edges: {stats.get('edge_count', 'unknown')}")
        
    except Exception as e:
        # If validation or stats functions fail, we need to update imports or add adapters
        logging.error(f"Graph compatibility test failed: {e}")
        logging.error("The functional graph may not be compatible with existing validation/stats utilities")
        logging.error("Consider updating imports or adding adapter functions in graph/builder.py")
        # Don't raise here - let the process continue, but warn about compatibility
    
    # Final timing log
    processing_time = round(time.time() - start_time, 2)
    logging.info(f"Total processing time (including functional core): {processing_time}s")
    
    return graph


def main(
    terms_path: str,
    level: int,
    web_content_path: Optional[str] = None,
    existing_graph_path: Optional[str] = None,
    output_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    use_llm: bool = False,
    llm_provider: str = "gemini"
) -> Dict[str, Any]:
    """
    Main entry point for CLI or programmatic use.
    
    Args:
        terms_path: Path to terms file
        level: Level for these terms
        web_content_path: Optional path to web content JSON
        existing_graph_path: Optional path to existing graph
        output_path: Optional path to save graph
        config: Optional configuration
        
    Returns:
        Result dictionary with graph and optionally canonical terms
    """
    # Read terms
    with open(terms_path, 'r', encoding='utf-8') as f:
        terms = [line.strip() for line in f if line.strip()]
    
    logging.info(f"Read {len(terms)} terms from {terms_path}")
    
    # Read web content if provided
    web_content = None
    if web_content_path:
        import json
        with open(web_content_path, 'r', encoding='utf-8') as f:
            web_content = json.load(f)
        logging.info(f"Loaded web content for {len(web_content)} terms")
    
    # Build graph
    graph = build_graph(
        terms_by_level={level: terms},
        web_content=web_content,
        existing_graph_path=existing_graph_path,
        config=config,
        use_llm=use_llm,
        llm_provider=llm_provider
    )
    
    # Save graph if output path provided
    if output_path:
        save_graph(graph, output_path)
        logging.info(f"Saved graph to {output_path}")
    
    # Return result
    result = {
        "graph": graph,
        "stats": get_graph_stats(graph)
    }
    
    # Optionally get canonical terms (using API)
    if not existing_graph_path:
        # Only get canonical terms if this is a complete build
        canonical_terms = get_canonical_terms(graph)
        result["canonical_terms"] = canonical_terms
        logging.info(f"Selected {len(canonical_terms)} canonical terms")
    
    return result


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Build deduplication graph - the core of all deduplication"
    )
    parser.add_argument("terms", help="Path to terms file")
    parser.add_argument("-l", "--level", type=int, required=True, 
                       help="Hierarchy level for these terms (0=colleges, 1=departments, etc)")
    parser.add_argument("-w", "--web-content", help="Path to web content JSON")
    parser.add_argument("-g", "--existing-graph", help="Path to existing graph to extend")
    parser.add_argument("-o", "--output", required=True, help="Path to save graph (without extension)")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--use-llm", action="store_true",
                       help="Use LLM for semantic duplicate detection")
    parser.add_argument("--llm-provider", default="gemini",
                       choices=["gemini", "openai"],
                       help="LLM provider to use (default: gemini)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Run main
        result = main(
            args.terms,
            args.level,
            args.web_content,
            args.existing_graph,
            f"{args.output}.graph.pkl",
            use_llm=args.use_llm,
            llm_provider=args.llm_provider
        )
        
        # Also save canonical terms for convenience
        if "canonical_terms" in result:
            output_txt = f"{args.output}.txt"
            with open(output_txt, 'w', encoding='utf-8') as f:
                for term in sorted(result["canonical_terms"]):
                    f.write(f"{term}\n")
            logging.info(f"Saved {len(result['canonical_terms'])} canonical terms to {output_txt}")
        
        print(f"\n✓ Graph built successfully!")
        print(f"  Nodes: {result['stats']['num_nodes']}")
        print(f"  Edges: {result['stats']['num_edges']}")
        print(f"  Components: {result['stats']['num_components']}")
        print(f"  Graph saved to: {args.output}.graph.pkl")
        
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


# Configuration conversion helper functions

def convert_legacy_config(config_dict: Dict[str, Any]) -> DeduplicationConfig:
    """Convert legacy dictionary config to DeduplicationConfig object."""
    rule_config = RuleConfig(
        min_similarity=config_dict.get("min_text_similarity", 0.85),
        synonym_patterns=config_dict.get("synonym_patterns", {}),
        blacklist_terms=frozenset(config_dict.get("blacklist_terms", [])),
        enable_acronym_detection=config_dict.get("enable_acronym_detection", True),
        enable_synonym_detection=config_dict.get("enable_synonym_detection", True)
    )

    web_config = WebConfig(
        min_url_overlap=config_dict.get("min_url_overlap", 2),
        min_relevance_score=config_dict.get("min_relevance_score", 0.3),
        domain_patterns=config_dict.get("domain_patterns", {}),
        min_content_similarity=config_dict.get("min_content_similarity", 0.6),
        enable_domain_specific=config_dict.get("enable_domain_specific", True),
        enable_content_similarity=config_dict.get("enable_content_similarity", True)
    )

    llm_config = None
    if any(key.startswith("llm_") for key in config_dict.keys()) or \
       config_dict.get("use_llm", False):
        llm_config = LLMConfig(
            provider=config_dict.get("llm_provider", "gemini"),
            batch_size=config_dict.get("llm_batch_size", 5),
            confidence_threshold=config_dict.get("llm_confidence_threshold", 0.8),
            min_url_overlap=config_dict.get("llm_min_url_overlap", 1),
            max_url_overlap=config_dict.get("llm_max_url_overlap", 2),
            level=config_dict.get("level"),
            use_case=config_dict.get("use_case"),
            temperature=config_dict.get("llm_temperature", 0.3)
        )

    return DeduplicationConfig(
        rule_config=rule_config,
        web_config=web_config,
        llm_config=llm_config,
        remove_weak_edges=config_dict.get("remove_weak_edges", True),
        weak_edge_threshold=config_dict.get("weak_edge_threshold", 0.3),
        parallel_processing=config_dict.get("parallel_processing", True)
    )


def create_default_config(use_llm: bool = False, llm_provider: str = "gemini") -> DeduplicationConfig:
    """Create default configuration with optional LLM settings."""
    rule_config = RuleConfig()
    web_config = WebConfig()

    llm_config = None
    if use_llm:
        llm_config = LLMConfig(provider=llm_provider)

    return DeduplicationConfig(
        rule_config=rule_config,
        web_config=web_config,
        llm_config=llm_config
    )