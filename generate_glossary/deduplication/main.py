"""
Main entry point for building the deduplication graph.

The graph is the core - everything else is just querying it.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import time

from .graph_builder import (
    create_deduplication_graph,
    add_terms_as_nodes,
    get_graph_stats,
    remove_weak_edges,
    validate_graph
)
from .rule_based_dedup import (
    add_rule_based_edges,
    add_acronym_edges,
    add_synonym_edges
)
from .web_based_dedup import (
    add_web_based_edges,
    add_domain_specific_edges,
    add_content_similarity_edges
)
from .llm_based_dedup import add_llm_based_edges
from .graph_io import save_graph, load_graph
from .api import get_canonical_terms, get_terms_with_variations


def build_graph(
    terms_by_level: Dict[int, List[str]],
    web_content: Optional[Dict[str, Any]] = None,
    existing_graph_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    use_llm: bool = False,
    llm_provider: str = "gemini"
) -> Any:
    """
    Build or extend the deduplication graph.
    
    This is the ONLY function needed to build the graph.
    Everything else is just querying this graph.
    
    Args:
        terms_by_level: Dictionary mapping level to terms
        web_content: Optional web content for all terms
        existing_graph_path: Optional path to existing graph to extend
        config: Configuration parameters
        
    Returns:
        The built/extended graph
    """
    start_time = time.time()
    
    # Default configuration
    if config is None:
        config = {
            "min_text_similarity": 0.85,
            "min_url_overlap": 2,
            "min_relevance_score": 0.3,
            "remove_weak_edges": True,
            "weak_edge_threshold": 0.3
        }
    
    # Load existing graph or create new
    if existing_graph_path:
        logging.info(f"Loading existing graph from {existing_graph_path}")
        graph = load_graph(existing_graph_path)
    else:
        logging.info("Creating new deduplication graph")
        graph = create_deduplication_graph()
    
    # Process each level
    levels = sorted(terms_by_level.keys())
    
    for level in levels:
        terms = terms_by_level[level]
        logging.info(f"Processing Level {level}: {len(terms)} terms")
        
        # Add terms as nodes
        graph = add_terms_as_nodes(graph, terms, level)
        
        # Add rule-based edges
        logging.info(f"Adding rule-based edges for level {level}")
        graph = add_rule_based_edges(
            graph,
            terms=terms,
            min_similarity=config["min_text_similarity"]
        )
        graph = add_acronym_edges(graph, terms=terms)
        graph = add_synonym_edges(graph, terms=terms)
        
        # Add web-based edges if content available
        if web_content:
            # Filter web content for current terms
            terms_web_content = {
                term: web_content[term]
                for term in terms
                if term in web_content
            }
            
            if terms_web_content:
                logging.info(f"Adding web-based edges for level {level}")
                graph = add_web_based_edges(
                    graph,
                    terms_web_content,
                    terms=terms,
                    min_url_overlap=config["min_url_overlap"],
                    min_relevance_score=config["min_relevance_score"]
                )
                graph = add_domain_specific_edges(graph, terms_web_content, terms=terms)
                graph = add_content_similarity_edges(graph, terms_web_content, terms=terms)
        
        # Add LLM-based edges if enabled and web content available
        if use_llm and web_content:
            # Filter web content for current terms
            terms_web_content_llm = {
                term: web_content[term]
                for term in terms
                if term in web_content
            }
            
            if terms_web_content_llm:
                logging.info(f"Adding LLM-based edges for level {level}")
                graph = add_llm_based_edges(
                    graph,
                    web_content=terms_web_content_llm,
                    terms=terms,
                    provider=llm_provider,
                    min_url_overlap=1,  # Terms with 1-2 URL overlap
                    max_url_overlap=config.get("min_url_overlap", 2),  # Below web-based threshold
                    confidence_threshold=0.8
                )
        
        # Log progress
        stats = get_graph_stats(graph)
        logging.info(
            f"Level {level} complete: {stats['num_edges']} edges, "
            f"{stats['num_components']} components"
        )
    
    # Remove weak edges if configured
    if config.get("remove_weak_edges", True):
        logging.info(f"Removing weak edges (threshold={config['weak_edge_threshold']})")
        graph = remove_weak_edges(graph, config["weak_edge_threshold"])
    
    # Validate graph
    is_valid, issues = validate_graph(graph)
    if not is_valid:
        logging.warning(f"Graph validation issues: {issues}")
    
    # Final stats
    final_stats = get_graph_stats(graph)
    logging.info(f"\n{'='*50}")
    logging.info("GRAPH BUILDING COMPLETE")
    logging.info(f"{'='*50}")
    logging.info(f"Nodes: {final_stats['num_nodes']}")
    logging.info(f"Edges: {final_stats['num_edges']}")
    logging.info(f"Components: {final_stats['num_components']}")
    logging.info(f"Processing time: {round(time.time() - start_time, 2)}s")
    
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