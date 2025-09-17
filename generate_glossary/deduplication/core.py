"""
Functional deduplication core with pure graph building orchestration.

This module provides a pure functional approach to deduplication graph building,
using immutable data structures, functional composition, and proper error handling.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Dict, List, Optional, Union, Any, Callable, Tuple, TypeVar
from dataclasses import dataclass

import networkx as nx

from .types import (
    Edge, EdgeBatch, DeduplicationConfig, RuleConfig, WebConfig, LLMConfig,
    EDGE_TYPES, METHODS
)
from .edges.rule_based import create_rule_edges
from .edges.web_based import create_web_edges
from .edges.llm_based import create_llm_edges
from .graph.builder import create_deduplication_graph, add_terms_as_nodes, get_graph_stats

# Type variables for functional composition
T = TypeVar('T')
E = TypeVar('E')


# Result types for functional error handling
@dataclass(frozen=True)
class Success:
    """Represents a successful operation result."""
    value: Any


@dataclass(frozen=True)
class Failure:
    """Represents a failed operation result."""
    error: str
    exception: Optional[Exception] = None


# Union type for Result
Result = Union[Success, Failure]

# Specific result types
EdgeCreationResult = Result
GraphBuildResult = Result


@dataclass(frozen=True)
class EdgeCreationTask:
    """Represents an edge creation task for parallel processing."""
    name: str
    creator_func: Callable[..., List[Edge]]
    args: Tuple
    kwargs: Dict[str, Any]


@dataclass(frozen=True)
class GraphBuildContext:
    """Context for graph building operations."""
    terms_by_level: Dict[int, List[str]]
    web_content: Optional[Dict[str, Any]]
    config: DeduplicationConfig
    base_graph: Optional[nx.Graph] = None


def is_success(result: Result) -> bool:
    """Check if a result is successful."""
    return isinstance(result, Success)


def is_failure(result: Result) -> bool:
    """Check if a result is a failure."""
    return isinstance(result, Failure)


def get_value(result: Result) -> Any:
    """Extract value from successful result."""
    if is_success(result):
        return result.value
    raise ValueError(f"Cannot get value from failure: {result.error}")


def get_error(result: Result) -> str:
    """Extract error from failed result."""
    if is_failure(result):
        return result.error
    raise ValueError("Cannot get error from successful result")


def safe_edge_creation(creator_func: Callable[..., List[Edge]]) -> Callable[..., EdgeCreationResult]:
    """
    Wrap an edge creation function to return Result type instead of raising exceptions.

    Args:
        creator_func: Function that creates edges

    Returns:
        Wrapped function that returns EdgeCreationResult
    """
    def wrapped(*args, **kwargs) -> EdgeCreationResult:
        try:
            edges = creator_func(*args, **kwargs)
            return Success(edges)
        except Exception as e:
            error_msg = f"Edge creation failed in {creator_func.__name__}: {str(e)}"
            logging.error(error_msg)
            return Failure(error_msg, e)

    return wrapped




def filter_edges(edges: List[Edge], predicate: Callable[[Edge], bool]) -> List[Edge]:
    """
    Filter edges based on a predicate function.

    Args:
        edges: List of edges to filter
        predicate: Function that returns True for edges to keep

    Returns:
        Filtered list of edges
    """
    return [edge for edge in edges if predicate(edge)]


def combine_edge_results(results: List[EdgeCreationResult]) -> EdgeCreationResult:
    """
    Combine multiple edge creation results into a single result.

    Args:
        results: List of EdgeCreationResult objects

    Returns:
        Combined EdgeCreationResult containing EdgeBatch with successful edges and errors
    """
    all_edges = []
    errors = []

    for result in results:
        if is_success(result):
            edges = get_value(result)
            all_edges.extend(edges)
        else:
            errors.append(get_error(result))

    # Return Success with EdgeBatch even if some tasks failed
    # Only return Failure if ALL tasks failed
    if errors and not all_edges:
        error_msg = f"All edge creation failed: {'; '.join(errors)}"
        logging.error(error_msg)
        return Failure(error_msg, Exception(error_msg))
    
    if errors:
        logging.warning(f"Some edge creation failed: {'; '.join(errors)}")
    
    edge_batch = EdgeBatch(edges=all_edges, errors=errors)
    return Success(edge_batch)


def parallel_edge_creation(
    tasks: List[EdgeCreationTask],
    max_workers: int = 3,
    timeout_per_task: float = 300.0
) -> List[EdgeCreationResult]:
    """
    Execute edge creation tasks in parallel with error isolation.

    Args:
        tasks: List of EdgeCreationTask objects to execute
        max_workers: Maximum number of parallel workers
        timeout_per_task: Timeout per task in seconds

    Returns:
        List of EdgeCreationResult objects (one per task)
    """
    import concurrent.futures as cf
    
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks directly with safe_edge_creation wrapper
        future_to_task = []
        for task in tasks:
            safe_creator = safe_edge_creation(task.creator_func)
            future = executor.submit(safe_creator, *task.args, **task.kwargs)
            future_to_task.append((future, task))

        # Process each future with proper timeout enforcement by iterating the original list
        for future, task in future_to_task:
            try:
                # Apply timeout directly to future.result() - this will enforce real timeout
                if timeout_per_task is None or timeout_per_task <= 0:
                    result = future.result()  # No timeout
                else:
                    result = future.result(timeout=timeout_per_task)
                
                results.append(result)
                
                if is_success(result):
                    logging.info(f"Edge creation task '{task.name}' completed successfully")
                else:
                    logging.error(f"Edge creation task '{task.name}' failed: {get_error(result)}")
                    
            except cf.TimeoutError as e:
                # Cancel the timed-out future and create failure result
                future.cancel()
                error_msg = f"Edge creation task '{task.name}' timed out after {timeout_per_task}s"
                logging.error(error_msg)
                results.append(Failure(error_msg, e))
                
            except Exception as e:
                error_msg = f"Edge creation task '{task.name}' failed with exception: {str(e)}"
                logging.error(error_msg)
                results.append(Failure(error_msg, e))

    return results


def create_edge_creation_tasks(
    terms: List[str],
    web_content: Optional[Dict[str, Any]],
    config: DeduplicationConfig
) -> List[EdgeCreationTask]:
    """
    Create edge creation tasks based on configuration.

    Args:
        terms: List of terms to process
        web_content: Optional web content for terms
        config: Deduplication configuration

    Returns:
        List of EdgeCreationTask objects
    """
    tasks = []

    # Always add rule-based task
    tasks.append(EdgeCreationTask(
        name="rule_based",
        creator_func=create_rule_edges,
        args=(terms,),
        kwargs={"config": config.rule_config}
    ))

    # Add web-based task if web content available
    if web_content:
        # Filter web content for current terms
        terms_web_content = {
            term: web_content[term]
            for term in terms
            if term in web_content
        }

        if terms_web_content:
            tasks.append(EdgeCreationTask(
                name="web_based",
                creator_func=create_web_edges,
                args=(terms, terms_web_content),
                kwargs={"config": config.web_config}
            ))

    # Add LLM-based task if enabled and web content available
    if config.llm_config is not None and web_content:
        # Filter web content for current terms
        terms_web_content_llm = {
            term: web_content[term]
            for term in terms
            if term in web_content
        }

        if terms_web_content_llm:
            tasks.append(EdgeCreationTask(
                name="llm_based",
                creator_func=create_llm_edges,
                args=(terms, terms_web_content_llm),
                kwargs={"config": config.llm_config}
            ))

    return tasks


def create_all_edges(
    terms: List[str],
    web_content: Optional[Dict[str, Any]],
    config: DeduplicationConfig
) -> EdgeCreationResult:
    """
    Orchestrate edge creation from all enabled methods.

    Args:
        terms: List of terms to process
        web_content: Optional web content for terms
        config: Deduplication configuration

    Returns:
        EdgeCreationResult containing all created edges
    """
    if not terms:
        return Success([])

    # Create edge creation tasks
    tasks = create_edge_creation_tasks(terms, web_content, config)

    if not tasks:
        logging.warning("No edge creation tasks created")
        return Success([])

    # Execute tasks
    if config.parallel_processing and len(tasks) > 1:
        logging.info(f"Executing {len(tasks)} edge creation tasks in parallel")
        results = parallel_edge_creation(tasks)
    else:
        logging.info(f"Executing {len(tasks)} edge creation tasks sequentially")
        results = []
        for task in tasks:
            safe_creator = safe_edge_creation(task.creator_func)
            result = safe_creator(*task.args, **task.kwargs)
            results.append(result)

            if is_success(result):
                edges_count = len(get_value(result))
                logging.info(f"Task '{task.name}' created {edges_count} edges")
            else:
                logging.error(f"Task '{task.name}' failed: {get_error(result)}")

    # Combine all results
    return combine_edge_results(results)


def add_edges_to_graph(graph: nx.Graph, edges: List[Edge]) -> nx.Graph:
    """
    Add edges to graph and return new graph (immutable operation).

    Args:
        graph: Existing graph
        edges: List of edges to add

    Returns:
        New graph with edges added
    """
    # Create a copy of the graph
    new_graph = graph.copy()

    for edge in edges:
        # Build edge data with core attributes
        edge_data = {
            "weight": edge.weight,
            "edge_type": edge.edge_type,
            "method": edge.method,
        }
        
        # Filter metadata to avoid keyword collisions with core attributes
        filtered_metadata = {
            key: value for key, value in edge.metadata.items()
            if key not in {"weight", "edge_type", "method"}
        }
        
        # Add edge with all data
        new_graph.add_edge(edge.source, edge.target, **edge_data, **filtered_metadata)

    # Update graph metadata
    new_graph.graph["total_edges"] = new_graph.number_of_edges()
    existing_edge_types = new_graph.graph.get("edge_types", set()).copy()
    existing_edge_types.update(edge.edge_type for edge in edges)
    new_graph.graph["edge_types"] = existing_edge_types

    return new_graph


def remove_weak_edges_functional(graph: nx.Graph, threshold: float) -> nx.Graph:
    """
    Remove weak edges from graph and return new graph (immutable operation).

    Args:
        graph: Existing graph
        threshold: Weight threshold below which edges are considered weak

    Returns:
        New graph with weak edges removed
    """
    new_graph = graph.copy()

    # Find weak edges
    weak_edges = [
        (u, v) for u, v, data in new_graph.edges(data=True)
        if data.get('weight', 1.0) < threshold
    ]

    # Remove weak edges
    new_graph.remove_edges_from(weak_edges)

    # Update metadata
    new_graph.graph["total_edges"] = new_graph.number_of_edges()

    logging.info(f"Removed {len(weak_edges)} weak edges (threshold={threshold})")

    return new_graph


def build_level_graph(
    level: int,
    terms: List[str],
    web_content: Optional[Dict[str, Any]],
    config: DeduplicationConfig,
    base_graph: Optional[nx.Graph] = None
) -> GraphBuildResult:
    """
    Build deduplication graph for a single level using pure functional approach.

    Args:
        level: Level number
        terms: List of terms for this level
        web_content: Optional web content for terms
        config: Deduplication configuration
        base_graph: Optional base graph to extend

    Returns:
        GraphBuildResult containing the built graph
    """
    try:
        logging.info(f"Building graph for Level {level}: {len(terms)} terms")

        # Start with base graph or create new
        if base_graph is not None:
            graph = base_graph.copy()
        else:
            graph = create_deduplication_graph()

        # Add terms as nodes
        graph = add_terms_as_nodes(graph, terms, level)

        # Create all edges for this level
        edge_result = create_all_edges(terms, web_content, config)

        if is_failure(edge_result):
            return Failure(f"Edge creation failed for level {level}: {get_error(edge_result)}")

        edge_batch = get_value(edge_result)
        edges = edge_batch.edges
        errors = edge_batch.errors
        
        logging.info(f"Created {len(edges)} edges for level {level}")
        if errors:
            logging.warning(f"Level {level} had {len(errors)} edge creation errors: {'; '.join(errors[:3])}{'...' if len(errors) > 3 else ''}")

        # Add edges to graph
        graph = add_edges_to_graph(graph, edges)

        # Log level stats
        stats = get_graph_stats(graph)
        logging.info(
            f"Level {level} complete: {stats['num_edges']} edges, "
            f"{stats['num_components']} components"
        )

        return Success(graph)

    except Exception as e:
        error_msg = f"Graph building failed for level {level}: {str(e)}"
        logging.error(error_msg)
        return Failure(error_msg, e)


def build_deduplication_graph(
    terms_by_level: Dict[int, List[str]],
    web_content: Optional[Dict[str, Any]] = None,
    config: Optional[DeduplicationConfig] = None,
    base_graph: Optional[nx.Graph] = None
) -> GraphBuildResult:
    """
    Build complete deduplication graph using pure functional composition.

    This is the main entry point for functional graph building.

    Args:
        terms_by_level: Dictionary mapping level to terms
        web_content: Optional web content for all terms
        config: Deduplication configuration (uses defaults if None)
        base_graph: Optional base graph to extend

    Returns:
        GraphBuildResult containing the complete built graph
    """
    start_time = time.time()

    # Use default config if none provided
    if config is None:
        config = DeduplicationConfig()

    logging.info("Starting functional deduplication graph building")
    logging.info(f"Levels to process: {sorted(terms_by_level.keys())}")
    logging.info(f"Parallel processing: {config.parallel_processing}")

    try:
        # Start with base graph or create new
        current_graph = base_graph.copy() if base_graph is not None else create_deduplication_graph()

        # Process each level
        levels = sorted(terms_by_level.keys())

        for level in levels:
            terms = terms_by_level[level]
            if not terms:
                logging.warning(f"No terms for level {level}, skipping")
                continue

            # Build graph for this level
            level_result = build_level_graph(level, terms, web_content, config, current_graph)

            if is_failure(level_result):
                return level_result  # Propagate failure

            current_graph = get_value(level_result)

        # Remove weak edges if configured
        if config.remove_weak_edges:
            logging.info(f"Removing weak edges (threshold={config.weak_edge_threshold})")
            current_graph = remove_weak_edges_functional(current_graph, config.weak_edge_threshold)

        # Final stats and validation
        final_stats = get_graph_stats(current_graph)
        processing_time = round(time.time() - start_time, 2)

        logging.info(f"\n{'='*50}")
        logging.info("FUNCTIONAL GRAPH BUILDING COMPLETE")
        logging.info(f"{'='*50}")
        logging.info(f"Nodes: {final_stats['num_nodes']}")
        logging.info(f"Edges: {final_stats['num_edges']}")
        logging.info(f"Components: {final_stats['num_components']}")
        logging.info(f"Processing time: {processing_time}s")

        return Success(current_graph)

    except Exception as e:
        error_msg = f"Functional graph building failed: {str(e)}"
        logging.error(error_msg)
        return Failure(error_msg, e)


def compose_edge_creators(*creators: Callable[..., List[Edge]]) -> Callable[..., List[Edge]]:
    """
    Compose multiple edge creation functions into a single function.

    Args:
        *creators: Edge creation functions to compose

    Returns:
        Composed function that creates edges from all creators
    """
    def composed(*args, **kwargs) -> List[Edge]:
        all_edges = []
        for creator in creators:
            try:
                edges = creator(*args, **kwargs)
                all_edges.extend(edges)
            except Exception as e:
                logging.error(f"Edge creator {creator.__name__} failed: {str(e)}")
                # Continue with other creators
        return all_edges

    return composed


def with_error_handling(creator_func: Callable[..., List[Edge]]) -> Callable[..., List[Edge]]:
    """
    Add error handling to an edge creation function.

    Args:
        creator_func: Edge creation function to wrap

    Returns:
        Wrapped function that handles errors gracefully
    """
    def wrapped(*args, **kwargs) -> List[Edge]:
        try:
            return creator_func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Edge creation failed in {creator_func.__name__}: {str(e)}")
            return []  # Return empty list on error

    return wrapped