"""
Graph-based deduplication module for concept terms.

This module implements a graph-based approach to term deduplication,
addressing transitive relationship issues in multi-level, multi-mode deduplication.
"""

import networkx as nx
import logging
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
import itertools
import nltk
from nltk.corpus import wordnet
from tqdm import tqdm
from urllib.parse import urlparse
import asyncio
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
# from scipy.spatial.distance import cosine # Using numpy implementation instead

from generate_glossary.deduplicator.dedup_utils import (
    normalize_text,
    get_term_variations,
    is_compound_term,
    timing_decorator,
    get_plural_variations,
    get_spelling_variations,
    get_dash_space_variations,
    SPELLING_VARIATIONS,
)
from generate_glossary.utils.llm import LLMFactory, Provider, OPENAI_MODELS, GEMINI_MODELS, BaseLLM

try:
    from generate_glossary.validator.validation_utils import calculate_relevance_score

    has_relevance_calculator = True
except ImportError:
    has_relevance_calculator = False
    logging.warning("Could not import relevance score calculator, using default scores")

load_dotenv('.env')

# Type alias for clarity
TermsByLevel = Dict[int, List[str]]
WebContent = Dict[str, List[Dict[str, Any]]]
CanonicalMapping = Dict[str, Set[str]]
DeduplicationResult = Dict[str, Any]

# Define similarity threshold
# Analysis of hierarchy.json data suggests 0.49 provides optimal F1 score for separating
# duplicate variations (mean=0.78) from distinct terms (mean=0.15)
EMBEDDING_SIMILARITY_THRESHOLD = 0.49


# Helper function for similarity (assuming embeddings are numpy arrays)
def calculate_embedding_similarity(embedding1: Optional[np.ndarray], embedding2: Optional[np.ndarray]) -> float:
    """Calculates cosine similarity between two embedding vectors."""
    if embedding1 is None or embedding2 is None or not isinstance(embedding1, np.ndarray) or not isinstance(embedding2, np.ndarray):
        return 0.0
    # Manual calculation:
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    # Clip to avoid potential floating point issues leading to values slightly outside [-1, 1]
    similarity = np.clip(dot_product / (norm1 * norm2), -1.0, 1.0)
    return float(similarity)


def init_llm(provider: Optional[str] = None) -> BaseLLM:
    """
    Initialize LLM with specified provider
    
    Args:
        provider: Optional provider name
        
    Returns:
        LLM instance
    """
    if not provider:
        provider = Provider.OPENAI
        
    return LLMFactory.create_llm(
        provider=provider,
        model=GEMINI_MODELS["default"] if provider == Provider.GEMINI else OPENAI_MODELS["default"],
        temperature=0
    )

@timing_decorator
def deduplicate_graph_based(
    terms_by_level: Dict[int, List[str]],
    web_content: Optional[Dict[str, List[WebContent]]] = None,
    url_overlap_threshold: int = 2,
    min_relevance_score: float = 0.75,
    cache_dir: Optional[str] = None,
    current_level: Optional[int] = None,
) -> DeduplicationResult:
    """Graph-based deduplication that combines rule-based and web-based approaches.

    This function builds a graph where:
    - Nodes are terms
    - Edges indicate terms are variations of each other
    - Edge weights indicate confidence in the relationship

    This implementation adopts a more conservative approach:
    1. Higher URL overlap thresholds to avoid spurious connections
    2. Level-aware processing to respect boundaries
    3. No hierarchical relationships to avoid false positives

    Args:
        terms_by_level: Dictionary mapping level numbers to lists of terms
        web_content: Optional dictionary mapping terms to their web content
        url_overlap_threshold: Minimum number of overlapping URLs required (default: 5)
        min_relevance_score: Minimum relevance score for content to be considered relevant
        cache_dir: Optional directory to cache/load graph data for incremental processing
        current_level: Optional current level being processed (for incremental processing)

    Returns:
        DeduplicationResult with terms and variations, respecting level boundaries
    """
    logging.info(
        f"Starting graph-based deduplication with {sum(len(terms) for terms in terms_by_level.values())} terms"
    )

    # Figure out current level if not provided explicitly
    if current_level is None:
        current_level = max(terms_by_level.keys())
    
    # Check if we have cached graph data from previous runs
    G = None
    term_embeddings = {}
    cached_levels = set()
    
    if cache_dir:
        try:
            import pickle
            import os
            
            # Ensure cache directory exists
            os.makedirs(cache_dir, exist_ok=True)
            
            # Path for cached graph data
            graph_cache_path = os.path.join(cache_dir, "graph_cache.pickle")
            embeddings_cache_path = os.path.join(cache_dir, "embeddings_cache.pickle")
            levels_cache_path = os.path.join(cache_dir, "processed_levels.pickle")
            
            # Check if cache files exist
            if os.path.exists(graph_cache_path) and os.path.exists(embeddings_cache_path) and os.path.exists(levels_cache_path):
                logging.info("Loading cached graph data from previous runs...")
                
                # Load cached graph
                with open(graph_cache_path, "rb") as f:
                    G = pickle.load(f)
                    
                # Load cached embeddings
                with open(embeddings_cache_path, "rb") as f:
                    term_embeddings = pickle.load(f)
                    
                # Load cached processed levels
                with open(levels_cache_path, "rb") as f:
                    cached_levels = pickle.load(f)
                    
                logging.info(f"Loaded cached graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
                logging.info(f"Loaded embeddings for {len(term_embeddings)} terms")
                logging.info(f"Previously processed levels: {cached_levels}")
        except Exception as e:
            logging.error(f"Error loading cached graph data: {e}. Building new graph from scratch.")
            G = None
            term_embeddings = {}
            cached_levels = set()

    logging.info("Initializing embedding model...")
    # Ensure you handle model loading appropriately (e.g., download if needed)
    try:
        # Example model, choose based on performance/needs
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        logging.error(f"Failed to load SentenceTransformer model: {e}. Embedding similarity check will be skipped.")
        embedder = None

    # Determine which terms need embeddings
    all_terms_list = [term for level, terms in terms_by_level.items() 
                     for term in terms if level not in cached_levels or term not in term_embeddings]
    
    if embedder and all_terms_list:
        logging.info(f"Computing embeddings for {len(all_terms_list)} new terms...")
        try:
            embeddings_list = embedder.encode(all_terms_list, show_progress_bar=True, convert_to_numpy=True)
            # Add new embeddings to existing dictionary
            for term, emb in zip(all_terms_list, embeddings_list):
                term_embeddings[term] = emb
            logging.info("Embeddings computed.")
        except Exception as e:
            logging.error(f"Error computing embeddings: {e}. Embedding similarity check will be skipped.")
            embedder = None # Disable further embedding checks if computation fails
    else:
        if not all_terms_list:
            logging.info("No new terms to compute embeddings for.")
        else:
            logging.warning("Embedder not available. Skipping embedding computation and similarity checks.")

    # Build initial graph if we don't have a cached one
    if G is None:
        # 1. Build initial graph with all terms (pass embeddings)
        G = build_term_graph(terms_by_level, term_embeddings)
    else:
        # If we have a cached graph, just add new terms
        new_terms_to_add = {}
        for level, terms in terms_by_level.items():
            if level not in cached_levels:
                new_terms_to_add[level] = terms
        
        if new_terms_to_add:
            logging.info(f"Adding {sum(len(terms) for terms in new_terms_to_add.values())} new terms to existing graph")
            # Add the new terms to the graph
            for level, terms in new_terms_to_add.items():
                for term in terms:
                    # Skip if node already exists
                    if term in G:
                        continue
                    # Add node attributes
                    G.add_node(
                        term,
                        level=level,
                        has_sciences_suffix=term.endswith(("sciences", "studies")),
                        word_count=len(term.split()),
                        embedding=term_embeddings.get(term)
                    )

    # Calculate which levels need to be processed
    levels_to_process = set(terms_by_level.keys()) - cached_levels
    
    if not levels_to_process:
        logging.info("All levels already processed. Using cached graph without further processing.")
    else:
        logging.info(f"Processing levels: {levels_to_process}")
        
        # 2. Add rule-based relationships with level awareness
        # Only process terms from levels that haven't been processed yet
        terms_to_process = {}
        for level in levels_to_process:
            if level in terms_by_level:
                terms_to_process[level] = terms_by_level[level]
        
        if terms_to_process:
            G = add_rule_based_edges(G, terms_to_process)

        # 3. Add compound term edges for new terms
        # Get all terms from all levels for compound term checking
        all_terms = [term for terms in terms_by_level.values() for term in terms]
        # But only process the new terms for compound relationships
        new_terms = [term for level, terms in terms_to_process.items() for term in terms]
        if new_terms:
            G = add_compound_term_edges(G, all_terms, new_terms)

        # 4. Add web-based relationships if web content is available
        if web_content:
            # For web-based edges, we need to consider terms from all levels
            # to find relationships between new terms and existing terms
            G = add_web_based_edges(
                G,
                terms_by_level,
                web_content,
                url_overlap_threshold=url_overlap_threshold,
                min_relevance_score=min_relevance_score,
                levels_to_process=levels_to_process,
            )

        # 5. Add level weighting information
        G = add_level_weights_to_edges(G, terms_by_level)

        # 6. Find and add explicit transitive relationships 
        # Only process transitive relationships involving new terms
        new_terms_set = set(new_terms)
        transitive_edges = find_transitive_relationships(G, all_terms, new_terms_set)
        G.add_edges_from(transitive_edges)

    # 7. Select canonical terms for each connected component, respecting level boundaries
    canonical_mapping = select_canonical_terms(G, terms_by_level)

    # 8. Clean up the canonical mapping to prevent inappropriate groupings
    canonical_mapping = clean_canonical_mapping(canonical_mapping)

    # 9. Convert to standard output format
    result = convert_to_deduplication_result(canonical_mapping, terms_by_level, G)

    # Save cache if cache_dir provided
    if cache_dir:
        try:
            import pickle
            import os
            
            # Ensure cache directory exists
            os.makedirs(cache_dir, exist_ok=True)
            
            # Path for cached graph data
            graph_cache_path = os.path.join(cache_dir, "graph_cache.pickle")
            embeddings_cache_path = os.path.join(cache_dir, "embeddings_cache.pickle")
            levels_cache_path = os.path.join(cache_dir, "processed_levels.pickle")
            
            # Save graph
            with open(graph_cache_path, "wb") as f:
                pickle.dump(G, f)
                
            # Save embeddings
            with open(embeddings_cache_path, "wb") as f:
                pickle.dump(term_embeddings, f)
                
            # Save processed levels (add newly processed levels)
            cached_levels.update(levels_to_process)
            with open(levels_cache_path, "wb") as f:
                pickle.dump(cached_levels, f)
                
            logging.info(f"Cached graph data saved to {cache_dir}")
        except Exception as e:
            logging.error(f"Error saving cached graph data: {e}")

    # Log results with detailed breakdown
    deduplicated_count = len(result["deduplicated_terms"])
    variation_count = sum(len(v) for v in result.get("variations", {}).values())
    
    # Count edges by detection method
    llm_verified_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("detection_method") == "llm_verified")
    wiki_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("detection_method") == "wikipedia_high_relevance")
    llm_wiki_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("detection_method") == "llm_verified_wikipedia")
    rule_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("detection_method") in ["suffix", "morphological_variant", "spelling", "plural_singular", "dash_space"])
    url_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("detection_method") == "url_overlap")
    transitive_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("detection_method") == "transitive")

    logging.info(f"Graph-based deduplication results:")
    logging.info(f"- {deduplicated_count} canonical terms")
    logging.info(f"- {variation_count} variations (same-level)")
    logging.info(f"- Edge breakdown:")
    logging.info(f"  - {rule_edges} rule-based edges")
    logging.info(f"  - {url_edges} URL overlap edges")
    logging.info(f"  - {wiki_edges} Wikipedia high relevance edges")
    logging.info(f"  - {llm_wiki_edges} LLM-verified Wikipedia edges")
    logging.info(f"  - {llm_verified_edges} LLM-verified edges")
    logging.info(f"  - {transitive_edges} transitive relationship edges")

    return result


def build_term_graph(terms_by_level: TermsByLevel, term_embeddings: Dict[str, np.ndarray]) -> nx.Graph:
    """
    Builds a graph where nodes are terms and edges represent relationships.

    Args:
        terms_by_level: Dict mapping level numbers to lists of terms
        term_embeddings: Dict mapping terms to their embeddings

    Returns:
        NetworkX graph with terms as nodes
    """
    G = nx.Graph()

    # Add all terms as nodes with level attribute
    for level, terms in terms_by_level.items():
        for term in terms:
            # Add node attributes that might be useful for later processing
            G.add_node(
                term,
                level=level,
                has_sciences_suffix=term.endswith(("sciences", "studies")),
                word_count=len(term.split()),
                embedding=term_embeddings.get(term) # Add embedding attribute
            )

    logging.info(f"Created graph with {G.number_of_nodes()} nodes")
    return G


def add_rule_based_edges(G: nx.Graph, terms_by_level: TermsByLevel) -> nx.Graph:
    """
    Adds edges to the graph based on rule-based relationships.

    Handles several specific types of variations:
    1. Plural/singular forms using lemmatization with POS tagging
    2. British/American spelling variations
    3. Dash-space variations
    4. Academic field suffixes: "studies", "sciences", "technologies", "education",
       "research", "techniques", "algorithms", "systems"
    5. Morphological variations: "politics" -> "political science", "environment" -> "environmental sciences"

    Each variation type is handled separately with clear documentation of the
    relationship type and detection method.

    Args:
        G: NetworkX graph to add edges to
        terms_by_level: Dict mapping level numbers to lists of terms

    Returns:
        Updated graph with rule-based relationship edges
    """
    initial_edge_count = G.number_of_edges()

    # Create term to level mapping
    term_to_level = {}
    for level, terms in terms_by_level.items():
        for term in terms:
            term_to_level[term] = level

    all_terms = [term for terms in terms_by_level.values() for term in terms]

    # Define the academic suffixes to check
    ACADEMIC_SUFFIXES = [
        "studies",
        "sciences",
        "technologies",
        "education",
        "research",
        "techniques",
        "algorithms",
        "systems",
        "theories",
        "methods",
        "principles",
    ]

    # Create a mapping of plural/singular forms for the suffixes
    SUFFIX_VARIATIONS = {
        # Plural -> Singular
        "studies": "study",
        "sciences": "science",
        "technologies": "technology",
        "techniques": "technique",
        "algorithms": "algorithm",
        "systems": "system",
        "theories": "theory",
        "methods": "method",
        "principles": "principle",
        # Singular -> Plural (for lookup in the other direction)
        "study": "studies",
        "science": "sciences",
        "technology": "technologies",
        "technique": "techniques",
        "algorithm": "algorithms",
        "system": "systems",
        "theory": "theories",
        "method": "methods",
        "principle": "principles",
        # These don't change in plural/singular form
        "education": "education",
        "research": "research",
    }

    # Define morphological variants for academic terms
    MORPHOLOGICAL_VARIANTS = {
        # Base form -> Adjectival form
        "politics": "political",
        "environment": "environmental",
        "economics": "economic",
        "biology": "biological",
        "chemistry": "chemical",
        "history": "historical",
        "geography": "geographical",
        "philosophy": "philosophical",
        "psychology": "psychological",
        "sociology": "sociological",
        "anthropology": "anthropological",
        "mathematics": "mathematical",
        "statistics": "statistical",
        "physics": "physical",
        "art": "artistic",
        "music": "musical",
        "literature": "literary",
        "linguistics": "linguistic",
        "religion": "religious",
        "business": "business",  # some don't change
        "education": "educational",
        "communication": "communications",
        "technology": "technological",
        "computer": "computational",
        "medicine": "medical",
        "law": "legal",
        # Adjectival form -> Base form (for reverse lookup)
        "political": "politics",
        "environmental": "environment",
        "economic": "economics",
        "biological": "biology",
        "chemical": "chemistry",
        "historical": "history",
        "geographical": "geography",
        "philosophical": "philosophy",
        "psychological": "psychology",
        "sociological": "sociology",
        "anthropological": "anthropology",
        "mathematical": "mathematics",
        "statistical": "statistics",
        "physical": "physics",
        "artistic": "art",
        "musical": "music",
        "literary": "literature",
        "linguistic": "linguistics",
        "religious": "religion",
        "educational": "education",
        "communications": "communication",
        "technological": "technology",
        "computational": "computer",
        "medical": "medicine",
        "legal": "law",
    }

    # Combine original suffixes with their variations
    all_suffix_forms = set(ACADEMIC_SUFFIXES + list(SUFFIX_VARIATIONS.keys()))

    # Pre-compute variations for all terms
    term_variations = {term: get_term_variations(term) for term in all_terms}
    plural_variations = {term: get_plural_variations(term) for term in all_terms}
    spelling_variations = {term: get_spelling_variations(term) for term in all_terms}
    dash_variations = {term: get_dash_space_variations(term) for term in all_terms}

    # Create lookup dictionaries for faster matching
    plural_lookup = {}
    for term, variations in plural_variations.items():
        for var in variations:
            if var not in plural_lookup:
                plural_lookup[var] = set()
            plural_lookup[var].add(term)

    spelling_lookup = {}
    for term, variations in spelling_variations.items():
        for var in variations:
            if var not in spelling_lookup:
                spelling_lookup[var] = set()
            spelling_lookup[var].add(term)

    dash_lookup = {}
    for term, variations in dash_variations.items():
        for var in variations:
            if var not in dash_lookup:
                dash_lookup[var] = set()
            dash_lookup[var].add(term)

    # Process each term for academic suffix variations
    academic_suffix_pairs = []

    for term1 in all_terms:
        level1 = term_to_level.get(term1, float("inf"))
        term1_lower = term1.lower()

        # Check if term1 ends with an academic suffix
        # Using word boundary check to avoid matching within words like "disorders"
        has_suffix = False
        matching_suffix = None
        canonical_suffix = None

        for suffix in all_suffix_forms:
            # Check if term ends with the suffix as a whole word
            if term1_lower.endswith(f" {suffix}"):
                has_suffix = True
                matching_suffix = suffix
                # Get the canonical form of the suffix (prefer plural forms)
                if suffix in SUFFIX_VARIATIONS and not suffix in ACADEMIC_SUFFIXES:
                    canonical_suffix = SUFFIX_VARIATIONS[suffix]
                else:
                    canonical_suffix = suffix
                break

        # If term1 has a suffix, extract the base term and look for matches
        if has_suffix and matching_suffix:
            # Extract the base term (e.g., "biology" from "biology studies")
            # Using proper word boundary to extract the exact base term
            base_term = term1_lower[
                : -len(matching_suffix) - 1
            ].strip()  # -1 for the space

            # Look for other terms that match this base term
            for term2 in all_terms:
                if term1 == term2:
                    continue

                level2 = term_to_level.get(term2, float("inf"))
                is_cross_level = level1 != level2

                # MODIFIED: Allow cross-level connections but track them for later

                term2_lower = term2.lower()

                # Check for direct match with base term
                if term2_lower == base_term:
                    # Direct match with base term
                    academic_suffix_pairs.append(
                        (term1, term2, canonical_suffix, level1 == level2)
                    )
                    continue

                # Check for morphological variants of the base term
                if base_term in MORPHOLOGICAL_VARIANTS:
                    adjectival_form = MORPHOLOGICAL_VARIANTS[base_term]
                    if term2_lower == adjectival_form:
                        academic_suffix_pairs.append(
                            (
                                term1,
                                term2,
                                canonical_suffix,
                                level1 == level2,
                                "morphological",
                            )
                        )
                        continue

                # Check if base_term is an adjectival form with a corresponding base form
                if base_term in MORPHOLOGICAL_VARIANTS:
                    base_form = MORPHOLOGICAL_VARIANTS[base_term]
                    if term2_lower == base_form:
                        academic_suffix_pairs.append(
                            (
                                term1,
                                term2,
                                canonical_suffix,
                                level1 == level2,
                                "morphological",
                            )
                        )
                        continue

                # FIX: Only match if term2 has another recognized academic suffix
                elif term2_lower.startswith(f"{base_term} "):
                    # Only match if the remaining part is another valid academic suffix
                    remaining_part = term2_lower[len(base_term) :].strip()

                    # Check if the remaining part is a valid academic suffix
                    is_valid_suffix = False
                    for other_suffix in all_suffix_forms:
                        if (
                            remaining_part == other_suffix
                            or remaining_part == f"{other_suffix}"
                        ):
                            is_valid_suffix = True
                            break

                    if is_valid_suffix:
                        academic_suffix_pairs.append(
                            (term1, term2, canonical_suffix, level1 == level2)
                        )

        # Check if term1 is a base term that others have suffixes for
        else:
            # First, check if this term has a morphological variant (e.g., politics -> political)
            if term1_lower in MORPHOLOGICAL_VARIANTS:
                adjectival_form = MORPHOLOGICAL_VARIANTS[term1_lower]

                # Look for terms using the adjectival form with an academic suffix
                for term2 in all_terms:
                    if term1 == term2:
                        continue

                    level2 = term_to_level.get(term2, float("inf"))
                    is_cross_level = level1 != level2
                    term2_lower = term2.lower()

                    # Check if term2 starts with the adjectival form and ends with an academic suffix
                    for suffix in all_suffix_forms:
                        # Exact match pattern: the term must be exactly "{adjectival_form} {suffix}"
                        # Not just starting with the adjectival form, to avoid cases like "medical laboratory science"
                        if term2_lower == f"{adjectival_form} {suffix}":
                            # Get the canonical form of the suffix
                            if (
                                suffix in SUFFIX_VARIATIONS
                                and not suffix in ACADEMIC_SUFFIXES
                            ):
                                canonical_suffix = SUFFIX_VARIATIONS[suffix]
                            else:
                                canonical_suffix = suffix
                            academic_suffix_pairs.append(
                                (
                                    term2,
                                    term1,
                                    canonical_suffix,
                                    level1 == level2,
                                    "morphological",
                                )
                            )

            # Also check the regular case where term1 is a base term
            for term2 in all_terms:
                if term1 == term2:
                    continue

                level2 = term_to_level.get(term2, float("inf"))
                is_cross_level = level1 != level2
                term2_lower = term2.lower()

                # Check if term1 is the adjectival form of a base that term2 extends
                if term1_lower in MORPHOLOGICAL_VARIANTS:
                    base_form = MORPHOLOGICAL_VARIANTS[term1_lower]

                    for suffix in all_suffix_forms:
                        # Exact match pattern: the term must be exactly "{base_form} {suffix}"
                        if term2_lower == f"{base_form} {suffix}":
                            if (
                                suffix in SUFFIX_VARIATIONS
                                and not suffix in ACADEMIC_SUFFIXES
                            ):
                                canonical_suffix = SUFFIX_VARIATIONS[suffix]
                            else:
                                canonical_suffix = suffix
                            academic_suffix_pairs.append(
                                (
                                    term2,
                                    term1,
                                    canonical_suffix,
                                    level1 == level2,
                                    "morphological",
                                )
                            )

                # Check if term2 has an academic suffix and is based on term1
                for suffix in all_suffix_forms:
                    if term2_lower.endswith(f" {suffix}"):
                        # Extract the base term from term2 with proper word boundary
                        base_term2 = term2_lower[
                            : -len(suffix) - 1
                        ].strip()  # -1 for the space

                        # Check direct match
                        if term1_lower == base_term2:
                            # Get the canonical form of the suffix (prefer plural forms)
                            if (
                                suffix in SUFFIX_VARIATIONS
                                and not suffix in ACADEMIC_SUFFIXES
                            ):
                                canonical_suffix = SUFFIX_VARIATIONS[suffix]
                            else:
                                canonical_suffix = suffix
                            academic_suffix_pairs.append(
                                (term2, term1, canonical_suffix, level1 == level2)
                            )
                            continue

                        # Check if term1 is a morphological variant of base_term2
                        # IMPORTANT: Only match if base_term2 is exactly the base form, not a compound term
                        # This prevents matching "medical laboratory" with "medicine" because "medical" is not the whole term
                        if (
                            base_term2 in MORPHOLOGICAL_VARIANTS
                            and term1_lower == MORPHOLOGICAL_VARIANTS[base_term2]
                        ):
                            # Additional check - make sure base_term2 is a single word or a known compound
                            # This prevents matching multi-word terms like "medical laboratory" incorrectly
                            if " " not in base_term2 or base_term2 in all_terms:
                                if (
                                    suffix in SUFFIX_VARIATIONS
                                    and not suffix in ACADEMIC_SUFFIXES
                                ):
                                    canonical_suffix = SUFFIX_VARIATIONS[suffix]
                                else:
                                    canonical_suffix = suffix
                                academic_suffix_pairs.append(
                                    (
                                        term2,
                                        term1,
                                        canonical_suffix,
                                        level1 == level2,
                                        "morphological",
                                    )
                                )

    # Process the pairs and add edges
    for pair in academic_suffix_pairs:
        if len(pair) == 5:
            term_with_suffix, base_term, suffix, same_level, variant_type = pair
            is_morphological = True
        else:
            term_with_suffix, base_term, suffix, same_level = pair
            is_morphological = False

        # Skip if we already have an edge
        if G.has_edge(term_with_suffix, base_term):
            continue

        # MODIFIED: Add edges for both same-level and cross-level terms
        level1 = term_to_level.get(term_with_suffix, float("inf"))
        level2 = term_to_level.get(base_term, float("inf"))
        is_cross_level = level1 != level2

        # Prefer the term with the suffix as canonical
        # For cross-level, mark the relationship but it will be handled during canonical selection
        if is_morphological:
            G.add_edge(
                term_with_suffix,
                base_term,
                relationship_type="rule",
                detection_method="morphological_suffix",
                strength=1.0,
                is_cross_level=is_cross_level,
                reason=f"Morphological variation with suffix: '{suffix}' term preferred",
            )
        else:
            G.add_edge(
                term_with_suffix,
                base_term,
                relationship_type="rule",
                detection_method="academic_suffix",
                strength=1.0,
                is_cross_level=is_cross_level,
                reason=f"Academic suffix variation: '{suffix}' term preferred",
            )

    # Process each term for other variations (plural, spelling, dash)
    for i, term1 in enumerate(all_terms):
        level1 = term_to_level.get(term1, float("inf"))

        # Get all variations of term1
        term1_variations = term_variations[term1]

        # Check for plural/singular variations
        for var in plural_variations[term1]:
            if var in plural_lookup:
                for term2 in plural_lookup[var]:
                    if term1 >= term2:  # Skip if we've already processed this pair
                        continue

                    level2 = term_to_level.get(term2, float("inf"))
                    is_cross_level = level1 != level2

                    # MODIFIED: Allow cross-level connections but track them for later

                    # Determine relationship type based on levels
                    relationship_type = "rule"

                    # Determine which term should be canonical (prefer plural)
                    if term1.endswith("s") and not term2.endswith("s"):
                        G.add_edge(
                            term1,
                            term2,
                            relationship_type=relationship_type,
                            detection_method="plural_form",
                            strength=1.0,
                            is_cross_level=is_cross_level,
                            reason="Plural form preferred",
                        )
                    elif term2.endswith("s") and not term1.endswith("s"):
                        G.add_edge(
                            term2,
                            term1,
                            relationship_type=relationship_type,
                            detection_method="plural_form",
                            strength=1.0,
                            is_cross_level=is_cross_level,
                            reason="Plural form preferred",
                        )

        # Check for spelling variations
        for var in spelling_variations[term1]:
            if var in spelling_lookup:
                for term2 in spelling_lookup[var]:
                    if term1 >= term2:  # Skip if we've already processed this pair
                        continue

                    level2 = term_to_level.get(term2, float("inf"))
                    is_cross_level = level1 != level2

                    # MODIFIED: Allow cross-level connections but track them for later

                    # Determine relationship type based on levels
                    relationship_type = "rule"

                    # Find which spelling variation matched
                    for british, american in SPELLING_VARIATIONS.items():
                        if (british in term1.lower() and american in term2.lower()) or (
                            american in term1.lower() and british in term2.lower()
                        ):
                            # Prefer American spelling as canonical
                            if american in term2.lower():
                                G.add_edge(
                                    term2,
                                    term1,
                                    relationship_type=relationship_type,
                                    detection_method="spelling_variation",
                                    strength=1.0,
                                    is_cross_level=is_cross_level,
                                    reason=f"American spelling preferred: {british} -> {american}",
                                )
                            else:
                                G.add_edge(
                                    term1,
                                    term2,
                                    relationship_type=relationship_type,
                                    detection_method="spelling_variation",
                                    strength=1.0,
                                    is_cross_level=is_cross_level,
                                    reason=f"American spelling preferred: {british} -> {american}",
                                )
                            break

        # Check for dash-space variations
        for var in dash_variations[term1]:
            if var in dash_lookup:
                for term2 in dash_lookup[var]:
                    if term1 >= term2:  # Skip if we've already processed this pair
                        continue

                    level2 = term_to_level.get(term2, float("inf"))
                    is_cross_level = level1 != level2

                    # MODIFIED: Allow cross-level connections but track them for later

                    # Determine relationship type based on levels
                    relationship_type = "rule"

                    # Prefer the form without dashes
                    if "-" in term1 and "-" not in term2:
                        G.add_edge(
                            term2,
                            term1,
                            relationship_type=relationship_type,
                            detection_method="dash_space_variation",
                            strength=1.0,
                            is_cross_level=is_cross_level,
                            reason="Space preferred over dash",
                        )
                    elif "-" in term2 and "-" not in term1:
                        G.add_edge(
                            term1,
                            term2,
                            relationship_type=relationship_type,
                            detection_method="dash_space_variation",
                            strength=1.0,
                            is_cross_level=is_cross_level,
                            reason="Space preferred over dash",
                        )

    logging.info(f"Added {G.number_of_edges() - initial_edge_count} rule-based edges")
    return G


def add_compound_term_edges(G: nx.Graph, terms: List[str], terms_to_process: Optional[List[str]] = None) -> nx.Graph:
    """
    Adds edges for compound terms based on the is_compound_term function.

    Args:
        G: NetworkX graph to add edges to
        terms: List of all terms to check for compounds
        terms_to_process: Optional list of terms to process (if None, process all terms)

    Returns:
        Updated graph with compound relationship edges
    """
    initial_edge_count = G.number_of_edges()
    
    # If terms_to_process is None, process all terms
    if terms_to_process is None:
        terms_to_process = terms

    for term in terms_to_process:
        compound_result = is_compound_term(term, terms)
        if compound_result["is_compound"] and compound_result["should_remove"]:
            # Add edges from this compound term to its atomic components
            for atom in compound_result["atomic_terms"]:
                # Find the actual term that matches this normalized atomic term
                for actual_term in terms:
                    if normalize_text(actual_term) == atom:
                        G.add_edge(
                            term,
                            actual_term,
                            relationship_type="compound",
                            detection_method="compound_term",
                        )
                        break

    logging.info(
        f"Added {G.number_of_edges() - initial_edge_count} compound term edges"
    )
    return G


def normalize_url(url: str) -> str:
    """
    Normalizes a URL by removing trailing slashes and query parameters.

    Args:
        url: URL to normalize

    Returns:
        Normalized URL
    """
    return url.split("?")[0].rstrip("/")


def assess_url_quality(url: str) -> float:
    """
    Assess URL quality based on domain and specificity.
    Returns a quality score between 0.0 and 1.0.
    """
    # Normalize the URL
    url = url.lower()

    # Base score
    score = 0.5

    # Domain quality
    if ".edu/" in url:
        score += 0.3  # Educational institutions are most relevant
    elif ".org/" in url:
        score += 0.2  # Organizations are generally good sources
    elif ".gov/" in url:
        score += 0.2  # Government sites are generally reliable
    elif "wikipedia.org/" in url:
        score += 0.25  # Wikipedia is particularly useful for academic concepts
    elif "scholar.google.com/" in url:
        score += 0.3  # Google Scholar is highly relevant

    # URL specificity - more segments generally means more specific content
    path_depth = url.count("/") - 2  # Subtract for protocol and domain slashes
    if path_depth > 0:
        # Add up to 0.2 for deeper paths
        score += min(0.2, 0.05 * path_depth)

    # Department/course pages are particularly valuable
    if any(
        pattern in url
        for pattern in ["/dept/", "/department/", "/faculty/", "/course/", "/program/"]
    ):
        score += 0.15

    # Avoid certain low-value pages
    if any(
        pattern in url for pattern in ["/search?", "/index.php?", "/tag/", "/category/"]
    ):
        score -= 0.15

    # Cap at 1.0
    return min(1.0, max(0.1, score))


def get_dynamic_threshold(term1: str, term2: str, term_to_level: Dict[str, int]) -> int:
    """
    Get dynamic URL overlap threshold based on term levels.

    Args:
        term1: First term
        term2: Second term
        term_to_level: Dict mapping terms to their levels

    Returns:
        URL overlap threshold to use for this term pair
    """
    level1 = term_to_level.get(term1, float("inf"))
    level2 = term_to_level.get(term2, float("inf"))

    # Check if cross-level and determine relationship
    is_cross_level = level1 != level2

    if is_cross_level:
        # For potential hierarchical relationships (parent-child), we need stronger evidence
        # Higher threshold (6) for lower level terms (level 0-1)
        if level1 <= 1 or level2 <= 1:
            return 6
        # Medium threshold (5) for mid-level terms
        elif level1 <= 2 or level2 <= 2:
            return 5
        # Base threshold (4) for higher level terms
        else:
            return 4
    else:
        # For same-level potential variations, we can use slightly lower thresholds
        if level1 <= 1:  # More conservative for important base disciplines
            return 5
        else:  # More permissive for specialized fields
            return 4


def add_web_based_edges(
    G: nx.Graph,
    terms_by_level: TermsByLevel,
    web_content: WebContent,
    url_overlap_threshold: int = 2,  # Default threshold
    min_relevance_score: float = 0.75,  # Default minimum relevance score
    levels_to_process: Optional[Set[int]] = None,  # Optional set of levels to process
) -> nx.Graph:
    """
    Add edges to the graph based on web content overlap.

    Uses a conservative approach that prioritizes:
    1. Higher URL overlap thresholds
    2. URL quality assessment
    3. Domain diversity
    4. Level-aware relationship detection

    Args:
        G: NetworkX graph to add edges to
        terms_by_level: Dict mapping level numbers to lists of terms
        web_content: Dict mapping terms to their web content
        url_overlap_threshold: Minimum number of overlapping URLs required
        min_relevance_score: Minimum relevance score for web content
        levels_to_process: Optional set of level numbers to process (for incremental processing)

    Returns:
        Updated graph with web-based relationship edges
    """
    # Skip if no web content provided
    if not web_content:
        logging.warning("No web content provided for web-based deduplication")
        return G

    initial_edge_count = G.number_of_edges()

    # Flatten terms_by_level into a single list - only process specified levels if provided
    all_terms = []
    terms_to_process = []
    
    for level, terms in terms_by_level.items():
        all_terms.extend(terms)
        if levels_to_process is None or level in levels_to_process:
            terms_to_process.extend(terms)

    logging.info(f"Processing web-based relationships for {len(terms_to_process)} terms out of {len(all_terms)} total terms")

    # Filter to terms with web content
    terms_with_content = [t for t in all_terms if t in web_content and web_content[t]]
    terms_to_process_with_content = [t for t in terms_to_process if t in web_content and web_content[t]]

    logging.info(
        f"{len(terms_with_content)} out of {len(all_terms)} terms have web content"
    )
    logging.info(
        f"{len(terms_to_process_with_content)} out of {len(terms_to_process)} terms to process have web content"
    )

    if not terms_with_content:
        logging.warning("No terms with web content found")
        return G

    # Create mapping from term to level
    term_to_level = {}
    for level, terms in terms_by_level.items():
        for term in terms:
            term_to_level[term] = level

    # Create term URL and domain mappings
    term_urls = {}
    term_domains = {}
    term_relevant_content = {}
    url_to_terms = defaultdict(set)

    logging.info("Building URL mappings")
    for term in tqdm(terms_with_content, desc="Processing term URLs"):
        # Initialize collections for this term
        urls = set()
        domains = defaultdict(bool)
        relevant_content = []

        # Process each web content entry
        for entry in web_content[term]:
            # Skip entries with low relevance if score is present
            relevance_score = entry.get("relevance_score", 1.0)
            if relevance_score < min_relevance_score:
                continue

            # Get URL and normalize it
            url = entry.get("url", "")
            if not url:
                continue

            normalized_url = normalize_url(url)

            # Add to collections
            urls.add(normalized_url)

            # Extract domain
            domain = normalized_url.split("/")[0]
            domains[domain] = True

            # Track which terms are associated with this URL
            url_to_terms[normalized_url].add(term)

            # Add to relevant content if it passes all filters
            relevant_content.append(
                {
                    "url": normalized_url,
                    "quality": entry.get("score", 1.0),
                    "relevance": relevance_score,
                    "title": entry.get("title", ""),
                    "processed_content": entry.get("processed_content", ""),
                }
            )

        # Store mappings for this term
        term_urls[term] = urls
        term_domains[term] = domains
        term_relevant_content[term] = relevant_content

    # Process terms in parallel to find relationships
    logging.info("Finding URL-based relationships between terms")

    # Process chunks of terms in parallel - only terms_to_process_with_content
    batch_size = 100
    term_batches = [
        terms_to_process_with_content[i : i + batch_size]
        for i in range(0, len(terms_to_process_with_content), batch_size)
    ]

    all_edges = []
    for batch in tqdm(term_batches, desc="Finding web-based relationships"):
        # Process a batch of terms to find relationships
        edges = process_term_chunk_web_based(
            batch,
            term_urls,
            term_domains,
            url_to_terms,
            term_relevant_content,
            url_overlap_threshold,
            term_to_level,
        )
        all_edges.extend(edges)

    # Add edges to graph, respecting level boundaries
    edges_added = 0
    for term1, term2, attributes in all_edges:
        # Skip if already connected
        if G.has_edge(term1, term2):
            continue

        # Get levels for both terms
        level1 = term_to_level.get(term1, float("inf"))
        level2 = term_to_level.get(term2, float("inf"))

        # Check if this is a cross-level relationship
        is_cross_level = level1 != level2

        # Get a dynamic threshold based on term levels
        dynamic_threshold = get_dynamic_threshold(term1, term2, term_to_level)

        # Check if it meets the dynamic threshold
        shared_urls = set(attributes.get("shared_urls", []))

        # Calculate quality scores for URLs
        quality_adjusted_count = sum(assess_url_quality(url) for url in shared_urls)
        edu_urls = sum(1 for url in shared_urls if ".edu" in url)
        
        # Simplified threshold logic
        meets_threshold = (
            len(shared_urls) >= dynamic_threshold or  # Raw count threshold
            quality_adjusted_count >= dynamic_threshold * 1.2 or  # Quality-adjusted threshold
            edu_urls >= max(2, dynamic_threshold * 0.7)  # Educational URLs threshold
        )
        
        # Simplified borderline logic - only check if not already meeting threshold
        is_borderline = False
        if not meets_threshold:
            is_borderline = (
                len(shared_urls) >= dynamic_threshold * 0.6 or  # Close to raw count
                quality_adjusted_count >= dynamic_threshold * 0.8 or  # Close to quality threshold
                edu_urls >= max(1, dynamic_threshold * 0.5)  # Some educational URLs
            )
            
        # Check domain diversity - require URLs from at least 2 different domains
        domain_overlap = set(attributes.get("domain_overlap", []))
        if len(domain_overlap) < 2:
            # Without domain diversity, require stronger evidence
            meets_threshold = meets_threshold and (
                len(shared_urls) >= dynamic_threshold + 1
            )
            # Borderline cases should have at least one domain
            if is_borderline and len(domain_overlap) < 1:
                is_borderline = False
            
        # Check if this is a Wikipedia high relevance match
        wiki_high_relevance = attributes.get("wiki_high_relevance", False)
        needs_llm_verification = attributes.get("needs_llm_verification", False)
        
        # For Wikipedia high relevance, ensure terms are at the same level
        if wiki_high_relevance and is_cross_level:
            wiki_high_relevance = False  # Disable for cross-level terms
        
        # LLM verification for Wikipedia high relevance edges
        if wiki_high_relevance and needs_llm_verification:
            # --- Add Embedding Check Here ---
            embedding_similarity = 0.0
            term1_embedding = G.nodes[term1].get('embedding')
            term2_embedding = G.nodes[term2].get('embedding')
            if term1_embedding is not None and term2_embedding is not None:
                embedding_similarity = calculate_embedding_similarity(
                    term1_embedding, term2_embedding
                )
            # --- End Embedding Check ---

            if embedding_similarity > EMBEDDING_SIMILARITY_THRESHOLD: # Check threshold
                # Get the shared Wikipedia URLs
                shared_wiki_urls = [url for url in shared_urls if 'wikipedia.org' in url]
                
                # Only proceed with LLM verification if there's at least one shared Wikipedia URL
                if shared_wiki_urls:
                    logging.info(f"Verifying Wikipedia high relevance with LLM: '{term1}' - '{term2}'")
                    logging.info(f"Shared Wikipedia URLs: {shared_wiki_urls}")
                    llm_verified = verify_edge_with_llm_sync(
                        term1, 
                        term2, 
                        term_relevant_content,
                        list(shared_urls),
                        provider=None
                    )
                    if llm_verified:
                        logging.info(f"LLM VERIFIED Wikipedia edge: '{term1}' - '{term2}'")
                        meets_threshold = True
                        # Keep wiki_high_relevance=True but add verification flag
                        attributes["llm_verified_wiki"] = True
                    else:
                        # LLM rejected the Wikipedia match
                        wiki_high_relevance = False
                        logging.info(f"LLM REJECTED Wikipedia edge: '{term1}' - '{term2}'")
                else:
                    # No shared Wikipedia URLs, so don't verify with LLM
                    wiki_high_relevance = False
                    logging.info(f"Skipping LLM verification - no shared Wikipedia URLs for: '{term1}' - '{term2}'")
            else:
                # Embedding similarity too low, skip LLM verification for this case
                wiki_high_relevance = False # Override flag if similarity is low
                logging.debug(f"Skipping Wiki LLM check for '{term1}' - '{term2}' due to low embedding similarity: {embedding_similarity:.4f}")
        
        # Use LLM to verify other borderline cases
        if is_borderline and not meets_threshold and not wiki_high_relevance:
            # Skip LLM verification for cross-level borderline cases (too aggressive)
            if is_cross_level:
                is_borderline = False
            else:
                # --- Add Embedding Check Here ---
                embedding_similarity = 0.0
                term1_embedding = G.nodes[term1].get('embedding')
                term2_embedding = G.nodes[term2].get('embedding')
                if term1_embedding is not None and term2_embedding is not None:
                    embedding_similarity = calculate_embedding_similarity(
                       term1_embedding, term2_embedding
                    )
                # --- End Embedding Check ---

                if embedding_similarity > EMBEDDING_SIMILARITY_THRESHOLD: # Check threshold
                    # Get the shared Wikipedia URLs
                    shared_wiki_urls = [url for url in shared_urls if 'wikipedia.org' in url]

                    # Only proceed with LLM verification if there's at least one shared Wikipedia URL
                    if shared_wiki_urls:
                        logging.info(f"Verifying borderline case with LLM: '{term1}' - '{term2}'")
                        logging.info(f"Borderline evidence: {len(shared_urls)} shared URLs, {quality_adjusted_count:.2f} quality score")
                        llm_verified = verify_edge_with_llm_sync(
                            term1, 
                            term2, 
                            term_relevant_content,
                            list(shared_urls),
                            provider=None
                        )
                        if llm_verified:
                            logging.info(f"LLM VERIFIED borderline edge: '{term1}' - '{term2}'")
                            meets_threshold = True
                            # Add LLM verification as the detection method
                            attributes["detection_method"] = "llm_verified"
                            attributes["is_llm_verified"] = True
                        else:
                            logging.info(f"LLM REJECTED borderline edge: '{term1}' - '{term2}'")
                    else:
                        # No shared Wikipedia URLs, so don't verify with LLM
                        logging.info(f"Skipping LLM verification for borderline case - no shared Wikipedia URLs: '{term1}' - '{term2}'")
                        is_borderline = False
                else:
                    # Embedding similarity too low, skip LLM verification for borderline case
                    is_borderline = False # Do not consider it borderline if similarity is low
                    logging.debug(f"Skipping Borderline LLM check for '{term1}' - '{term2}' due to low embedding similarity: {embedding_similarity:.4f}")

        # Only add edge if it meets threshold or is a Wikipedia high relevance match
        if meets_threshold or wiki_high_relevance:
            # Calculate relationship strength based on overlap
            if wiki_high_relevance and attributes.get("llm_verified_wiki", False):
                # LLM-verified Wikipedia high relevance matches get very high strength
                strength = 0.95
                detection_method = "llm_verified_wikipedia"
            elif wiki_high_relevance:
                # Wikipedia high relevance matches without LLM verification (shouldn't happen anymore)
                strength = 0.9
                detection_method = "wikipedia_high_relevance"
            elif attributes.get("is_llm_verified", False) or attributes.get("detection_method") == "llm_verified":
                # LLM verified relationships get high strength
                strength = 0.85
                detection_method = "llm_verified"
            else:
                # Regular calculation for standard matches
                base_strength = min(1.0, len(shared_urls) / (dynamic_threshold * 1.5))
                quality_boost = min(0.3, quality_adjusted_count * 0.1)
                edu_boost = min(0.2, edu_urls * 0.1)
                strength = min(1.0, base_strength + quality_boost + edu_boost)
                detection_method = "url_overlap"

            # Determine relationship type
            relationship_type = "web"

            # Create full attributes
            edge_attrs = {
                **attributes,
                "relationship_type": relationship_type,
                "detection_method": detection_method,
                "is_cross_level": is_cross_level,
                "strength": strength,
                "dynamic_threshold": dynamic_threshold,
                "high_quality_urls": quality_adjusted_count,
                "edu_urls": edu_urls,
                "quality_adjusted_count": round(quality_adjusted_count, 2),
                "shared_url_count": len(shared_urls),
                "wiki_high_relevance": wiki_high_relevance,
            }

            G.add_edge(term1, term2, **edge_attrs)
            edges_added += 1

    logging.info(f"Added {edges_added} web-based edges")
    return G


def extract_domain(url: str) -> str:
    """
    Extract domain from URL without subdomain.
    For example: https://www.example.com/path -> example.com
    """
    try:
        # Use urlparse to break down the URL
        parsed = urlparse(url)
        # Get the netloc (domain with subdomains)
        netloc = parsed.netloc

        # Handle special cases (like IP addresses)
        if not netloc or netloc.replace(".", "").isdigit():
            return netloc

        # Split by dots and extract domain
        parts = netloc.split(".")

        # Handle special cases like co.uk
        if len(parts) > 2 and parts[-2] in [
            "co",
            "com",
            "org",
            "net",
            "edu",
            "gov",
            "ac",
        ]:
            return ".".join(parts[-3:])

        # Return domain and TLD
        return ".".join(parts[-2:]) if len(parts) > 1 else netloc
    except Exception:
        # In case of parsing errors, return original
        return url


def process_term_chunk_web_based(
    chunk_terms: List[str],
    term_urls: Dict[str, Set[str]],
    term_domains: Dict[str, Dict[str, bool]],
    url_to_terms: Dict[str, Set[str]],
    term_relevant_content: Dict[str, List[Dict[str, Any]]],
    url_overlap_threshold: int,
    term_to_level: Dict[str, int] = None,  # Add term_to_level parameter
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Process a chunk of terms to find web content overlap.

    Uses a more sophisticated approach that considers:
    1. URL overlap quantity and quality
    2. Domain diversity
    3. Content similarity

    Args:
        chunk_terms: List of terms to process
        term_urls: Dict mapping terms to their URLs
        term_domains: Dict mapping terms to their domain presence
        url_to_terms: Dict mapping URLs to terms that use them
        term_relevant_content: Dict mapping terms to their relevant content
        url_overlap_threshold: Minimum number of overlapping URLs required
        term_to_level: Dict mapping terms to their hierarchical level, used to ensure
                      Wikipedia high relevance edges are only created between same-level terms

    Returns:
        List of (term1, term2, attributes) tuples
    """
    edges = []

    # Find pairs of terms with URL overlap
    for i, term1 in enumerate(chunk_terms):
        urls1 = term_urls.get(term1, set())
        if not urls1:
            continue

        domains1 = term_domains.get(term1, {})

        # Get all terms that share at least one URL with term1
        potential_related_terms = set()
        for url in urls1:
            potential_related_terms.update(url_to_terms[url])

        # Process each potential related term
        for term2 in potential_related_terms:
            debug = set([term1, term2]) == set(["black studies", "africana studies"])
            # Skip self and already processed pairs
            if term1 >= term2:
                continue

            # Get URLs for term2
            urls2 = term_urls.get(term2, set())
            if not urls2:
                continue

            # Find shared URLs
            shared_urls = urls1 & urls2
            shared_wikis = set()
            wiki_high_relevance = False
            
            for url in shared_urls:
                if "wikipedia.org" in url:
                    shared_wikis.add(url)
                    
                    # Check term1's relevance for this Wikipedia URL
                    term1_has_high_relevance = False
                    for entry in term_relevant_content.get(term1, []):
                        if entry.get("url") == url and entry.get("relevance", 0) > 0.6:  # Changed from 0.8 to 0.6
                            term1_has_high_relevance = True
                            break
                    
                    # Check term2's relevance for this Wikipedia URL
                    term2_has_high_relevance = False
                    for entry in term_relevant_content.get(term2, []):
                        if entry.get("url") == url and entry.get("relevance", 0) > 0.6:  # Changed from 0.8 to 0.6
                            term2_has_high_relevance = True
                            break
                    
                    # If both terms have high relevance for this Wikipedia URL
                    wiki_high_relevance = term1_has_high_relevance and term2_has_high_relevance
                    
                    # Debug for specific term pair
                    if debug:
                        print(f"URL: {url}")
                        print(f"Term1 high relevance: {term1_has_high_relevance}")
                        print(f"Term2 high relevance: {term2_has_high_relevance}")
                        print(f"Wiki high relevance: {wiki_high_relevance}")

            if debug:
                print(term1, term2, shared_wikis)
            
            # Special case: Add edge if terms share a Wikipedia link with high relevance
            # But only if they're at the same level
            if shared_wikis and wiki_high_relevance:
                # Only process Wikipedia rule for terms at the same level
                if term_to_level is not None:
                    level1 = term_to_level.get(term1, float('inf'))
                    level2 = term_to_level.get(term2, float('inf'))
                    is_same_level = level1 == level2
                    if not is_same_level:
                        # Skip Wikipedia high relevance rule for cross-level terms
                        pass
                    else:
                        # Mark for LLM verification instead of directly adding an edge
                        wiki_url = next(iter(shared_wikis))
                        edges.append(
                            (
                                term1,
                                term2,
                                {
                                    "shared_urls": list(shared_urls),
                                    "domain_overlap": [extract_domain(wiki_url)],
                                    "high_quality_urls": 1,
                                    "edu_urls": 0,
                                    "content_similarity": 1.0,
                                    "quality_adjusted_count": 1.0, 
                                    "url_quality_scores": {wiki_url: 1.0},
                                    "wiki_high_relevance": True,
                                    "is_same_level": True,
                                    "needs_llm_verification": True,  # Mark for LLM verification
                                },
                            )
                        )
                        continue

            # More conservative: require at least 2 shared URLs as a minimum filter
            if len(shared_urls) < 2:
                continue

            # Early filtering: if it's unlikely to meet threshold after quality adjustments
            # Use a more conservative pre-filtering
            if len(shared_urls) < url_overlap_threshold * 0.6:  # Increased from 0.5
                continue
                
            # Calculate domain diversity
            domains2 = term_domains.get(term2, {})
            domain_overlap = set(domains1.keys()) & set(domains2.keys())

            # More conservative: require stronger domain diversity
            if len(domain_overlap) < 2:
                # Unless we have very strong overlap
                if len(shared_urls) < url_overlap_threshold + 1:  # Increased by 1
                    continue

            # Calculate high-quality and educational URLs
            high_quality_urls = 0
            edu_urls = 0

            # Track quality metrics for each shared URL
            url_quality_scores = {}

            for url in shared_urls:
                # Check if it's a high-quality URL (from a respected domain)
                is_high_quality = (
                    domains1.get(extract_domain(url), {}).get("quality", 0) >= 0.4
                )  # Increased from 0.3

                # Check if it's an educational URL (.edu domain)
                is_edu = ".edu" in url

                if is_high_quality:
                    high_quality_urls += 1
                if is_edu:
                    edu_urls += 1

                # Store quality score for this URL
                url_quality_scores[url] = (
                    1.0 if is_edu else (0.8 if is_high_quality else 0.5)
                )

            # Calculate content similarity if available
            content_similarity = 0.0
            matching_snippets = 0

            content1 = term_relevant_content.get(term1, [])
            content2 = term_relevant_content.get(term2, [])

            # Compare snippets from shared URLs if content is available
            if content1 and content2:
                for entry1 in content1:
                    url1 = entry1.get("url", "")
                    if url1 not in shared_urls:
                        continue

                    relevance1 = entry1.get("relevance", 0)
                    # Skip low relevance content
                    if relevance1 < 0.3:  # More conservative relevance threshold
                        continue

                    snippet1 = entry1.get("processed_content", "").lower()
                    if not snippet1:
                        continue

                    for entry2 in content2:
                        url2 = entry2.get("url", "")
                        if url1 != url2:
                            continue

                        relevance2 = entry2.get("relevance", 0)
                        # Both entries need good relevance
                        if relevance2 < 0.3:
                            continue

                        snippet2 = entry2.get("processed_content", "").lower()
                        if not snippet2:
                            continue

                        # Enhanced content similarity check
                        word_overlap = sum(
                            1 for word in snippet1.split() if word in snippet2.split()
                        )
                        if (
                            snippet1 in snippet2
                            or snippet2 in snippet1
                            or word_overlap
                            >= min(8, max(5, len(snippet1.split()) * 0.3))
                        ):
                            matching_snippets += 1

                            # Boost URL quality score for URLs with matching content
                            url_quality_scores[url1] = min(
                                1.0, url_quality_scores.get(url1, 0.5) + 0.2
                            )
                            break

                # Calculate similarity as ratio of matching snippets to shared URLs
                if shared_urls:
                    content_similarity = matching_snippets / len(shared_urls)
            
            # Calculate quality-adjusted count (giving more weight to high-quality URLs)
            quality_adjusted_count = sum(
                url_quality_scores.get(url, 0.5) for url in shared_urls
            )
            
            # Add "is_llm_verified" field to attributes to handle LLM verification later
            edges.append(
                (
                    term1,
                    term2,
                    {
                        "shared_urls": list(shared_urls),
                        "domain_overlap": list(domain_overlap),
                        "high_quality_urls": high_quality_urls,
                        "edu_urls": edu_urls,
                        "content_similarity": round(content_similarity, 2),
                        "quality_adjusted_count": round(quality_adjusted_count, 2),
                        "url_quality_scores": {
                            url: round(score, 2)
                            for url, score in url_quality_scores.items()
                        },
                        "is_llm_verified": False,  # Will be set to True if verified by LLM later
                    },
                )
            )

    return edges


def add_level_weights_to_edges(G: nx.Graph, terms_by_level: TermsByLevel) -> nx.Graph:
    """
    Adds level information to edges to help with canonical selection.
    Higher priority levels (lower numbers) get higher weights.

    Args:
        G: NetworkX graph to add edge weights to
        terms_by_level: Dict mapping level numbers to lists of terms

    Returns:
        Updated graph with level priority information on edges
    """
    # Create reverse mapping from term to level
    term_to_level = {}
    for level, terms in terms_by_level.items():
        for term in terms:
            term_to_level[term] = level

    # Add level information to each edge
    for term1, term2 in G.edges():
        level1 = term_to_level.get(term1, float("inf"))
        level2 = term_to_level.get(term2, float("inf"))

        # Higher priority (lower level number) gets higher weight
        if level1 < level2:
            G.edges[term1, term2]["level_priority"] = "term1"
        elif level2 < level1:
            G.edges[term1, term2]["level_priority"] = "term2"
        else:
            G.edges[term1, term2]["level_priority"] = "equal"

    return G


def select_canonical_terms(
    G: nx.Graph, terms_by_level: TermsByLevel
) -> CanonicalMapping:
    """
    Select canonical terms for each connected component in the graph.

    Uses a clear priority scheme for selecting canonical terms:
    1. Level (prefer lowest level first)
    2. Connectivity (prefer terms with most connections)
    3. Web content (prefer terms with most web content)
    4. Length (prefer shorter terms)

    Args:
        G: NetworkX graph with terms and relationships
        terms_by_level: Dict mapping level numbers to lists of terms

    Returns:
        Dict mapping canonical terms to their variations
    """
    # Find connected components
    connected_components = list(nx.connected_components(G))
    logging.info(f"Found {len(connected_components)} connected components")

    # Map terms to their level
    term_to_level = {}
    for level, terms in terms_by_level.items():
        for term in terms:
            # If a term appears in multiple levels, prefer the lowest level
            if term in term_to_level:
                term_to_level[term] = min(term_to_level[term], level)
            else:
                term_to_level[term] = level

    canonical_mapping = {}

    # Process each component
    for component in connected_components:
        # Skip singleton components
        if len(component) == 1:
            term = next(iter(component))
            canonical_mapping[term] = set()
            continue

        # Get terms in this component
        component_terms = list(component)

        # Score each term based on priority criteria
        term_scores = []
        for term in component_terms:
            # MODIFIED: Always consider the level, with lower levels preferred
            level = term_to_level.get(term, float("inf"))

            # Get connectivity (number of edges)
            connectivity = len(list(G.neighbors(term)))

            # Get URL count from web content
            url_count = 0
            for neighbor in G.neighbors(term):
                edge_data = G.edges[term, neighbor]
                if edge_data.get("detection_method") == "url_overlap":
                    shared_urls = edge_data.get("shared_urls", [])
                    url_count += len(shared_urls)

            # Get term length (shorter is better)
            term_length = len(term)

            # Create score tuple (level, connectivity, url_count, -term_length)
            # Negative term_length because we want shorter terms to score higher
            term_scores.append((term, (level, -connectivity, -url_count, term_length)))

        # Sort by score tuple (level, connectivity, url_count, term_length)
        # This implements a clear, multi-criteria priority scheme
        sorted_terms = [term for term, score in sorted(term_scores, key=lambda x: x[1])]

        # The first term is our canonical term
        canonical_term = sorted_terms[0]
        variations = set(sorted_terms[1:])

        # Add to canonical mapping
        canonical_mapping[canonical_term] = variations

    return canonical_mapping


def process_term_pair_chunk_transitive(
    chunk: List[Tuple[Tuple[str, str], Set[str]]],
    term_info: Dict[str, Dict[str, Any]],
    G: nx.Graph,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Process a chunk of term pairs for transitive relationship detection.

    Uses a more conservative approach to detect transitive relationships:
    1. Requires multiple high-quality common neighbors
    2. Common neighbors must have strong connections
    3. Terms must have semantic similarity
    4. No cross-field connections (e.g., avoid connecting "education" with "engineering")

    Args:
        chunk: List of ((term1, term2), common_neighbors) tuples to process
        term_info: Dict mapping terms to their normalized info
        G: NetworkX graph with existing edges

    Returns:
        List of edges to add (term1, term2, attributes)
    """
    edges = []
    for (term1, term2), common_neighbors in chunk:
        # Skip if terms are identical or already connected
        if term1 == term2 or G.has_edge(term1, term2):
            continue

        info1 = term_info[term1]
        info2 = term_info[term2]

        # First check if the terms themselves appear to be related
        term_similarity = calculate_term_similarity(info1, info2)
        if term_similarity < 0.25:  # Increased threshold for more conservative matching
            continue

        # Calculate path quality scores
        path_scores = []
        relationship_types = set()
        detection_methods = set()

        # Track high-quality paths
        strong_paths = 0
        rule_based_paths = 0

        for neighbor in common_neighbors:
            # Get edge data for both connections
            edge1 = G.edges[term1, neighbor]
            edge2 = G.edges[neighbor, term2]

            # Skip if both edges are web-based with weak evidence
            if (
                edge1.get("detection_method") == "url_overlap"
                and edge2.get("detection_method") == "url_overlap"
            ):
                # Check the overlap strength
                shared_urls1 = len(edge1.get("shared_urls", []))
                shared_urls2 = len(edge2.get("shared_urls", []))
                if shared_urls1 < 3 or shared_urls2 < 3:
                    continue

            # Track relationship types and detection methods
            rel_type1 = edge1.get("relationship_type", "unknown")
            rel_type2 = edge2.get("relationship_type", "unknown")
            relationship_types.update([rel_type1, rel_type2])

            method1 = edge1.get("detection_method", "unknown")
            method2 = edge2.get("detection_method", "unknown")
            detection_methods.update([method1, method2])

            # Count rule-based paths (more reliable)
            if "rule" in [rel_type1, rel_type2] or "academic_suffix" in [
                method1,
                method2,
            ]:
                rule_based_paths += 1

            # Calculate path strength based on edge strengths
            path_strength = min(edge1.get("strength", 0.5), edge2.get("strength", 0.5))

            # Strong paths need high confidence
            if path_strength >= 0.8:
                strong_paths += 1

            # Only consider paths with reasonable strength
            if path_strength >= 0.6:
                path_scores.append(path_strength)

        if not path_scores:
            continue

        # Calculate aggregate path quality
        avg_path_quality = sum(path_scores) / len(path_scores)
        max_path_quality = max(path_scores)

        # Use more conservative criteria for adding transitive edges
        # 1. Need higher term similarity
        # 2. Need multiple strong paths or very high-quality path
        # 3. Require higher average path quality
        should_add_edge = False
        edge_strength = 0.0
        reason = ""

        # Case 1: Strong linguistic relationship with good evidence
        if term_similarity >= 0.6 and (strong_paths >= 1 or rule_based_paths >= 1):
            should_add_edge = True
            edge_strength = 0.7
            reason = "Strong linguistic similarity with reliable path"

        # Case 2: Multiple strong paths (more evidence-based)
        elif strong_paths >= 3 and avg_path_quality >= 0.7:
            should_add_edge = True
            edge_strength = 0.65
            reason = "Multiple strong connecting paths"

        # Case 3: Rule-based evidence (most reliable)
        elif rule_based_paths >= 2 and term_similarity >= 0.3:
            should_add_edge = True
            edge_strength = 0.75
            reason = "Multiple rule-based connecting paths"

        if should_add_edge:
            # Determine primary relationship type
            primary_type = "mixed"
            if "web" in relationship_types and len(relationship_types) == 1:
                primary_type = "web"
            elif "rule" in relationship_types and len(relationship_types) == 1:
                primary_type = "rule"

            edges.append(
                (
                    term1,
                    term2,
                    {
                        "relationship_type": primary_type,
                        "detection_method": "transitive",
                        "common_neighbors": list(common_neighbors),
                        "path_quality": round(avg_path_quality, 2),
                        "term_similarity": round(term_similarity, 2),
                        "strength": round(edge_strength, 2),
                        "reason": reason,
                        "evidence": {
                            "path_scores": [round(s, 2) for s in path_scores],
                            "relationship_types": list(relationship_types),
                            "detection_methods": list(detection_methods),
                            "strong_paths": strong_paths,
                            "rule_based_paths": rule_based_paths,
                        },
                    },
                )
            )

    return edges


def calculate_term_similarity(info1: Dict[str, Any], info2: Dict[str, Any]) -> float:
    """
    Calculate semantic similarity between two terms using their normalized forms.

    Uses multiple similarity metrics:
    1. Word overlap ratio
    2. Character n-gram similarity
    3. Semantic field similarity

    Args:
        info1: Normalized info for first term
        info2: Normalized info for second term

    Returns:
        Similarity score between 0 and 1
    """
    # Get normalized terms and word sets
    norm1 = info1["normalized"]
    norm2 = info2["normalized"]
    words1 = info1["words"]
    words2 = info2["words"]

    # 1. Calculate word overlap ratio
    total_words = len(words1 | words2)
    common_words = len(words1 & words2)
    word_overlap = common_words / total_words if total_words > 0 else 0

    # 2. Calculate character n-gram similarity
    def get_ngrams(text: str, n: int) -> Set[str]:
        return {text[i : i + n] for i in range(len(text) - n + 1)}

    trigrams1 = get_ngrams(norm1, 3)
    trigrams2 = get_ngrams(norm2, 3)
    total_trigrams = len(trigrams1 | trigrams2)
    common_trigrams = len(trigrams1 & trigrams2)
    trigram_sim = common_trigrams / total_trigrams if total_trigrams > 0 else 0

    # 3. Calculate semantic field similarity
    field_sim = 0.0
    if (
        (info1.get("arts") and info2.get("arts"))
        or (info1.get("science") and info2.get("science"))
        or (info1.get("social") and info2.get("social"))
    ):
        field_sim = 1.0
    elif (info1.get("social") and info2.get("science")) or (
        info1.get("science") and info2.get("social")
    ):
        field_sim = 0.5

    # Combine scores with weights
    similarity = (
        0.4 * word_overlap  # Word overlap is most important
        + 0.3 * trigram_sim  # Character-level similarity
        + 0.3 * field_sim  # Semantic field similarity
    )

    return similarity


def find_transitive_relationships(
    G: nx.Graph, terms: List[str], terms_to_check: Optional[Set[str]] = None
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Find terms that are connected via transitive relationships.

    Uses a more sophisticated approach to find and validate transitive relationships:
    1. Pre-computes term information for efficient processing
    2. Finds potential transitive paths through common neighbors
    3. Evaluates path quality and term similarity
    4. Uses parallel processing for efficiency

    Args:
        G: NetworkX graph with terms and relationships
        terms: List of terms to check for transitive relationships
        terms_to_check: Optional set of terms to specifically check (for incremental processing)

    Returns:
        List of edges to add (term1, term2, attributes)
    """
    # Pre-compute term info in the main process
    logging.info("Pre-computing term info in main process")
    term_info = {}
    for term in terms:
        term_lower = term.lower()
        term_words = set(term_lower.split())

        # Basic term info
        info = {
            "normalized": term_lower,
            "words": term_words,
        }

        # Add semantic field information
        for field, keywords in [
            (
                "arts",
                {
                    "art",
                    "arts",
                    "creative",
                    "design",
                    "music",
                    "visual",
                    "performance",
                    "theater",
                    "theatre",
                    "media",
                    "film",
                    "literature",
                    "humanities",
                },
            ),
            (
                "science",
                {
                    "science",
                    "sciences",
                    "engineering",
                    "technology",
                    "mathematics",
                    "biology",
                    "chemistry",
                    "physics",
                    "medicine",
                    "health",
                },
            ),
            (
                "social",
                {
                    "social",
                    "economics",
                    "sociology",
                    "psychology",
                    "anthropology",
                    "education",
                    "communication",
                    "political",
                    "geography",
                },
            ),
        ]:
            info[field] = any(kw in term_lower for kw in keywords)

        term_info[term] = info

    # Find potential transitive relationships through common neighbors
    logging.info("Computing common neighbors")
    common_neighbors_dict = {}
    
    # If terms_to_check is provided, only check those terms for transitive relationships
    # Otherwise, check all terms
    if terms_to_check:
        # For incremental processing, we need to check:
        # 1. New terms with each other
        # 2. New terms with existing terms
        term_pairs = []
        for term1 in terms:
            for term2 in terms:
                if term1 >= term2:  # Skip if already processed or same term
                    continue
                # Only include pairs where at least one term is in terms_to_check
                if term1 in terms_to_check or term2 in terms_to_check:
                    term_pairs.append((term1, term2))
        
        logging.info(f"Checking {len(term_pairs)} term pairs for transitive relationships (incremental)")
    else:
        # For full processing, check all term pairs
        term_pairs = [(term1, term2) for term1 in terms for term2 in terms if term1 < term2]
        logging.info(f"Checking {len(term_pairs)} term pairs for transitive relationships (full)")
    
    # Find common neighbors for each pair
    for term1, term2 in term_pairs:
        neighbors1 = set(G.neighbors(term1))
        neighbors2 = set(G.neighbors(term2))
        common = neighbors1 & neighbors2
        if common:
            common_neighbors_dict[(term1, term2)] = common

    # Process in parallel
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    # Split into chunks for parallel processing
    items = list(common_neighbors_dict.items())
    chunk_size = max(100, len(items) // (multiprocessing.cpu_count() * 2))
    chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    logging.info(f"Processing {len(chunks)} chunks in parallel")

    added_edges = []
    with ProcessPoolExecutor() as executor:
        futures = []
        for chunk in chunks:
            future = executor.submit(
                process_term_pair_chunk_transitive, chunk, term_info, G
            )
            futures.append(future)

        for future in futures:
            try:
                edges = future.result()
                added_edges.extend(edges)
            except Exception as e:
                logging.error(f"Error processing chunk: {e}")

    logging.info(f"Found {len(added_edges)} transitive edges")
    return added_edges


def convert_to_deduplication_result(
    canonical_mapping: CanonicalMapping, terms_by_level: TermsByLevel, G: nx.Graph
) -> DeduplicationResult:
    """
    Convert canonical mapping to standard deduplication result format.

    Args:
        canonical_mapping: Dict mapping canonical terms to their variations
        terms_by_level: Dict mapping level numbers to lists of terms
        G: The graph used for deduplication

    Returns:
        DeduplicationResult with standard fields
    """
    # Create term to level mapping
    term_to_level = {}
    for level, terms in terms_by_level.items():
        for term in terms:
            # If a term appears in multiple levels, keep only the lowest level
            if term in term_to_level:
                term_to_level[term] = min(term_to_level[term], level)
            else:
                term_to_level[term] = level

    # Get connected components for metadata
    connected_components = list(nx.connected_components(G))
    # Create a mapping from term to component ID
    component_mapping = {}
    for i, component in enumerate(connected_components):
        for term in component:
            component_mapping[term] = i

    # Handle exact duplicates between levels
    # Find exact duplicate terms (case-insensitive) across levels
    # Normalize terms to lowercase for comparison
    lower_to_original = {}
    duplicate_terms = set()

    # First build a mapping of lowercase terms to their original form
    for term in canonical_mapping:
        term_lower = term.lower()
        if term_lower in lower_to_original:
            # This is a duplicate (same term in different case)
            duplicate_terms.add(term_lower)
        else:
            lower_to_original[term_lower] = term

    # For each duplicate, determine which one to keep (the one from the lowest level)
    terms_to_remove = set()
    for duplicate in duplicate_terms:
        # Get all terms that map to this lowercase version
        matching_terms = [
            term for term in canonical_mapping if term.lower() == duplicate
        ]

        # Find the term with the lowest level
        lowest_level = float("inf")
        term_to_keep = None

        for term in matching_terms:
            level = term_to_level.get(term, float("inf"))
            if level < lowest_level:
                lowest_level = level
                term_to_keep = term

        # Mark other terms for removal
        for term in matching_terms:
            if term != term_to_keep:
                terms_to_remove.add(term)
                # Add the term and its variations to the variations of the term we're keeping
                if term_to_keep not in canonical_mapping:
                    canonical_mapping[term_to_keep] = set()
                canonical_mapping[term_to_keep].update(
                    canonical_mapping.get(term, set())
                )
                canonical_mapping[term_to_keep].add(term)

    # Remove the duplicate terms from canonical mapping
    for term in terms_to_remove:
        if term in canonical_mapping:
            canonical_mapping.pop(term)

    # Initialize result with only the required fields
    result = {
        "deduplicated_terms": list(canonical_mapping.keys()),
        "variation_reasons": {},
        "component_details": {},
    }

    # Process variations
    all_variations = set()

    for canonical, variations in canonical_mapping.items():
        # Skip empty variations
        if not variations:
            continue

        # Track all variations
        all_variations.update(variations)

        # Record variation reasons
        for variation in variations:
            if canonical in G and variation in G and G.has_edge(canonical, variation):
                # Edge exists directly between canonical and variation
                edge_data = G.edges[canonical, variation]
                method = edge_data.get("detection_method", "unknown")
                reason = edge_data.get("reason", f"Direct relationship via {method}")
                result["variation_reasons"][variation] = {
                    "canonical": canonical,
                    "reason": reason,
                    "method": method,
                }
            else:
                # For exact duplicates across levels, add a specific reason
                if variation.lower() == canonical.lower():
                    result["variation_reasons"][variation] = {
                        "canonical": canonical,
                        "reason": "Exact duplicate across levels",
                        "method": "exact_match",
                    }
                else:
                    # If no direct edge, find the shortest path in the graph
                    try:
                        path = nx.shortest_path(G, canonical, variation)
                        methods = []
                        reasons = []

                        for i in range(len(path) - 1):
                            edge_data = G.edges[path[i], path[i + 1]]
                            methods.append(edge_data.get("detection_method", "unknown"))
                            reasons.append(
                                edge_data.get("reason", "No reason provided")
                            )

                        result["variation_reasons"][variation] = {
                            "canonical": canonical,
                            "reason": "Indirect relationship via other terms",
                            "path": path,
                            "path_methods": methods,
                            "path_reasons": reasons,
                        }
                    except nx.NetworkXNoPath:
                        # This should not happen if the graph is correctly built
                        result["variation_reasons"][variation] = {
                            "canonical": canonical,
                            "reason": "Unknown relationship (no path found)",
                            "method": "unknown",
                        }

    # Remove variations from deduplicated terms
    result["deduplicated_terms"] = [
        term for term in result["deduplicated_terms"] if term not in all_variations
    ]

    # Add component details
    for i, component in enumerate(connected_components):
        component_list = list(component)
        canonical_in_component = [
            term for term in component_list if term in canonical_mapping
        ]

        # For each canonical term in the component, create a cluster entry
        for canonical_term in canonical_in_component:
            # Get non-canonical members (filter out the canonical term itself)
            members = [term for term in component_list if term != canonical_term]
            # Use canonical term as the key and list members directly
            result["component_details"][canonical_term] = members

    return result


def clean_canonical_mapping(canonical_mapping: CanonicalMapping) -> CanonicalMapping:
    """
    Clean up the canonical mapping to prevent inappropriate groupings, especially
    arts and sciences being grouped together.

    Args:
        canonical_mapping: Dict mapping canonical terms to their variations

    Returns:
        Cleaned canonical mapping
    """
    # Define domain categories
    art_humanities = {
        "creative arts",
        "art",
        "arts",
        "humanities",
        "literature",
        "music",
        "philosophy",
        "theater",
        "theatre",
        "design",
        "media",
        "film",
        "visual arts",
        "performance arts",
    }

    science_fields = {
        "science",
        "sciences",
        "biology",
        "chemistry",
        "physics",
        "mathematics",
        "computer science",
        "engineering",
        "medicine",
        "health",
        "agriculture",
        "environmental science",
        "earth science",
        "technology",
    }

    # Clean each canonical term's variations
    result = {}
    for canonical, variations in canonical_mapping.items():
        canonical_lower = canonical.lower()

        is_canonical_arts = any(art in canonical_lower for art in art_humanities)
        is_canonical_science = any(sci in canonical_lower for sci in science_fields)

        # Filter variations
        filtered_variations = set()
        new_canonical_terms = {}  # Terms that should become their own canonicals

        for variation in variations:
            variation_lower = variation.lower()
            is_variation_arts = any(art in variation_lower for art in art_humanities)
            is_variation_science = any(sci in variation_lower for sci in science_fields)

            # If the canonical and variation are from different domains, make the variation a canonical term
            if (is_canonical_arts and is_variation_science) or (
                is_canonical_science and is_variation_arts
            ):
                new_canonical_terms[variation] = set()
            else:
                filtered_variations.add(variation)

        # Add the filtered variations
        result[canonical] = filtered_variations

        # Add new canonical terms
        for term in new_canonical_terms:
            result[term] = set()

    return result


async def verify_edge_with_llm(
    term1: str, 
    term2: str, 
    term_relevant_content: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    shared_urls: Optional[List[str]] = None,
    provider: Optional[str] = None
) -> bool:
    """
    Use LLM to verify if two academic terms are related enough to create an edge.
    
    Args:
        term1: First academic term
        term2: Second academic term
        term_relevant_content: Dict mapping terms to their relevant content
        shared_urls: List of shared URLs between the terms
        provider: Optional LLM provider
        
    Returns:
        Boolean indicating whether the terms should be connected
    """
    try:
        llm = init_llm(provider)
        
        # Extract Wikipedia snippets for context
        wiki_snippets_term1 = []
        wiki_snippets_term2 = []
        
        shared_wikis = []
        
        for url in shared_urls:
            if "wikipedia.org" in url:
                shared_wikis.append(url)
        
        for term1_url in term_relevant_content[term1]:
            if "wikipedia.org" in term1_url['url']:
                wiki_snippets_term1.append(f"- {term1_url['processed_content']}")
        
        for term2_url in term_relevant_content[term2]:
            if "wikipedia.org" in term2_url['url']:
                wiki_snippets_term2.append(f"- {term2_url['processed_content']}")
            
        # Prepare snippets outside the f-string to avoid backslash issue
        snippet1_str = " ".join(wiki_snippets_term1[:1]).replace("\\n", " ")
        snippet2_str = " ".join(wiki_snippets_term2[:1]).replace("\\n", " ")

        wiki_context = f"""
Here are the Wikipedia snippets for the 2 terms:

{term1}:
    {snippet1_str}

{term2}:
    {snippet2_str}
        """
        
        system_prompt = f"""You are an expert in academic terminology with deep knowledge of academic disciplines.
Your task is to determine if two academic terms represent variations of the same concept or are closely related.
Respond with ONLY "YES" or "NO".

YES = The terms represent the same concept, are variations of each other, or are so closely related they should be considered equivalent.
NO = The terms represent different concepts, distinct fields, or are not closely related enough to be considered the same.

When determining if terms are duplicates, consider:
1. Terms are duplicates if their fields of research are so close they can be used interchangeably in academic literature
2. Terms are duplicates if they represent the same concept with different phrasing or regional variations
3. Terms are duplicates if one is explicitly stated to be another name for the other in authoritative sources
4. Terms are duplicates if they cover essentially the same domain of knowledge with only minor differences in emphasis

The Wikipedia snippets provided are extractive summarizations of shared Wikipedia pages. Use them to inform your decision:
- If the snippets explicitly state that the terms are the same or synonymous, answer YES
- If the snippets show the terms cover the same subject matter with similar scope, answer YES
- If one term is described as a subset or specialized version of the other, answer NO
- If the snippets describe fundamentally different subject areas or approaches, answer NO

Examples of terms that are variations or closely related (YES):
- "Computer Science" and "Computing Science" (same field with different naming conventions)
- "Africana Studies" and "Black Studies" (same interdisciplinary field with different naming)
- "Media Studies" and "Media Science" (same domain with different academic framing)
- "Cybersecurity" and "Computer Security" (explicitly noted as synonymous in literature)
- "Machine Learning" and "Statistical Learning" (used interchangeably in many academic contexts)

Examples of terms that are NOT variations or closely related (NO):
- "Computer Science" and "Computer Engineering" (related but distinct disciplines with different focus)
- "Biomedical Engineering" and "Biological Engineering" (overlapping but distinct specializations)
- "Applied Statistics" and "Statistics" (one is a specialized application of the other)
- "Cognitive Psychology" and "Clinical Psychology" (different branches within psychology)
- "Financial Economics" and "Macroeconomics" (different areas within economics with distinct focuses)
"""
        
        user_prompt = f"""Are the following two academic terms variations of the same concept or closely related enough to be considered equivalent?

Term 1: {term1}
Term 2: {term2}

The Wikipedia snippets provided are extractive summarizations from shared Wikipedia pages. Pay careful attention to these snippets:
- If they explicitly state the terms are synonymous (e.g., "also known as", "also called", "or", terms in parentheses)
- If they describe the same field of study with similar boundaries and applications
- If academic literature would generally treat these terms as interchangeable

---
{wiki_context}

---

Answer with ONLY "YES" or "NO"."""

        # print(user_prompt)
        # assert False
        
        response = llm.infer(prompt=user_prompt, system_prompt=system_prompt)
        
        # Parse the response - look for YES or NO
        response_text = response.text.strip().upper()
        
        # Log the full response
        logging.info(f"LLM verification for '{term1}' and '{term2}': {response_text}")
        
        if "YES" in response_text:
            return True
        elif "NO" in response_text:
            return False
        else:
            # Default to conservative approach (no edge) if response is unclear
            logging.debug(f"Unclear LLM verification response for {term1} and {term2}: {response_text}")
            return False
            
    except Exception as e:
        logging.error(f"Error in LLM edge verification for {term1} and {term2}: {str(e)}")
        return False  # Default to conservative approach on errors

def verify_edge_with_llm_sync(
    term1: str, 
    term2: str, 
    term_relevant_content: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    shared_urls: Optional[List[str]] = None,
    provider: Optional[str] = None
) -> bool:
    """
    Synchronous wrapper for verify_edge_with_llm
    
    Args:
        term1: First academic term
        term2: Second academic term
        term_relevant_content: Dict mapping terms to their relevant content
        shared_urls: List of shared URLs between the terms
        provider: Optional LLM provider
        
    Returns:
        Boolean indicating whether the terms should be connected
    """
    try:
        # Log when the synchronous function is called
        logging.debug(f"Calling LLM verification for '{term1}' and '{term2}'")
        
        # Create a new event loop or use the existing one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If there's no event loop in this thread, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        result = loop.run_until_complete(
            verify_edge_with_llm(
                term1, 
                term2, 
                term_relevant_content, 
                shared_urls, 
                provider
            )
        )
        logging.debug(f"LLM verification result for '{term1}' and '{term2}': {result}")
        return result
    except Exception as e:
        logging.error(f"Error in synchronous LLM edge verification: {str(e)}")
        return False  # Default to conservative approach on errors
