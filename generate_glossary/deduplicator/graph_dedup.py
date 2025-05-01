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
import time
import random
from numpy.random import choice

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


def init_llm(provider: Optional[str] = None, model: Optional[str] = None) -> BaseLLM:
    """
    Initialize LLM with specified provider and model
    
    Args:
        provider: Optional provider name
        model: Optional model name ("default" or "mini")
        
    Returns:
        LLM instance
    """
    if not provider:
        provider = Provider.OPENAI
    
    # Choose appropriate model based on parameter
    if model is None:
        model = "default"
    selected_model = GEMINI_MODELS[model] if provider == Provider.GEMINI else OPENAI_MODELS[model]
        
    return LLMFactory.create_llm(
        provider=provider,
        model=selected_model,
        temperature=0
    )

def get_random_llm_config() -> Tuple[str, str]:
    """Get a random LLM provider and model configuration"""
    provider = choice([Provider.OPENAI, Provider.GEMINI])
    model = choice(["pro", "default"], p=[0.4, 0.6])
    return provider, model

@timing_decorator
def deduplicate_graph_based(
    terms_by_level: Dict[int, List[str]],
    web_content: Optional[Dict[str, List[WebContent]]] = None,
    url_overlap_threshold: int = 2,
    min_relevance_score: float = 0.75,
    cache_dir: Optional[str] = None,
    current_level: Optional[int] = None,
) -> DeduplicationResult:
    """
    Deduplicate terms using a graph-based approach.

    Builds a graph where nodes are terms and edges represent similarity relationships.
    Uses a multi-criteria approach to select canonical terms with clear priority rules.

    Args:
        terms_by_level: Dict mapping level numbers to lists of terms
        web_content: Optional dict mapping terms to lists of WebContent objects
        url_overlap_threshold: Minimum number of shared URLs to consider terms related
        min_relevance_score: Minimum relevance score for content to be considered
        cache_dir: Optional directory to cache embeddings
        current_level: Optional specific level to focus on

    Returns:
        DeduplicationResult with deduplicated terms and variation info
    """
    logging.info("Starting graph-based deduplication")

    # Check for overlapping terms between levels
    all_terms = set()
    overlap_terms = set()
    for level, terms in terms_by_level.items():
        for term in terms:
            if term in all_terms:
                overlap_terms.add(term)
            else:
                all_terms.add(term)
    
    # Initialize the base NetworkX graph
    G = nx.Graph()
    
    # Determine terms to process in this run
    terms_to_process = terms_by_level

    # Figure out current level if not provided explicitly
    if current_level is None:
        current_level = max(terms_by_level.keys())
    
    # Check if we have cached graph data from previous runs
    G = None
    term_embeddings = {}
    cached_levels = set()
    cached_had_web_content = False
    cached_levels_with_web_content = set()  # Track which levels had web content
    
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
            web_content_flag_path = os.path.join(cache_dir, "had_web_content.pickle")
            web_content_levels_path = os.path.join(cache_dir, "levels_with_web_content.pickle")
            
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
                    
                # Load web content flag if it exists
                if os.path.exists(web_content_flag_path):
                    with open(web_content_flag_path, "rb") as f:
                        cached_had_web_content = pickle.load(f)
                
                # Load levels with web content if file exists
                if os.path.exists(web_content_levels_path):
                    with open(web_content_levels_path, "rb") as f:
                        cached_levels_with_web_content = pickle.load(f)
                
                logging.info(f"Loaded cached graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
                logging.info(f"Loaded embeddings for {len(term_embeddings)} terms")
                logging.info(f"Previously processed levels: {cached_levels}")
                logging.info(f"Previous run had web content: {cached_had_web_content}")
                logging.info(f"Previous levels with web content: {cached_levels_with_web_content}")
                
                # Determine which levels have web content in the current run
                current_levels_with_web_content = set()
                if web_content:
                    # Check which levels have terms with web content
                    for level, terms in terms_by_level.items():
                        for term in terms:
                            if term in web_content and web_content[term]:
                                current_levels_with_web_content.add(level)
                                break
                
                logging.info(f"Current levels with web content: {current_levels_with_web_content}")
                
                # Force reprocessing for levels where web content is newly available
                levels_to_reprocess = current_levels_with_web_content - cached_levels_with_web_content
                if levels_to_reprocess:
                    logging.info(f"Web content newly available for levels: {levels_to_reprocess}. Will reprocess these levels.")
                    cached_levels = cached_levels - levels_to_reprocess
                
        except Exception as e:
            logging.error(f"Error loading cached graph data: {e}. Building new graph from scratch.")
            G = None
            term_embeddings = {}
            cached_levels = set()
            cached_had_web_content = False
            cached_levels_with_web_content = set()

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
            logging.info(f"Adding rule-based edges for {sum(len(terms) for terms in terms_to_process.values())} new terms")
            G = add_rule_based_edges(G, terms_to_process)

        # 3. Add compound term edges for new terms
        # Get all terms from all levels for compound term checking
        all_terms = [term for terms in terms_by_level.values() for term in terms]
        # But only process the new terms for compound relationships
        new_terms = [term for level, terms in terms_to_process.items() for term in terms]
        if new_terms:
            logging.info(f"Adding compound term edges for {len(new_terms)} new terms")
            G = add_compound_term_edges(G, all_terms, new_terms)

        # 4. Add web-based relationships if web content is available
        if web_content:
            # Check if we actually have web content for any of the new terms
            new_terms_with_content = [t for t in new_terms if t in web_content and web_content[t]]
            if new_terms_with_content:
                logging.info(f"Found web content for {len(new_terms_with_content)} new terms, adding web-based edges")
                G = add_web_based_edges(
                    G,
                    terms_by_level,
                    web_content,
                    levels_to_process=levels_to_process,
                )
            else:
                logging.info("No web content found for new terms, skipping web-based edge addition")

        # 5. Add level weighting information
        G = add_level_weights_to_edges(G, terms_by_level)

        # 6. Find and add explicit transitive relationships 
        # Only process transitive relationships involving new terms
        new_terms_set = set(new_terms)
        logging.info(f"Finding transitive relationships for {len(new_terms_set)} new terms")
        transitive_edges = find_transitive_relationships(G, all_terms, new_terms_set)
        if transitive_edges:
            logging.info(f"Adding {len(transitive_edges)} transitive edges to graph")
            G.add_edges_from(transitive_edges)
        else:
            logging.info("No transitive edges found")

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
            web_content_flag_path = os.path.join(cache_dir, "had_web_content.pickle")
            web_content_levels_path = os.path.join(cache_dir, "levels_with_web_content.pickle")
            
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
                
            # Save web content flag
            with open(web_content_flag_path, "wb") as f:
                pickle.dump(bool(web_content), f)
                
            # Determine which levels have web content in this run
            current_levels_with_web_content = set()
            if web_content:
                # Check which levels have terms with web content
                for level, terms in terms_by_level.items():
                    for term in terms:
                        if term in web_content and web_content[term]:
                            current_levels_with_web_content.add(level)
                            break
            
            # Save levels with web content
            with open(web_content_levels_path, "wb") as f:
                pickle.dump(current_levels_with_web_content, f)
                
            logging.info(f"Cached graph data saved to {cache_dir}")
            logging.info(f"Saved levels with web content: {current_levels_with_web_content}")
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
    
    # Create dictionaries to quickly look up terms
    all_terms_lower = {term.lower(): term for term in all_terms}
    
    # OPTIMIZATION: Create indexes for fast lookups
    # Index terms by their base forms for fast lookup
    term_base_index = {}
    
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

    # Variations of academic suffixes (singular forms, etc.)
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
        "system": "systems",
        "algorithm": "algorithms",
        "system": "systems",
        "theory": "theories",
        "method": "methods",
        "principle": "principles",
        # These don't change in plural/singular form
        "education": "education",
        "research": "research",
    }

    # All forms combined for efficient iteration
    all_suffix_forms = list(ACADEMIC_SUFFIXES) + list(SUFFIX_VARIATIONS.keys())

    # Define morphological variants for academic terms
    MORPHOLOGICAL_VARIANTS = {
        # Base form -> Adjectival form
        "politics": "political",
        "environment": "environmental",
        "economics": "economic",
        "biology": "biological",
        "chemistry": "chemical",
        "health": "healthcare",
        "health": "health care",
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
        "healthcare": "health",
        "health care": "health",
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

    # British/American spelling variations
    # Note: SPELLING_VARIATIONS is also defined globally, ensure consistency or use one definition
    LOCAL_SPELLING_VARIATIONS = [
        ("re", "er"),  # centre/center
        ("our", "or"),  # colour/color
        ("ence", "ense"),  # defence/defense
        ("ogue", "og"),  # dialogue/dialog
        ("ise", "ize"),  # organise/organize
        ("yse", "yze"),  # analyse/analyze
        ("ll", "l"),  # modelling/modeling
        ("ae", "e"),  # anaemia/anemia
        ("mme", "m"),  # programme/program
    ]

    # --- Refactored Plural/Singular Handling using NLTK Lemmatizer ---
    logging.info("Using NLTK WordNet Lemmatizer for plural/singular detection.")
    try:
        from nltk.corpus import wordnet
        from nltk import pos_tag, word_tokenize # Ensure these are available/imported
        # Download necessary NLTK data if not present (optional, can be done beforehand)
        # nltk.download('punkt', quiet=True)
        # nltk.download('averaged_perceptron_tagger', quiet=True)
        # nltk.download('wordnet', quiet=True)
    except ImportError:
        logging.error("NLTK components (wordnet, pos_tag, word_tokenize) not found. Cannot perform lemmatization.")
        return G # Cannot proceed without NLTK

    def get_wordnet_pos(treebank_tag):
        # Map Treebank POS tags to WordNet POS tags for lemmatizer
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN # Default to noun if tag is unknown

    lemmatizer = nltk.WordNetLemmatizer()
    lemma_to_terms = defaultdict(set)

    logging.info("Generating lemmas for terms...")
    for term in tqdm(all_terms, desc="Lemmatizing"):
        term_lower = term.lower()
        tokens = word_tokenize(term_lower)
        if not tokens:
            continue # Skip empty terms
            
        tagged_tokens = pos_tag(tokens)
        last_word, last_tag = tagged_tokens[-1]
        wn_tag = get_wordnet_pos(last_tag)
        
        # Lemmatize the last word using its POS tag
        lemma = lemmatizer.lemmatize(last_word, pos=wn_tag)

        # Construct the lemma key: use preceding words + lemma of last word
        lemma_key = " ".join(tokens[:-1] + [lemma]) if len(tokens) > 1 else lemma
        
        # Map the constructed lemma key back to the original term
        lemma_to_terms[lemma_key].add(term)

        # Debugging for specific case
        if term_lower == 'infectious disease' or term_lower == 'infectious diseases':
            logging.info(f"LEMMA GEN: Term='{term}', Lemma Key='{lemma_key}', Mapped Terms={lemma_to_terms[lemma_key]}")

    logging.info("Adding edges based on shared lemmas...")
    edges_added_lemma = 0
    for lemma, terms_with_lemma in lemma_to_terms.items():
        if len(terms_with_lemma) > 1:
            # Add edges between all pairs of terms sharing this lemma
            for term1, term2 in itertools.combinations(terms_with_lemma, 2):
                 if not G.has_edge(term1, term2): # Check if edge already exists
                     level1 = term_to_level.get(term1, float("inf"))
                     level2 = term_to_level.get(term2, float("inf"))
                     is_cross_level = level1 != level2
                     reason = f"Share common lemma based on last word: '{lemma}'"
                     # Use a helper function for attributes if available, otherwise define dict directly
                     attrs = {
                         "relationship_type": "rule",
                         "detection_method": "lemma_match",
                         "strength": 1.0,
                         "is_cross_level": is_cross_level,
                         "reason": reason,
                     }
                     G.add_edge(term1, term2, **attrs)
                     edges_added_lemma += 1
                     # Log specific pair addition
                     if set([term1.lower(), term2.lower()]) == set(['infectious diseases', 'infectious disease']):
                         logging.info(f"LEMMA ADD: Added edge ('{term1}', '{term2}') based on lemma '{lemma}'")
                     else:
                         logging.debug(f"Adding lemma edge: '{term1}' <-> '{term2}' (Lemma: '{lemma}')")

    logging.info(f"Added {edges_added_lemma} lemma-based (plural/singular) edges")
    # --- End Refactored Plural/Singular Handling ---
    
    # --- Keep existing logic for other rule types (Suffix, Spelling, Dash) ---

    # Initialize collections for spelling and dash variations
    spelling_variations = defaultdict(set)
    spelling_lookup = defaultdict(set)
    dash_variations = defaultdict(set)
    dash_lookup = defaultdict(set)
    
    # Populate spelling and dash variations (can be done in the same term loop)
    logging.info("Processing spelling and dash variations...")
    for term in tqdm(all_terms, desc="Spelling/Dash Variations"):
        term_lower = term.lower()
        
        # Add spelling variations
        for british, american in LOCAL_SPELLING_VARIATIONS:
            if british in term_lower:
                american_form = term_lower.replace(british, american)
                if american_form in all_terms_lower: # Check if variation exists
                    spelling_variations[term].add(all_terms_lower[american_form])
                    spelling_lookup[all_terms_lower[american_form]].add(term)
            if american in term_lower:
                british_form = term_lower.replace(american, british)
                if british_form in all_terms_lower: # Check if variation exists
                    spelling_variations[term].add(all_terms_lower[british_form])
                    spelling_lookup[all_terms_lower[british_form]].add(term)
        
        # Add dash-space variations
        if "-" in term_lower:
            space_form = term_lower.replace("-", " ")
            if space_form in all_terms_lower: # Check if variation exists
                dash_variations[term].add(all_terms_lower[space_form])
                dash_lookup[all_terms_lower[space_form]].add(term)
        if " " in term_lower:
            dash_form = term_lower.replace(" ", "-")
            if dash_form in all_terms_lower: # Check if variation exists
                dash_variations[term].add(all_terms_lower[dash_form])
                dash_lookup[all_terms_lower[dash_form]].add(term)

    # OPTIMIZATION: Process academic suffix variations using indexing
    logging.info("Processing academic suffix and morphological variations")
    
    # First, index terms by their base forms (to quickly find potential matches)
    # This indexing might need refinement depending on requirements
    for term in all_terms:
        term_lower = term.lower()
        term_base_index[term_lower] = term # Store direct term mapping
        for suffix in all_suffix_forms:
            if term_lower.endswith(f" {suffix}"):
                base = term_lower[:-len(suffix)-1].strip()
                if base:
                    # Store mapping from base to term_with_suffix
                    # This assumes base forms might map to multiple suffixed terms
                    if base not in term_base_index:
                        term_base_index[base] = set()
                    # Ensure we're adding to a set
                    if isinstance(term_base_index[base], str): # If base existed as a direct term
                        term_base_index[base] = {term_base_index[base]} # Convert to set
                    if isinstance(term_base_index[base], set):
                         term_base_index[base].add(term)
    
    # Academic suffix pairs for batch processing
    academic_suffix_pairs = []

    # Process morphological variants more efficiently using the index
    for term1 in all_terms:
        term1_lower = term1.lower()
        level1 = term_to_level.get(term1, float("inf"))
        
        # CASE 1: Is term1 a base form that has morphological variants?
        if term1_lower in MORPHOLOGICAL_VARIANTS:
            adjectival_form = MORPHOLOGICAL_VARIANTS[term1_lower]
            for suffix in all_suffix_forms:
                combined = f"{adjectival_form} {suffix}"
                if combined in all_terms_lower:
                    term2 = all_terms_lower[combined]
                    level2 = term_to_level.get(term2, float("inf"))
                    canonical_suffix = suffix
                    if suffix in SUFFIX_VARIATIONS and suffix not in ACADEMIC_SUFFIXES:
                        canonical_suffix = SUFFIX_VARIATIONS[suffix]
                    academic_suffix_pairs.append(
                        (term2, term1, canonical_suffix, level1 == level2, "morphological")
                    )
        
        # CASE 2: Is term1 an adjectival form? (Requires reverse lookup)
        # This part seems complex and might need MORPHOLOGICAL_VARIANTS to be bidirectional
        # Assuming MORPHOLOGICAL_VARIANTS maps adj -> base as well for simplicity here
        if term1_lower in MORPHOLOGICAL_VARIANTS: # Check if term1 is an adjectival form key
             base_form = MORPHOLOGICAL_VARIANTS[term1_lower] # Get corresponding base
             for suffix in all_suffix_forms:
                 combined = f"{base_form} {suffix}"
                 if combined in all_terms_lower:
                     term2 = all_terms_lower[combined]
                     level2 = term_to_level.get(term2, float("inf"))
                     canonical_suffix = suffix
                     if suffix in SUFFIX_VARIATIONS and suffix not in ACADEMIC_SUFFIXES:
                         canonical_suffix = SUFFIX_VARIATIONS[suffix]
                     academic_suffix_pairs.append(
                         (term2, term1, canonical_suffix, level1 == level2, "morphological")
                     )
        
        # CASE 3: Is term1 a term with an academic suffix?
        for suffix in all_suffix_forms:
            if term1_lower.endswith(f" {suffix}"):
                base_term = term1_lower[:-len(suffix)-1].strip()
                if base_term in all_terms_lower:
                    term2 = all_terms_lower[base_term]
                    level2 = term_to_level.get(term2, float("inf"))
                    canonical_suffix = suffix
                    if suffix in SUFFIX_VARIATIONS and suffix not in ACADEMIC_SUFFIXES:
                        canonical_suffix = SUFFIX_VARIATIONS[suffix]
                    academic_suffix_pairs.append(
                        (term1, term2, canonical_suffix, level1 == level2)
                    )
                # Check if base term has a morphological variant that exists
                if base_term in MORPHOLOGICAL_VARIANTS: # Is base_term a base that has an adjective form?
                    morph_variant_adj = MORPHOLOGICAL_VARIANTS[base_term] # Get the adjective form
                    if morph_variant_adj in all_terms_lower: # Does the adjective form exist as a term?
                        term2 = all_terms_lower[morph_variant_adj] # The existing adj form term
                        level2 = term_to_level.get(term2, float("inf"))
                        canonical_suffix = suffix
                        if suffix in SUFFIX_VARIATIONS and suffix not in ACADEMIC_SUFFIXES:
                            canonical_suffix = SUFFIX_VARIATIONS[suffix]
                        academic_suffix_pairs.append(
                            (term1, term2, canonical_suffix, level1 == level2, "morphological")
                        )

    # --- Process Suffix/Morphological pairs --- 
    logging.info(f"Processing {len(academic_suffix_pairs)} academic suffix/morphological pairs")
    edges_added_suffix = 0
    for pair in academic_suffix_pairs:
        is_morphological = len(pair) == 5
        if is_morphological:
            term_with_suffix, base_term, suffix, _, _ = pair # Unpack carefully
            method = "morphological_suffix"
            reason = f"Morphological variation with suffix: '{suffix}'"
        else:
            term_with_suffix, base_term, suffix, _ = pair
            method = "academic_suffix"
            reason = f"Academic suffix variation: '{suffix}'"

        if not G.has_edge(term_with_suffix, base_term):
            level1 = term_to_level.get(term_with_suffix, float("inf"))
            level2 = term_to_level.get(base_term, float("inf"))
            is_cross_level = level1 != level2
            attrs = {
                "relationship_type": "rule",
                "detection_method": method,
                "strength": 1.0,
                "is_cross_level": is_cross_level,
                "reason": reason,
            }
            G.add_edge(term_with_suffix, base_term, **attrs)
            edges_added_suffix += 1
            logging.debug(f"Adding {method} edge: '{term_with_suffix}' <-> '{base_term}'")
    logging.info(f"Added {edges_added_suffix} academic suffix/morphological edges")

    # --- Process Spelling and Dash variations --- 
    logging.info("Adding edges for spelling and dash variations")
    edges_added_spelling_dash = 0
    
    # Helper function for edge attributes if not defined globally
    def edge_attrs(reason, method, cross_level):
        return {
            "relationship_type": "rule",
            "detection_method": method,
            "strength": 1.0,
            "is_cross_level": cross_level,
            "reason": reason,
        }
        
    processed_pairs_spelling_dash = set() # Avoid duplicate checks

    for term1 in all_terms:
        # Check spelling variations
        for term2 in spelling_variations.get(term1, set()):
            pair = tuple(sorted((term1, term2)))
            if pair in processed_pairs_spelling_dash or G.has_edge(term1, term2):
                continue
            processed_pairs_spelling_dash.add(pair)
            
            level1 = term_to_level.get(term1, float("inf"))
            level2 = term_to_level.get(term2, float("inf"))
            is_cross_level = level1 != level2
            reason = f"Spelling variation detected"
            attrs = edge_attrs(reason, "spelling_variation", is_cross_level)
            G.add_edge(term1, term2, **attrs)
            edges_added_spelling_dash += 1
            logging.debug(f"Adding spelling edge: '{term1}' <-> '{term2}'")

        # Check dash variations
        for term2 in dash_variations.get(term1, set()):
            pair = tuple(sorted((term1, term2)))
            if pair in processed_pairs_spelling_dash or G.has_edge(term1, term2):
                continue
            processed_pairs_spelling_dash.add(pair)

            level1 = term_to_level.get(term1, float("inf"))
            level2 = term_to_level.get(term2, float("inf"))
            is_cross_level = level1 != level2
            reason = f"Dash/space variation detected"
            attrs = edge_attrs(reason, "dash_space_variation", is_cross_level)
            G.add_edge(term1, term2, **attrs)
            edges_added_spelling_dash += 1
            logging.debug(f"Adding dash/space edge: '{term1}' <-> '{term2}'")
            
    logging.info(f"Added {edges_added_spelling_dash} spelling/dash edges")

    total_rule_edges = edges_added_lemma + edges_added_suffix + edges_added_spelling_dash
    logging.info(f"Added {total_rule_edges} total rule-based edges (Lemma: {edges_added_lemma}, Suffix/Morph: {edges_added_suffix}, Spelling/Dash: {edges_added_spelling_dash})")
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

    # OPTIMIZATION: Pre-compute normalized terms for lookup
    logging.info("Computing normalized terms for compound term detection")
    normalized_terms = {normalize_text(term): term for term in terms}
    
    # Create lookup sets for fast membership checks
    terms_set = set(terms)
    normalized_terms_set = set(normalized_terms.keys())
    
    # OPTIMIZATION: Batch process terms
    logging.info(f"Processing {len(terms_to_process)} terms for compound relationships")
    
    # Process in batches for logging progress
    batch_size = 1000
    term_batches = [terms_to_process[i:i+batch_size] for i in range(0, len(terms_to_process), batch_size)]
    
    edges_to_add = []
    total_compounds = 0
    
    # Get the typical length of terms to help with filtering
    term_lengths = [len(term.split()) for term in terms]
    avg_term_length = sum(term_lengths) / len(term_lengths) if term_lengths else 3
    max_common_length = max(2, int(avg_term_length * 0.7))  # Cap the common term length

    for batch_idx, batch in enumerate(term_batches):
        batch_compounds = 0
        logging.info(f"Processing compound term batch {batch_idx+1}/{len(term_batches)} ({len(batch)} terms)")
        
        for term in batch:
            # Skip terms not in the graph (should not happen, but just in case)
            if term not in G:
                continue
                
            # Normalize the term
            term_norm = normalize_text(term)
            
            # Check if this is a compound term with "and" or commas
            if ' and ' not in term_norm and ',' not in term_norm:
                continue
                
            # Extract parts from the compound term
            parts = []
            for and_part in term_norm.split(' and '):
                parts.extend(part.strip() for part in and_part.split(','))
                
            # Clean up parts and remove empty ones
            atomic_terms_normalized = [part.strip() for part in parts if part.strip()]
            
            # Skip if no atomic terms found
            if not atomic_terms_normalized:
                continue
            
            # IMPROVED LOGIC: Filter out potential atomic terms that are too short or general
            filtered_atomic_terms = []
            for atom in atomic_terms_normalized:
                # Skip very short terms (likely not meaningful on their own)
                if len(atom.split()) < 2 and len(atomic_terms_normalized) > 1:
                    # For single-word terms, only include if they're relatively uncommon words
                    # This helps avoid matching general terms like "management", "science", etc.
                    if atom in ["management", "science", "studies", "analysis", "research", 
                                "education", "technology", "engineering", "arts", "design"]:
                        continue
                
                # Skip very short substrings of the original compound
                if len(atom.split()) < max_common_length and len(term_norm.split()) > 5:
                    # Only include it if it appears as a standalone term
                    if atom in normalized_terms_set:
                        filtered_atomic_terms.append(atom)
                else:
                    filtered_atomic_terms.append(atom)
            
            # Skip if no valid atomic terms remain
            if not filtered_atomic_terms:
                continue
                
            # Check which atomic terms appear in the term list
            found_terms_normalized = [atom for atom in filtered_atomic_terms if atom in normalized_terms_set]
            
            # IMPROVED LOGIC: Only consider it a valid compound if most atomic terms are found
            # This helps prevent incorrect matching of general terms
            if len(found_terms_normalized) < len(filtered_atomic_terms) * 0.5 or not found_terms_normalized:
                continue
            
            # IMPROVED LOGIC: For terms with "and", ensure both sides of the "and" are represented
            # This prevents cases like "risk analysis and management" -> "management"
            if ' and ' in term_norm:
                left_side = term_norm.split(' and ')[0].strip()
                right_side = term_norm.split(' and ')[1].strip()
                
                # Check if at least one term from each side is represented
                left_represented = any(atom in left_side for atom in found_terms_normalized)
                right_represented = any(atom in right_side for atom in found_terms_normalized)
                
                # Skip if one side is completely unrepresented
                if not (left_represented and right_represented) and len(found_terms_normalized) < 2:
                    continue
                
            # Add edges from this compound term to its atomic components
            # OPTIMIZATION: Use normalized_terms dictionary to find the original form
            for atom_norm in found_terms_normalized:
                actual_term = normalized_terms[atom_norm]
                # Add edge
                edges_to_add.append((
                    term,
                    actual_term,
                    {
                        "relationship_type": "compound",
                        "detection_method": "compound_term",
                        "strength": 0.9,
                        "reason": f"Contains term as a component: '{actual_term}'"
                    }
                ))
            
            # Increment counter
            batch_compounds += 1
            
        total_compounds += batch_compounds
        logging.info(f"Found {batch_compounds} compound terms in batch {batch_idx+1}")
    
    # Add all edges at once
    edges_added_count = 0 # Track actual edges added
    if edges_to_add:
        logging.info(f"Processing {len(edges_to_add)} potential compound term edges to graph")
        for term, actual_term, attrs in edges_to_add:
             if not G.has_edge(term, actual_term): # Check if edge already exists
                G.add_edge(term, actual_term, **attrs)
                edges_added_count += 1
        # G.add_edges_from(edges_to_add) # Old way - replaced with checked loop

    # logging.info(f"Processed {total_compounds} compound terms, added {G.number_of_edges() - initial_edge_count} edges") # Old log
    logging.info(f"Processed {total_compounds} potential compound terms, added {edges_added_count} new compound edges") # Updated log
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
    levels_to_process: Optional[Set[int]] = None,  # Optional set of levels to process
) -> nx.Graph:
    """
    Add edges to the graph based on web content overlap and LLM verification.

    New simplified logic:
    1. If two terms share > 50% of their respective URLs, add a direct edge.
    2. If terms share >= 1 URL but not > 50%, use LLM to verify the relationship based on shared content snippets.

    Args:
        G: NetworkX graph to add edges to
        terms_by_level: Dict mapping level numbers to lists of terms
        web_content: Dict mapping terms to their web content
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
    terms_to_process_set = set()
    
    for level, terms in terms_by_level.items():
        all_terms.extend(terms)
        if levels_to_process is None or level in levels_to_process:
            terms_to_process_set.update(terms)

    # Filter to terms with web content for efficient processing
    terms_with_content = {t for t in all_terms if t in web_content and web_content[t]}
    terms_to_process_with_content = list(terms_with_content & terms_to_process_set)

    logging.info(
        f"{len(terms_with_content)} out of {len(all_terms)} total terms have web content"
    )
    logging.info(
        f"Processing web-based relationships for {len(terms_to_process_with_content)} terms out of {len(terms_to_process_set)} specified terms"
    )

    if not terms_to_process_with_content:
        logging.warning("No terms to process with web content found")
        return G

    # Create mapping from term to level
    term_to_level = {}
    for level, terms in terms_by_level.items():
        for term in terms:
            term_to_level[term] = level

    # --- Precompute URL sets and relevant content --- 
    term_urls = {}
    term_relevant_content = defaultdict(list) # Use defaultdict

    logging.info("Building URL mappings and extracting relevant content")
    for term in tqdm(terms_with_content, desc="Processing term URLs and content"):
        urls = set()
        content_list = web_content.get(term, []) # Get content for the term
        
        for entry in content_list:
            url = entry.get("url", "")
            if not url:
                continue
            normalized_url = normalize_url(url)
            urls.add(normalized_url)
            
            # Store relevant content entry keyed by normalized URL
            # Keep only necessary fields to reduce memory usage
            processed_content = entry.get("processed_content")
            if processed_content:
                 term_relevant_content[term].append({
                    "url": normalized_url,
                    "processed_content": processed_content,
                 }) # Store by term

        term_urls[term] = urls
    # --- End Precomputation ---

    # --- Iterate through pairs and apply new logic ---
    edges_added = 0
    llm_verifications = 0
    high_overlap_edges = 0

    # Use combinations to avoid processing pairs twice
    term_pairs = list(itertools.combinations(terms_to_process_with_content, 2))
    
    logging.info(f"Checking {len(term_pairs)} pairs for web-based relationships")
    
    # Process pairs
    for term1, term2 in tqdm(term_pairs, desc="Finding web relationships"):
        urls1 = term_urls.get(term1, set())
        urls2 = term_urls.get(term2, set())
        
        if not urls1 or not urls2: # Skip if one term has no URLs
            continue
            
        shared_urls = urls1.intersection(urls2)
        shared_count = len(shared_urls)
        
        # Skip if number of shared URLs is less than 2
        if shared_count < 2:
            continue
            
        # --- High Overlap Check --- 
        total_urls1 = len(urls1)
        total_urls2 = len(urls2)
        
        is_high_overlap = (shared_count / total_urls1 > 0.75) and (shared_count / total_urls2 > 0.75)
        
        if is_high_overlap:
            level1 = term_to_level.get(term1, float("inf"))
            level2 = term_to_level.get(term2, float("inf"))
            is_cross_level = level1 != level2
            
            edge_attrs = {
                "relationship_type": "web",
                "detection_method": "high_url_overlap",
                "strength": 0.9, # High confidence for significant overlap
                "is_cross_level": is_cross_level,
                "reason": f">75% URL overlap ({shared_count}/{total_urls1} and {shared_count}/{total_urls2})",
                "shared_url_count": shared_count,
            }
            G.add_edge(term1, term2, **edge_attrs)
            edges_added += 1
            high_overlap_edges += 1
            logging.debug(f"Added high overlap edge: '{term1}' <-> '{term2}'")
            continue # Move to next pair
            
        # --- LLM Verification for pairs with >= 2 shared URL (but not high overlap) ---
        llm_verifications += 1
        logging.debug(f"Checking LLM for: '{term1}' <-> '{term2}' (Shared URLs: {shared_count})")
    
        # Gather relevant content snippets for shared URLs
        # The verify_edge_with_llm function now handles snippet extraction internally
        
        llm_verified, votes = verify_edge_with_llm_sync(
            term1, 
            term2, 
            term_relevant_content, # Pass the precomputed content
            list(shared_urls),
            provider=None # Use random provider
        )
        
        # Log the concise summary line here
        logging.info(f"LLM verification for '{term1}' - '{term2}': {llm_verified} (Votes: {votes})")
        
        if llm_verified:
            level1 = term_to_level.get(term1, float("inf"))
            level2 = term_to_level.get(term2, float("inf"))
            is_cross_level = level1 != level2
            
            edge_attrs = {
                "relationship_type": "web",
                "detection_method": "llm_verified_web", 
                "strength": 0.85, # High confidence if LLM verifies
                "is_cross_level": is_cross_level,
                "reason": f"LLM verified based on content from {shared_count} shared URLs (Votes: {votes})",
                "shared_url_count": shared_count,
                "llm_votes": votes, # Store votes for analysis
            }
            G.add_edge(term1, term2, **edge_attrs)
            edges_added += 1
            logging.debug(f"Added LLM-verified edge: '{term1}' <-> '{term2}'")

    logging.info(f"Web-based edge summary:")
    logging.info(f"- Total pairs checked: {len(term_pairs)}")
    logging.info(f"- Pairs with shared URLs checked by LLM: {llm_verifications}")
    logging.info(f"- Edges added via >50% URL overlap: {high_overlap_edges}")
    logging.info(f"- Edges added via LLM verification: {edges_added - high_overlap_edges}")
    logging.info(f"- Total web-based edges added in this run: {edges_added}")
    
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
    1. Level (prefer lowest level first) - This is the absolute priority
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
        
        # First, group terms by level
        terms_grouped_by_level = {}
        for term in component_terms:
            level = term_to_level.get(term, float("inf"))
            if level not in terms_grouped_by_level:
                terms_grouped_by_level[level] = []
            terms_grouped_by_level[level].append(term)
        
        # Find the minimum level in this component
        min_level = min(terms_grouped_by_level.keys())
        
        # If there's only one term at the minimum level, it's automatically the canonical form
        if len(terms_grouped_by_level[min_level]) == 1:
            canonical_term = terms_grouped_by_level[min_level][0]
            variations = set(t for t in component_terms if t != canonical_term)
            canonical_mapping[canonical_term] = variations
            continue
            
        # If there are multiple terms at the minimum level, use secondary criteria
        candidate_terms = terms_grouped_by_level[min_level]
        
        # Score each candidate term based on secondary criteria
        term_scores = []
        for term in candidate_terms:
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

            # Create score tuple (connectivity, url_count, -term_length)
            term_scores.append((term, (-connectivity, -url_count, term_length)))

        # Sort by secondary criteria
        sorted_terms = [term for term, score in sorted(term_scores, key=lambda x: x[1])]

        # The first term is our canonical term
        canonical_term = sorted_terms[0]
        
        # All other terms in the component are variations
        variations = set(t for t in component_terms if t != canonical_term)

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
    if not terms_to_check:
        logging.info("No new terms to check for transitive relationships")
        return []

    # Pre-compute term info only for terms we need
    logging.info("Pre-computing term info for new terms")
    term_info = {}
    
    # We need info for both new terms and their potential neighbors
    terms_needing_info = set(terms_to_check)  # Start with new terms
    
    # Add neighbors of new terms that might form transitive relationships
    for term in terms_to_check:
        if term in G:
            terms_needing_info.update(G.neighbors(term))
    
    logging.info(f"Computing term info for {len(terms_needing_info)} terms (new terms + their neighbors)")
    
    for term in terms_needing_info:
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
    logging.info("Computing common neighbors for new terms")
    common_neighbors_dict = {}
    
    # For incremental processing, we need to check:
    # 1. New terms with each other
    # 2. New terms with existing terms that have common neighbors
    term_pairs = []
    
    # Get all existing terms (excluding new terms)
    existing_terms = set(terms) - terms_to_check
    
    # First, check pairs of new terms
    new_terms_list = list(terms_to_check)
    for i, term1 in enumerate(new_terms_list):
        # Check with other new terms
        for term2 in new_terms_list[i+1:]:
            term_pairs.append((term1, term2))
            
        # Check with existing terms
        for term2 in existing_terms:
            if term1 < term2:  # Maintain consistent ordering
                term_pairs.append((term1, term2))
            else:
                term_pairs.append((term2, term1))
    
    logging.info(f"Checking {len(term_pairs)} term pairs for transitive relationships")
    
    # Find common neighbors for each pair
    pairs_processed = 0
    pairs_with_common = 0
    total_pairs = len(term_pairs)
    
    for term1, term2 in term_pairs:
        if term1 in G and term2 in G:  # Only check if both terms are in graph
            neighbors1 = set(G.neighbors(term1))
            neighbors2 = set(G.neighbors(term2))
            common = neighbors1 & neighbors2
            if common:
                common_neighbors_dict[(term1, term2)] = common
                pairs_with_common += 1
        pairs_processed += 1
        
        # Log progress every 10%
        if pairs_processed % max(1, total_pairs // 10) == 0:
            percentage = (pairs_processed / total_pairs) * 100
            logging.info(f"Processed {pairs_processed}/{total_pairs} pairs ({percentage:.1f}%). Found {pairs_with_common} pairs with common neighbors.")

    if not common_neighbors_dict:
        logging.info("No pairs with common neighbors found")
        return []

    logging.info(f"Found {len(common_neighbors_dict)} pairs with common neighbors")

    # Process in parallel
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    # Split into chunks for parallel processing
    items = list(common_neighbors_dict.items())
    chunk_size = max(100, len(items) // (multiprocessing.cpu_count() * 2))
    chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    logging.info(f"Processing {len(chunks)} chunks in parallel")

    added_edges = []
    chunks_processed = 0
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
                chunks_processed += 1
                if chunks_processed % max(1, len(chunks) // 10) == 0:
                    percentage = (chunks_processed / len(chunks)) * 100
                    logging.info(f"Processed {chunks_processed}/{len(chunks)} chunks ({percentage:.1f}%). Found {len(added_edges)} edges so far.")
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
                
                # Check if the edge direction matches the canonical direction
                if method == "plural_form":
                    # If we're dealing with plural form but levels caused a different term to be canonical
                    if (canonical.endswith("s") and not variation.endswith("s")) or \
                       (not canonical.endswith("s") and variation.endswith("s")):
                        # Check which one is the canonical form by level
                        canonical_level = term_to_level.get(canonical, float("inf"))
                        variation_level = term_to_level.get(variation, float("inf"))
                        
                        if canonical_level < variation_level:
                            # Level-based selection overrode the plural form preference
                            if canonical.endswith("s"):
                                reason = "Level priority: plural form preferred (lower level)"
                            else:
                                reason = "Level priority: singular form preferred (lower level)"
                elif method == "academic_suffix":
                    # For academic suffix variations, also check if level caused a different term to be canonical
                    canonical_level = term_to_level.get(canonical, float("inf"))
                    variation_level = term_to_level.get(variation, float("inf"))
                    
                    if canonical_level < variation_level:
                        # The form from the lower level was chosen as canonical
                        if "theories" in variation or "theory" in variation:
                            reason = "Level priority: term chosen from lower level (overriding theory preference)"
                        elif "education" in variation:
                            reason = "Level priority: term chosen from lower level (overriding education suffix)"
                        else:
                            # Extract the suffix that differs between them for a clearer message
                            canonical_parts = canonical.split()
                            variation_parts = variation.split()
                            reason = "Level priority: term chosen from lower level (overriding academic suffix preference)"
                
                # Create the variation reason entry
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
    Use LLM to verify if two academic terms are related enough to create an edge based on shared web content.
    Calls the LLM 3 times with randomly selected providers and takes the majority decision (at least 2 out of 3).
    
    Args:
        term1: First academic term
        term2: Second academic term
        term_relevant_content: Dict mapping terms to their relevant content, including processed snippets.
        shared_urls: List of URLs shared between the terms.
        provider: Optional LLM provider (overridden by random selection if not specified)
        
    Returns:
        Tuple containing:
        - Boolean indicating whether the terms should be connected (majority vote).
        - List of boolean results from each LLM call.
    """
    try:
        # Extract relevant snippets from shared URLs for both terms
        snippets_term1 = []
        snippets_term2 = []
        
        # Check if shared_urls and term_relevant_content are available
        if shared_urls and term_relevant_content:
            # Limit to a maximum number of snippets to avoid overly long prompts
            MAX_SNIPPETS_PER_TERM = 3 
            
            # Collect snippets for term1 from shared URLs
            term1_content = term_relevant_content.get(term1, [])
            for url in shared_urls:
                if len(snippets_term1) >= MAX_SNIPPETS_PER_TERM:
                    break
                for entry in term1_content:
                    if entry.get("url") == url and entry.get("processed_content"):
                        # Add snippet, clean up newlines
                        snippet = entry["processed_content"].replace("\\n", " ").strip()
                        if snippet:
                             snippets_term1.append(f"- URL: {url}\n  Snippet: {snippet}")
                        break # Only take one snippet per URL per term
                        
            # Collect snippets for term2 from shared URLs
            term2_content = term_relevant_content.get(term2, [])
            for url in shared_urls:
                if len(snippets_term2) >= MAX_SNIPPETS_PER_TERM:
                    break
                for entry in term2_content:
                    if entry.get("url") == url and entry.get("processed_content"):
                         # Add snippet, clean up newlines
                        snippet = entry["processed_content"].replace("\\n", " ").strip()
                        if snippet:
                            snippets_term2.append(f"- URL: {url}\n  Snippet: {snippet}")
                        break # Only take one snippet per URL per term

        # Format context for the LLM prompt
        snippet1_str = "\n".join(snippets_term1) if snippets_term1 else "No relevant snippets found."
        snippet2_str = "\n".join(snippets_term2) if snippets_term2 else "No relevant snippets found."

        context_str = f"""
Context for '{term1}':
{snippet1_str}

Context for '{term2}':
{snippet2_str}
        """
        
        system_prompt = f"""**Role:** AI Expert in Academic Terminology

**Goal:** Determine if Term 1 and Term 2 are functionally equivalent for deduplication purposes. Equivalence means they are synonyms, represent the identical academic concept/field, or are extremely close variations commonly used interchangeably.

**Examples of Expected Decisions:**
*   "Accountancy" vs "Accounting" -> YES (Synonyms/Identical Field)
*   "Biostatistics" vs "Statistics" -> NO (Subfield/Specialization)
*   "Computer Science" vs "Computer Engineering" -> NO (Distinct but related fields with different focus)
*   "Theoretical Physics" vs "Experimental Physics" -> NO (Different approaches within a field)
*   "Health Care Management" vs "Healthcare Administration" -> YES (Very close variations, often used interchangeably)
*   "Economics" vs "Political Science" -> NO (Distinct fields)

**Inputs:**
1.  Term 1: [Term String]
2.  Term 2: [Term String]
3.  Context Snippets: Text extracted from web pages mentioning both terms.
4.  Your Internal Knowledge Base: Your understanding of academic fields, concepts, and relationships.

**Process:**

1.  **Internal Knowledge Assessment:**
    *   Based SOLELY on your internal knowledge, what is the relationship between Term 1 and Term 2? (e.g., Equivalent, Distinct, Subfield, Related, Unsure).

2.  **Context Snippet Analysis:**
    *   Analyze the provided Context Snippets. What relationship do they explicitly state or strongly imply?
    *   Look for: Explicit definitions, synonym indicators ("also known as", "or"), clear interchangeable use for the *exact same* concept, or statements defining a hierarchy/difference (e.g., "X is a branch of Y", "X focuses on ..., while Y focuses on...").
    *   Evaluate the strength and clarity of the snippet evidence. Is it direct and unambiguous, or vague/contradictory?

3.  **Synthesis & Decision:**
    *   **If Knowledge & Snippets AGREE (Equivalent):** Output YES.
    *   **If Knowledge & Snippets AGREE (Distinct/Subfield):** Output NO.
    *   **If Knowledge says Equivalent, BUT Snippets provide STRONG evidence for Distinct/Subfield:** Trust the strong snippet evidence. Output NO.
    *   **If Knowledge says Distinct/Subfield, BUT Snippets provide STRONG evidence for Equivalence:** Trust the strong snippet evidence. Output YES.
    *   **If EITHER Knowledge OR Snippets are Uncertain/Ambiguous/Contradictory/Insufficient:** Default to conservative. Output NO.
    *   **If Knowledge is Uncertain, but Snippets provide WEAK/UNCLEAR evidence:** Output NO.

**Output:**
*   Respond with ONLY "YES" or "NO".
"""
        
        user_prompt = f"""Term 1: "{term1}"
Term 2: "{term2}"

Context Snippets:
{context_str}

Are "{term1}" and "{term2}" equivalent based on the synthesis process defined in the system prompt? (YES/NO)
"""

        # Run 3 LLM calls and collect the results
        results = []
        num_attempts = 3
        
        for attempt in range(num_attempts):
            try:
                # Get random provider and model for each attempt
                random_provider, random_model = get_random_llm_config()
                
                # Override with provided provider if specified
                if provider:
                    random_provider = provider
                
                # Get a fresh LLM instance for each attempt with the random provider/model
                llm = init_llm(random_provider, random_model)
                
                logging.debug(f"LLM verification attempt {attempt+1}/{num_attempts} for '{term1}' and '{term2}' using {random_provider}/{random_model}")
                
                response = llm.infer(prompt=user_prompt, system_prompt=system_prompt)
                
                # Parse the response - look for YES or NO
                response_text = response.text.strip().upper()
                
                # Log the response for this attempt
                logging.debug(f"LLM verification attempt {attempt+1} result: {response_text}")
                
                # Record the result (True for YES, False for NO)
                if "YES" in response_text:
                    results.append(True)
                elif "NO" in response_text:
                    results.append(False)
                else:
                    # Default to conservative approach (no edge) if response is unclear
                    logging.debug(f"Unclear LLM verification response for attempt {attempt+1}: {response_text}")
                    results.append(False)
            
            except Exception as e:
                logging.error(f"Error in LLM edge verification attempt {attempt+1}: {str(e)}")
                # Default to conservative approach on errors
                results.append(False)
        
        # Count the number of True results
        true_count = results.count(True)
        
        # Take the majority decision
        final_result = true_count >= 2
        
        # Return the final result and the list of individual votes
        return final_result, results 
            
    except Exception as e:
        logging.error(f"Error preparing LLM edge verification for {term1} and {term2}: {str(e)}")
        return False, [] # Return a default tuple on error

def verify_edge_with_llm_sync(
    term1: str, 
    term2: str, 
    term_relevant_content: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    shared_urls: Optional[List[str]] = None,
    provider: Optional[str] = None
) -> Tuple[bool, List[bool]]: # Updated return type hint
    """
    Synchronous wrapper for verify_edge_with_llm
    
    Args:
        term1: First academic term
        term2: Second academic term
        term_relevant_content: Dict mapping terms to their relevant content
        shared_urls: List of shared URLs between the terms
        provider: Optional LLM provider
        
    Returns:
         Tuple containing:
        - Boolean indicating whether the terms should be connected (majority vote).
        - List of boolean results from each LLM call.
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
            
        result_tuple = loop.run_until_complete( # Assign tuple to variable
            verify_edge_with_llm(
                term1, 
                term2, 
                term_relevant_content, 
                shared_urls, 
                provider
            )
        )
        logging.debug(f"LLM verification result for '{term1}' and '{term2}': {result_tuple[0]} (Votes: {result_tuple[1]})") # Log using tuple elements
        return result_tuple # Return the complete tuple
    except Exception as e:
        logging.error(f"Error in synchronous LLM edge verification: {str(e)}")
        return False, [] # Return a default tuple on error
