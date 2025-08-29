"""
Module for deduplicating technical concepts using different strategies:
1. Rule-based deduplication (within-level and cross-level)
2. Web-based deduplication (within-level and cross-level)
3. LLM-based deduplication (within-level only)
4. Graph-based deduplication (recommended approach combining rule and web-based)
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
from pydantic import BaseModel, Field
import logging

import os
import sys

# Add parent directory to path to allow imports from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_glossary.utils.llm_simple import infer_text, get_random_llm_config

# Try both import paths for WebContent to handle different import structures
try:
    from utils.web_miner import WebContent
except ImportError:
    try:
        from generate_glossary.utils.web_miner import WebContent
    except ImportError:
        logging.error("Could not import WebContent from either utils.web_miner or generate_glossary.utils.web_miner")
        # Define a minimal version to allow code to run
        class WebContent:
            url: str
            title: str
            snippet: str
            raw_content: str
            processed_content: str
            score: float = 0.5
            is_verified: bool = False
            verification_reason: str = ""

from .utils import (
    normalize_text, get_term_variations, is_compound_term,
    process_in_parallel, timing_decorator,
)

# Import new pipeline for graph-based deduplication
from .pipeline import deduplicate_progressive as _deduplicate_progressive

# Type aliases
DeduplicationResult = Dict[str, Any]
VariationMap = Dict[str, Set[str]]

ACADEMIC_VARIATIONS = {
    "plural_forms": {
        "s": "es",
        "x": "xes",
        "ch": "ches",
        "sh": "shes",
    },
    "spelling_variations": {
        "behaviour": "behavior",
        "analyse": "analyze",
        "modelling": "modeling",
        "defence": "defense",
        "centre": "center",
        "programme": "program",
    },
    "acronyms": {
        "AI": "artificial intelligence",
        "ML": "machine learning",
        "DL": "deep learning",
        "NLP": "natural language processing",
        "CV": "computer vision",
    }
}
# Add Pydantic models for LLM deduplication
class TermVariation(BaseModel):
    """Model for a term and its variations"""
    preferred_term: str = Field(description="The preferred term to use")
    variations: List[str] = Field(description="List of variations of this term")

class TermVariations(BaseModel):
    """Model for all term variations"""
    term_groups: List[TermVariation] = Field(description="Groups of related terms")

# LLM system prompt
LLM_SYSTEM_PROMPT = """You are an expert in academic terminology and research classification.
Your task is to identify terms that are variations of each other and choose the most appropriate
preferred term for each group."""

def build_llm_prompt(terms: List[str]) -> str:
    """Build prompt for term variation analysis"""
    terms_str = "\n".join(f"- {term}" for term in terms)
    
    return f"""Analyze these academic terms and identify which ones are variations of each other.
Only consider these specific variation patterns:

1. Plural/singular forms:
   - science/sciences
   - study/studies
   - art/arts

2. British/American spelling:
   - behaviour/behavior
   - analyse/analyze
   - modelling/modeling
   - defence/defense
   - centre/center
   - programme/program

Do NOT consider:
- Prefixes (e.g., "bio-", "neuro-", etc.)
- Field qualifiers (e.g., "applied", "theoretical")
- Acronyms or abbreviations
- Word reorderings
- Any other patterns not explicitly listed above

{terms_str}

Group related terms together and:
1. Choose the most appropriate preferred term for each group (prefer plural forms when available)
2. List all variations of that term that match the above patterns

Return the results in this format:
{{
  "term_groups": [
    {{
      "preferred_term": "computer sciences",
      "variations": ["computer science"]
    }},
    ...
  ]
}}

Only group terms that match the specified patterns exactly. If a term has no variations, don't include it."""

def process_batch_basic(batch_terms: List[str]) -> Dict[str, Any]:
    """Process a batch of terms for basic deduplication."""
    logging.info(f"Processing batch of {len(batch_terms)} terms")
    
    normalized_terms = {}
    variations = defaultdict(set)
    
    # First normalize all terms
    for term in batch_terms:
        norm = normalize_text(term)
        normalized_terms[term] = norm
    
    logging.info(f"Normalized terms: {normalized_terms}")
    
    # Group terms by their normalized form
    norm_to_terms = defaultdict(set)
    for term, norm in normalized_terms.items():
        norm_to_terms[norm].add(term)
    
    logging.info(f"Terms grouped by normalized form: {dict(norm_to_terms)}")
    
    # Process each term for variations
    for term in batch_terms:
        term_lower = term.lower()
        
        # Check plural/singular variations
        for singular, plural in ACADEMIC_VARIATIONS["plural_forms"].items():
            if term_lower.endswith(singular):
                plural_form = term_lower[:-len(singular)] + plural
                for other_term in batch_terms:
                    if other_term.lower() == plural_form:
                        # Always prefer the plural form as canonical
                        variations[other_term].add(term)
                        break
            elif term_lower.endswith(plural):
                singular_form = term_lower[:-len(plural)] + singular
                for other_term in batch_terms:
                    if other_term.lower() == singular_form:
                        # Always prefer the plural form as canonical
                        variations[term].add(other_term)
                        break
        
        # Check spelling variations
        for british, american in ACADEMIC_VARIATIONS["spelling_variations"].items():
            if british in term_lower:
                american_form = term_lower.replace(british, american)
                for other_term in batch_terms:
                    if other_term.lower() == american_form:
                        # Prefer American spelling as canonical
                        variations[other_term].add(term)
                        break
            elif american in term_lower:
                british_form = term_lower.replace(american, british)
                for other_term in batch_terms:
                    if other_term.lower() == british_form:
                        # Prefer American spelling as canonical
                        variations[term].add(other_term)
                        break
    
    logging.info(f"Variations: {dict(variations)}")
    
    return {
        "normalized_terms": normalized_terms,
        "variations": dict(variations)
    }

@timing_decorator
def deduplicate_basic(
    terms: List[str], 
    batch_size: int = 100, 
    max_workers: Optional[int] = None
) -> DeduplicationResult:
    """Basic deduplication using reliable academic patterns."""
    if not terms:
        return {"deduplicated_terms": [], "variations": {}}

    logging.info(f"Starting basic deduplication of {len(terms)} terms")
    
    # Process terms in parallel batches
    batch_results = process_in_parallel(
        items=terms,
        process_func=process_batch_basic,
        batch_size=batch_size,
        max_workers=max_workers,
        desc="Basic deduplication"
    )
    
    # Combine results from all batches
    all_normalized = {}
    all_variations = defaultdict(set)
    
    for result in batch_results:
        all_normalized.update(result.get("normalized_terms", {}))
        for term, vars in result.get("variations", {}).items():
            all_variations[term].update(vars)
    
    # Get unique terms (those that aren't variations of other terms)
    all_variations_set = {var for vars in all_variations.values() for var in vars}
    deduplicated_terms = [term for term in terms if term not in all_variations_set]
    
    # Add canonical terms from variations
    deduplicated_terms.extend(all_variations.keys())
    deduplicated_terms = sorted(set(deduplicated_terms))
    
    logging.info(f"Found {len(deduplicated_terms)} unique terms and {len(all_variations)} variation groups")
    
    return {
        "deduplicated_terms": deduplicated_terms,
        "variations": dict(all_variations)
    }

@timing_decorator
def deduplicate_rule_based(
    terms: List[str],
    higher_level_terms: Optional[Dict[int, List[str]]] = None,
    batch_size: int = 100,
    max_workers: Optional[int] = None
) -> DeduplicationResult:
    """Rule-based deduplication using reliable academic patterns."""
    if not terms:
        return {
            "deduplicated_terms": [],
            "variations": {},
            "cross_level_variations": {}
        }

    logging.info(f"Starting rule-based deduplication with {len(terms)} terms")
    
    # First do within-level deduplication
    within_level_result = deduplicate_basic(
        terms,
        batch_size=batch_size,
        max_workers=max_workers
    )
    
    deduplicated_terms = within_level_result["deduplicated_terms"]
    variations = within_level_result["variations"]
    
    # If there are higher level terms, do cross-level deduplication
    cross_level_variations = {}
    if higher_level_terms:
        for level, level_terms in higher_level_terms.items():
            # For each higher level term, check if any of our terms are variations
            for higher_term in level_terms:
                higher_variations = get_term_variations(higher_term)
                
                for term in deduplicated_terms[:]:  # Copy list to allow modification
                    if term.lower() in higher_variations:
                        # Remove term from deduplicated terms and add as variation
                        deduplicated_terms.remove(term)
                        if higher_term not in cross_level_variations:
                            cross_level_variations[higher_term] = []
                        cross_level_variations[higher_term].append(term)
                        
                        # Also move any variations of this term
                        if term in variations:
                            cross_level_variations[higher_term].extend(variations[term])
                            del variations[term]
    
    logging.info(f"Found {len(deduplicated_terms)} unique terms and {len(variations)} variation groups")
    if higher_level_terms:
        logging.info(f"Cross-level variations: {cross_level_variations}")
    
    return {
        "deduplicated_terms": deduplicated_terms,
        "variations": variations,
        "cross_level_variations": cross_level_variations
    }

@timing_decorator
def deduplicate_web_based(
    terms, 
    web_content, 
    higher_level_terms=None, 
    batch_size=100, 
    max_workers=4, 
    url_overlap_threshold=2, 
    content_similarity_threshold=0.6,
    min_relevance_score=0.3
):
    """Deduplicate terms based on web content overlap.
    
    Args:
        terms: List of terms to deduplicate
        web_content: Dict mapping terms to their web content
        higher_level_terms: Optional list of terms from higher levels to check against
        batch_size: Size of batches for parallel processing
        max_workers: Number of parallel workers
        url_overlap_threshold: Minimum number of overlapping URLs required
        content_similarity_threshold: Minimum content similarity required
        min_relevance_score: Minimum relevance score for content to be considered relevant
    
    Returns:
        Dict with deduplicated terms and variations
    """
    if not terms:
        return {"deduplicated_terms": [], "variations": {}}

    # First pass - identify canonical terms (prefer terms with "sciences" or "studies")
    canonical_terms = []
    variations = {}
    processed_terms = set()

    # Helper function to normalize URLs
    def normalize_url(url):
        # Remove trailing slashes and query parameters
        url = url.split('?')[0].rstrip('/')
        return url

    # Helper function to get normalized URLs for a term, filtering by relevance score
    def get_normalized_urls(term):
        if term not in web_content:
            return set()
        urls = set()
        for entry in web_content[term]:
            if 'url' not in entry:
                continue
                
            # Check for relevance score in the entry
            relevance_score = None
            if isinstance(entry, dict):
                relevance_score = entry.get("relevance_score", None)
            else:
                relevance_score = getattr(entry, "relevance_score", None)
            
            # If relevance score is not available in the entry, calculate it
            if relevance_score is None:
                try:
                    # Import here to avoid circular imports
                    from generate_glossary.validator.validation_utils import calculate_relevance_score
                    relevance_score = calculate_relevance_score(term, entry)
                    
                    # Store the calculated score back in the entry for future use
                    if isinstance(entry, dict):
                        entry["relevance_score"] = relevance_score
                    else:
                        setattr(entry, "relevance_score", relevance_score)
                except ImportError:
                    # If calculation fails, use a default score
                    logging.warning(f"Could not calculate relevance score for {term}, using default")
                    relevance_score = 0.5
            
            # Only include URLs from content with sufficient relevance
            if relevance_score >= min_relevance_score:
                urls.add(normalize_url(entry['url']))
                
        return urls

    # First pass - identify canonical terms with "sciences" or "studies"
    for term in terms:
        if term in processed_terms:
            continue
            
        if term.endswith(('sciences', 'studies')):
            canonical_terms.append(term)
            processed_terms.add(term)
            
            # Find variations by checking URL overlap
            term_urls = get_normalized_urls(term)
            if not term_urls:
                continue
                
            variations[term] = []
            for other_term in terms:
                if other_term == term or other_term in processed_terms:
                    continue
                    
                other_urls = get_normalized_urls(other_term)
                if len(term_urls & other_urls) >= url_overlap_threshold:
                    variations[term].append(other_term)
                    processed_terms.add(other_term)

    # Second pass - handle remaining terms
    for term in terms:
        if term in processed_terms:
            continue
            
        term_urls = get_normalized_urls(term)
        if not term_urls:
            canonical_terms.append(term)
            continue
            
        # Check if this term should be a variation of an existing canonical term
        is_variation = False
        for canonical_term in canonical_terms:
            canonical_urls = get_normalized_urls(canonical_term)
            if len(term_urls & canonical_urls) >= url_overlap_threshold:
                if canonical_term not in variations:
                    variations[canonical_term] = []
                variations[canonical_term].append(term)
                processed_terms.add(term)
                is_variation = True
                break
                
        if not is_variation:
            canonical_terms.append(term)
            processed_terms.add(term)

    # Handle cross-level deduplication if higher_level_terms provided
    if higher_level_terms:
        for higher_term in higher_level_terms:
            if higher_term not in web_content:
                continue
                
            higher_urls = get_normalized_urls(higher_term)
            if not higher_urls:
                continue
                
            # Check each canonical term against higher level terms
            for term in canonical_terms[:]:
                term_urls = get_normalized_urls(term)
                if len(higher_urls & term_urls) >= url_overlap_threshold:
                    canonical_terms.remove(term)
                    if higher_term not in variations:
                        variations[higher_term] = []
                    variations[higher_term].append(term)
                    # Move this term's variations to the higher term
                    if term in variations:
                        variations[higher_term].extend(variations[term])
                        del variations[term]

    # Log results
    logging.info(f"Found {len(canonical_terms)} unique terms and {sum(len(v) for v in variations.values())} variations")
    
    return {
        "deduplicated_terms": canonical_terms,
        "variations": variations
    }

@timing_decorator
def deduplicate_llm_based(
    terms: List[str],
    potential_duplicates: Optional[Dict[str, List[str]]] = None,
    batch_size: int = 20,
    max_workers: Optional[int] = None,
    provider: str = "gemini"
) -> DeduplicationResult:
    """
    LLM-based deduplication that can verify potential duplicates found by other methods.
    
    Args:
        terms: List of all terms
        potential_duplicates: Dictionary of potential duplicates to verify
            (key: preferred term, value: list of potential variations)
        batch_size: Number of terms per batch for LLM processing
        max_workers: Number of parallel workers
        provider: LLM provider to use
    
    Returns:
        DeduplicationResult with verified duplicates
    """
    logging.info(f"Starting LLM-based deduplication")
    
    # Initialize LLM
    # LLM is now initialized automatically when calling infer_text
    
    # If potential_duplicates is provided, use it for verification
    if potential_duplicates:
        logging.info(f"Verifying {len(potential_duplicates)} potential duplicate groups")
        verified_variations = {}
        
        # Process each group of potential duplicates
        for preferred_term, variations in potential_duplicates.items():
            all_terms = [preferred_term] + variations
            
            verification_prompt = f"""Analyze these academic terms and determine if they are truly variations of the same concept:

Potential preferred term: {preferred_term}
Potential variations:
{', '.join(variations)}

Consider:
- Singular/plural forms
- Different word orders
- British/American spelling
- Technical variations
- Field-specific terminology

Return only YES if they are variations of the same concept, or NO if they are distinct concepts.
"""
            try:
                # Get LLM verification
                response = llm.complete(
                    system_prompt=LLM_SYSTEM_PROMPT,
                    user_prompt=verification_prompt
                ).strip().upper()
                
                # Parse response
                if "YES" in response:
                    verified_variations[preferred_term] = set(variations)
                else:
                    logging.info(f"LLM rejected duplicate group: {preferred_term} -> {variations}")
            except Exception as e:
                logging.error(f"Error in LLM verification: {e}")
        
        # Get unique terms after removing verified variations
        all_variations = {var for vars in verified_variations.values() for var in vars}
        deduplicated_terms = [term for term in terms if term not in all_variations]
        
        return {
            "deduplicated_terms": deduplicated_terms,
            "variations": verified_variations
        }
    
    # If no potential_duplicates provided, process terms in batches to find duplicates
    def process_batch(batch_terms: List[str]) -> Dict[str, Any]:
        prompt = build_llm_prompt(batch_terms)
        
        try:
            # Get LLM response
            response = llm.complete(
                system_prompt=LLM_SYSTEM_PROMPT,
                user_prompt=prompt
            )
            
            # Parse response
            variations = TermVariations.parse_raw(response)
            
            # Convert to dictionary format
            batch_variations = {}
            for group in variations.term_groups:
                batch_variations[group.preferred_term] = set(group.variations)
            
            return batch_variations
            
        except Exception as e:
            logging.error(f"Error in LLM processing: {e}")
            return {}
    
    # Process in parallel batches
    batch_results = process_in_parallel(
        items=terms,
        process_func=process_batch,
        batch_size=batch_size,
        max_workers=max_workers,
        desc="LLM-based deduplication"
    )
    
    # Combine results from all batches
    variations = {}
    for result in batch_results:
        for canonical, variants in result.items():
            if canonical not in variations:
                variations[canonical] = set()
            variations[canonical].update(variants)
    
    # Get unique terms
    all_variations = {var for vars in variations.values() for var in vars}
    deduplicated_terms = [term for term in terms if term not in all_variations]
    
    return {
        "deduplicated_terms": deduplicated_terms,
        "variations": variations
    }

# Add wrapper for graph-based deduplication to match the interface of other deduplication functions
@timing_decorator
def deduplicate_graph_based(
    terms: List[str],
    web_content: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    higher_level_terms: Optional[Dict[int, List[str]]] = None,
    higher_level_web_content: Optional[Dict[int, Dict[str, List[Dict[str, Any]]]]] = None,
    min_score: float = 0.7,
    min_relevance_score: float = 0.3,
    batch_size: int = 100,
    max_workers: Optional[int] = None,
    use_enhanced_linguistics: bool = True,
    current_level: Optional[int] = None,
    cache_dir: Optional[str] = None,
    max_workers_transitive: Optional[int] = None
) -> DeduplicationResult:
    """
    Graph-based deduplication that combines rule-based and web-based approaches.
    
    This function builds a graph where:
    - Nodes are terms
    - Edges indicate terms are variations of each other
    - Edge weights indicate confidence in the variation relationship
    
    Args:
        terms: List of terms to deduplicate
        web_content: Dictionary mapping terms to their web content
        higher_level_terms: Dictionary mapping level numbers to lists of terms
        higher_level_web_content: Dictionary mapping level numbers to web content dictionaries
        min_score: Minimum score for web content to be considered valid
        min_relevance_score: Minimum relevance score for web content to be considered relevant
        batch_size: Batch size for parallel processing
        max_workers: Maximum number of worker processes
        use_enhanced_linguistics: Whether to use enhanced linguistic analysis
        current_level: Explicitly specified level for the current input terms
        cache_dir: Directory for caching intermediate results
        max_workers_transitive: Maximum number of worker processes for transitive closure
        
    Returns:
        DeduplicationResult: Dictionary containing deduplicated terms and metadata
    """
    # Prepare terms by level
    terms_by_level = {}
    
    # Add current level terms - use provided current_level if available
    if current_level is None:
        # Fall back to calculating based on higher_level_terms
        current_level = 0
        if higher_level_terms:
            current_level = max(higher_level_terms.keys()) + 1
    
    # Log the current level being used
    logging.info(f"Using level {current_level} for current input terms")
    
    # First, add all higher level terms to their respective levels
    if higher_level_terms:
        for level, level_terms in higher_level_terms.items():
            terms_by_level[level] = level_terms.copy()  # Make a copy to avoid modifying the original
            logging.info(f"Added {len(level_terms)} terms to level {level}")
    
    # Create a set of terms already assigned to higher levels for fast lookup
    higher_level_term_set = set()
    for level, level_terms in terms_by_level.items():
        if level < current_level:
            higher_level_term_set.update(level_terms)
    
    # Now add remaining terms to the current level, excluding those already in higher levels
    current_level_terms = [term for term in terms if term not in higher_level_term_set]
    terms_by_level[current_level] = current_level_terms
    logging.info(f"Added {len(current_level_terms)} terms to current level {current_level}")
    
    # Log which specific terms from current input were found in higher levels
    higher_level_matches = [term for term in terms if term in higher_level_term_set]
    if higher_level_matches:
        logging.info(f"Found {len(higher_level_matches)} terms from current input that are also in higher levels")
    
    # Prepare web content
    all_web_content = {}
    
    # Add current level web content
    if web_content:
        all_web_content.update(web_content)
    
    # Add higher level web content if provided
    if higher_level_web_content:
        for level, level_web_content in higher_level_web_content.items():
            all_web_content.update(level_web_content)
    
    # Prepare config for new pipeline
    config = {
        "min_text_similarity": 0.85,
        "min_embedding_similarity": 0.49,
        "min_url_overlap": 2,
        "min_relevance_score": min_relevance_score,
        "use_cross_level": True,
        "prefer_higher_level": False,
        "remove_weak_edges": True,
        "weak_edge_threshold": 0.3
    }
    
    # Call the new progressive deduplication pipeline
    result = _deduplicate_progressive(
        terms_by_level=terms_by_level,
        web_content=all_web_content,
        embeddings=None,  # TODO: Add embeddings support if available
        config=config,
        cache_dir=cache_dir,
        use_cache=True if cache_dir else False,
        save_checkpoints=True if cache_dir else False
    )
    
    # Convert new result format to match old format for backward compatibility
    if "variations" in result:
        # Convert variations dict to old format
        all_terms = set(result.get("canonical_terms", []))
        for variations in result["variations"].values():
            all_terms.update(variations)
        result["deduplicated_terms"] = sorted(list(all_terms))
    
    # Add higher level terms and terms_by_level to result for use by the CLI
    result["higher_level_terms"] = higher_level_terms
    result["terms_by_level"] = terms_by_level
    
    # Add a list of canonical terms from higher levels
    higher_level_canonicals = []
    for level, level_terms in terms_by_level.items():
        if level < current_level:
            higher_level_canonicals.extend(level_terms)
    
    result["cross_level_canonicals"] = higher_level_canonicals
    
    return result 