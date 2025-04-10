import json
import logging
from collections import defaultdict
import numpy as np
import os
import re
from sentence_transformers import SentenceTransformer
import datetime
from typing import Optional, Dict, List, Tuple, Literal
import glob
import sys
from pydantic import BaseModel, Field

# Add the project root to sys.path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our existing LLM implementation
from generate_glossary.utils.llm import LLMFactory, Provider, OPENAI_MODELS, GEMINI_MODELS
from generate_glossary.utils.logger import setup_logger

# Setup logging
logger = setup_logger("sense_disambiguation.splitter")

# Define level-specific parameters
LEVEL_PARAMS = {
    0: {"eps": 0.6, "min_samples": 3, "description": "college or broad academic domain"},
    1: {"eps": 0.5, "min_samples": 2, "description": "academic department or field"},
    2: {"eps": 0.4, "min_samples": 2, "description": "research area or specialized topic"},
    3: {"eps": 0.3, "min_samples": 2, "description": "conference or journal topic"}
}

# Level-specific examples for prompting
LEVEL_EXAMPLES = {
    0: "Arts and Sciences, Engineering, Medicine, Business",
    1: "Computer Science, Psychology, Economics, Mechanical Engineering",
    2: "Machine Learning, Cognitive Psychology, Econometrics, Fluid Dynamics",
    3: "Natural Language Processing, Behavioral Economics, Computer Vision"
}

# Pydantic models for structured response
class FieldDistinctnessAnalysis(BaseModel):
    """Structured model for field distinctness analysis"""
    field1: str = Field(description="First academic field being compared")
    field2: str = Field(description="Second academic field being compared")
    verdict: Literal["DISTINCT", "NOT_DISTINCT"] = Field(description="Final determination of whether fields are distinct")
    explanation: str = Field(description="Detailed explanation for the verdict, comparing the fields based on core concepts, methodologies, and relationship (e.g., subfield, overlap, distinct domains).")

class SenseSplitter:
    """
    Handles the process of splitting potentially ambiguous terms into distinct senses
    based on resource content clustering and context.
    """

    def __init__(self,
                 hierarchy_file_path: str,
                 candidate_terms_list: list[str],
                 cluster_results: dict[str, list[int]], 
                 embedding_model_name: str = 'all-MiniLM-L6-v2',
                 use_llm_for_tags: bool = True,
                 llm_provider: str = Provider.OPENAI,
                 llm_model: Optional[str] = None,
                 level: int = 2,
                 output_dir: Optional[str] = None):
        """Initialize the SenseSplitter to split ambiguous terms into distinct senses."""
        self.hierarchy_file_path = hierarchy_file_path
        self.candidate_terms = candidate_terms_list
        self.cluster_results = cluster_results
        self.embedding_model_name = embedding_model_name
        self.use_llm_for_tags = use_llm_for_tags
        self.level = level
        self.output_dir = output_dir or "data/sense_disambiguation_results"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize storage
        self.hierarchy_data = None
        self.term_details = None
        self._embedding_model = None
        self.cluster_metrics = {}
        self._loaded = False
        self.detailed_cluster_info = {}
        self.level_params = LEVEL_PARAMS[self.level]
        
        # LLM settings
        self.llm_provider = llm_provider.lower() if llm_provider else Provider.OPENAI
        if self.llm_provider not in [Provider.OPENAI, Provider.GEMINI]:
            logger.warning(f"Unsupported provider: {llm_provider}. Falling back to OpenAI.")
            self.llm_provider = Provider.OPENAI
            
        # Set default model based on provider
        if not llm_model:
            self.llm_model = OPENAI_MODELS["default"] if self.llm_provider == Provider.OPENAI else GEMINI_MODELS["default"]
        else:
            self.llm_model = llm_model
            
        # Initialize LLM
        self._llm = None
        try:
            if self.use_llm_for_tags:
                self._init_llm()
            self.using_real_llm = self.use_llm_for_tags
        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {e}. Will use simulation for tagging.")
            self.using_real_llm = False
        
        # Caches
        self._parent_context_cache = {}
        self._term_level_cache = {}

    def _init_llm(self) -> None:
        """Initialize the LLM using the factory"""
        if self._llm is None:
            self._llm = LLMFactory.create_llm(
                provider=self.llm_provider,
                model=self.llm_model,
                temperature=0.2  # Low temperature for consistent tagging
            )
            logger.info(f"Initialized LLM: {self.llm_provider} - {self.llm_model}")

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            try:
                logger.info(f"Lazily loading embedding model: {self.embedding_model_name}")
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
            except Exception as e:
                logger.error(f"Failed to load sentence transformer model '{self.embedding_model_name}': {e}")
                # Prevent repeated attempts
                self._embedding_model = False # Use False to indicate failed load attempt
        # Return the model if loaded, otherwise return None (or False if load failed)
        return self._embedding_model if self._embedding_model else None

    def _load_hierarchy(self):
        """Loads the hierarchy data."""
        if self._loaded:
            return True
        logger.info(f"Loading hierarchy from {self.hierarchy_file_path}...")
        try:
            with open(self.hierarchy_file_path, 'r') as f:
                self.hierarchy_data = json.load(f)
            self.term_details = self.hierarchy_data.get('terms', {})
            if not self.term_details:
                logger.warning("Hierarchy file loaded, but 'terms' dictionary is missing or empty.")
                return False
            logger.info(f"Loaded {len(self.term_details)} terms from hierarchy.")
            self._loaded = True
            return True
        except FileNotFoundError:
            logger.error(f"Hierarchy file not found: {self.hierarchy_file_path}")
            return False
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {self.hierarchy_file_path}")
            return False

    def _get_term_level(self, term: str) -> Optional[int]:
        """
        Get the hierarchy level of a term from the loaded hierarchy data.
        
        Args:
            term: The term to look up
            
        Returns:
            The level as an integer (0-3) or None if term not found or level missing
        """
        # Check cache first
        if term in self._term_level_cache:
            return self._term_level_cache[term]
            
        if not self.term_details or term not in self.term_details:
            return None
            
        term_data = self.term_details[term]
        level = term_data.get('level')
        
        # Cache the result
        if level is not None:
            self._term_level_cache[term] = level
            
        return level

    def _filter_candidate_terms_by_level(self) -> list[str]:
        """
        Filter the candidate terms list to only include terms at the current processing level.
        
        Returns:
            A list of candidate terms that match the current level.
        """
        if not self._load_hierarchy():
            logger.error("Failed to load hierarchy. Cannot filter terms by level.")
            return []
            
        filtered_terms = []
        for term in self.candidate_terms:
            term_level = self._get_term_level(term)
            if term_level == self.level:
                filtered_terms.append(term)
                
        logger.info(f"Filtered {len(self.candidate_terms)} candidate terms to {len(filtered_terms)} terms at level {self.level}")
        return filtered_terms

    def _call_llm_for_tag(self, term: str, cluster_id: int, resource_snippets: list[str], parent_context: Optional[List[str]] = None) -> str | None:
        """
        Calls the LLM to generate a domain tag for a cluster using our codebase's LLM utilities.
        Includes level-specific context to improve tag relevance.
        
        Args:
            term: The term string.
            cluster_id: The cluster ID.
            resource_snippets: A list of resource content snippets for the cluster.
            parent_context: Optional list of parent terms for additional context.
            
        Returns:
            A tag string or None if the call failed.
        """
        if not self.using_real_llm or not self._llm:
            logger.warning(f"No LLM available. Using simulation for term '{term}', cluster {cluster_id}")
            return self._call_simulated_llm_for_tag(term, cluster_id, resource_snippets)
            
        logger.info(f"Calling LLM ({self.llm_provider}) for level {self.level} term '{term}', cluster {cluster_id}")
        
        SYSTEM_PROMPT = f"""You are an expert academic classification system specializing in categorizing content into specific academic domains.
Your task is to provide a concise, 1-3 word academic domain label for a term based on contextual information.
Focus on standard academic classifications rather than creating new or unusual categories.
Use established terminology from academic taxonomies.

You are categorizing a term at hierarchy level {self.level}, which represents a {LEVEL_PARAMS[self.level]['description']}.
Examples of terms at this level include: {LEVEL_EXAMPLES[self.level]}
"""
        
        # Take first 5 snippets at most, and truncate each snippet
        MAX_SNIPPETS = 5
        MAX_SNIPPET_LENGTH = 300
        truncated_snippets = [(" ".join(s.split()[:MAX_SNIPPET_LENGTH]) + ("..." if len(s.split()) > MAX_SNIPPET_LENGTH else "")) 
                             for s in resource_snippets[:MAX_SNIPPETS]]
        
        # Format as a simpler prompt that doesn't require JSON
        formatted_snippets = "\n\n".join([f"[Resource {i+1}]: {snippet}" 
                                         for i, snippet in enumerate(truncated_snippets)])
        
        # Include parent context information if available
        parent_context_str = ""
        if parent_context and len(parent_context) > 0:
            parent_context_str = f"\nParent context: This term appears under: {', '.join(parent_context[:5])}"
            if len(parent_context) > 5:
                parent_context_str += f" and {len(parent_context) - 5} more"
        
        prompt = f"""Generate a concise academic domain label (1-3 words) for the term '{term}' based on these resource snippets representing one sense/meaning.

This term is a {LEVEL_PARAMS[self.level]['description']} in the academic hierarchy.
Terms at this level typically include examples like: {LEVEL_EXAMPLES[self.level]}

Resource snippets:
{formatted_snippets}{parent_context_str}

Return ONLY the domain label, nothing else. No JSON, no explanation, just the 1-3 word label."""
        
        # Make multiple attempts in case of failure
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                # Use the LLM directly without pydantic model
                response = self._llm.infer(
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT
                )
                
                # Extract and return the tag
                if response and hasattr(response, 'text'):
                    # Just use the raw text response
                    tag = response.text.strip()
                    logger.info(f"LLM generated tag '{tag}' for term '{term}', cluster {cluster_id}")
                    return tag
                else:
                    logger.warning(f"Invalid response format from LLM for term '{term}', cluster {cluster_id}")
                    
            except Exception as e:
                logger.error(f"Error calling LLM (attempt {attempt+1}/{max_attempts}): {e}")
                if attempt < max_attempts - 1:
                    logger.info(f"Retrying LLM call for term '{term}', cluster {cluster_id}")
                
        # If all attempts fail, use simulation fallback
        logger.warning(f"All LLM attempts failed for term '{term}', cluster {cluster_id}. Using simulation.")
        return self._call_simulated_llm_for_tag(term, cluster_id, resource_snippets)

    def _call_simulated_llm_for_tag(self, term: str, cluster_id: int, resource_snippets: list[str]) -> str | None:
        """
        Simulates calling an LLM to generate a domain tag for a cluster.
        Used as fallback when no real LLM API is available.
        """
        logger.debug(f"Simulating LLM call for term '{term}', cluster {cluster_id}")
        
        # Check if we have resources to work with
        if not resource_snippets:
            logger.warning(f"Cannot simulate LLM tag for '{term}' cluster {cluster_id}: No snippets provided.")
            return None

        # Generate a generic domain tag based on the term and cluster ID
        # This creates a more generic simulation without hard-coded special cases
        simulated_tag = f"sense_{cluster_id + 1}"
        
        logger.debug(f"Simulated LLM response: '{simulated_tag}'")
        return simulated_tag

    def _standardize_tag(self, tag: str) -> str:
        """Converts a tag to lowercase, replaces spaces/hyphens with underscores, removes other symbols."""
        if not tag or not isinstance(tag, str):
            return ""
        tag = tag.lower()
        tag = re.sub(r'[\s-]+', '_', tag) # Replace spaces and hyphens with underscore
        tag = re.sub(r'[^a-z0-9_]', '', tag) # Remove non-alphanumeric or underscore
        tag = re.sub(r'_+', '_', tag) # Collapse multiple underscores
        tag = tag.strip('_') # Remove leading/trailing underscores
        return tag

    def _get_term_parent_context(self, term: str) -> List[str]:
        """Get the parent terms for additional context."""
        if term in self._parent_context_cache:
            return self._parent_context_cache[term]
            
        if not self.term_details or term not in self.term_details:
            return []
            
        term_data = self.term_details[term]
        parents = term_data.get('parents', [])
        
        # Store in cache
        self._parent_context_cache[term] = parents
        return parents

    def _calculate_cluster_separation(self, term: str, clusters: Dict[int, List[dict]]) -> float:
        """
        Calculate how well-separated the clusters are using embeddings.
        Returns a score between 0-1, where higher values indicate better separation.
        """
        if not self.embedding_model or len(clusters) <= 1:
            logger.warning(f"Cannot calculate separation for '{term}': {'No embedding model available' if not self.embedding_model else 'Insufficient clusters'}")
            return 0.0
        
        try:
            # Get centroids for each cluster
            centroids = {}
            cluster_keywords = {}
            
            # Extract text and keywords from each cluster
            for cluster_id, resources in clusters.items():
                texts = []
                
                # Get text content from resources
                for r in resources:
                    content = r.get('processed_content', '')
                    if isinstance(content, str) and content:
                        texts.append(content)
                    elif isinstance(content, list):
                        text_elements = [str(elem) for elem in content if elem]
                        if text_elements:
                            texts.append(' '.join(text_elements))
                
                if not texts:
                    continue
                
                # Extract keywords for semantic analysis
                try:
                    keywords = " ".join([' '.join(str(t).split()[:10]) for t in texts])
                    cluster_keywords[cluster_id] = keywords
                except Exception as e:
                    logger.warning(f"Error extracting keywords for '{term}' cluster {cluster_id}: {e}")
                    cluster_keywords[cluster_id] = ""
                
                # Calculate embeddings and average for centroid
                try:
                    embeddings = self.embedding_model.encode(texts)
                    centroids[cluster_id] = np.mean(embeddings, axis=0)
                except Exception as e:
                    logger.warning(f"Error calculating embeddings for '{term}' cluster {cluster_id}: {e}")
            
            # Need at least 2 centroids to calculate separation
            if len(centroids) <= 1:
                return 0.0
            
            # Calculate pairwise distances between centroids
            distances = []
            keyword_based_scores = []
            
            # Get all cluster IDs
            keys = list(centroids.keys())
            
            # Calculate distances between each pair of clusters
            for i in range(len(keys)):
                for j in range(i+1, len(keys)):
                    # Get centroids
                    c1 = centroids[keys[i]]
                    c2 = centroids[keys[j]]
                    
                    # Normalize vectors
                    c1_norm = np.linalg.norm(c1)
                    c2_norm = np.linalg.norm(c2)
                    
                    if c1_norm == 0 or c2_norm == 0:
                        continue
                    
                    # Calculate cosine distance
                    sim = np.dot(c1, c2) / (c1_norm * c2_norm)
                    sim = max(min(sim, 1.0), -1.0)  # Clamp to valid range
                    dist = 1 - sim
                    distances.append(dist)
                    
                    # Calculate keyword distinctness
                    kw1 = cluster_keywords.get(keys[i], "")
                    kw2 = cluster_keywords.get(keys[j], "")
                    if kw1 and kw2:
                        words1 = set(re.findall(r'\b\w+\b', kw1.lower()))
                        words2 = set(re.findall(r'\b\w+\b', kw2.lower()))
                        if words1 and words2:
                            common_ratio = len(words1.intersection(words2)) / min(len(words1), len(words2))
                            distinctness = 1 - common_ratio
                            keyword_based_scores.append(distinctness)
            
            if not distances:
                return 0.0
            
            # Calculate average distance
            avg_dist = sum(distances) / len(distances)
            
            # Get level-specific threshold
            level_scale = [0.65, 0.55, 0.45, 0.25][min(self.level, 3)]
            
            # Calculate score based on distance
            score = max(0, min(1, (avg_dist - level_scale) / (1 - level_scale) * 1.5))
            
            # Boost score based on keyword distinctness if needed
            if keyword_based_scores:
                keyword_distinctness = sum(keyword_based_scores) / len(keyword_based_scores)
                if keyword_distinctness > 0.7 and score < 0.5:
                    boost = (keyword_distinctness - 0.7) * 0.6
                    score = min(1.0, score + boost)
            
            # Apply minimum score for obvious different contexts
            if len(set(tag for r in clusters.values() for res in r if 'tag' in res for tag in [res['tag']])) > 1:
                if avg_dist > 0.2 and score == 0.0:
                    score = 0.1
            
            logger.info(f"Cluster separation for '{term}': {score:.2f} (avg distance: {avg_dist:.2f}, threshold: {level_scale:.2f})")
            return score
            
        except Exception as e:
            logger.error(f"Error calculating cluster separation for '{term}': {e}")
            return 0.0

    def _validate_split(self, term: str, clusters: Dict[int, List[dict]], sense_tags: Dict[int, str]) -> Tuple[bool, str]:
        """
        Determine whether a split is warranted based on multiple signals.
        Checks separation score before tag distinctness to potentially avoid LLM calls.
        """
        # 1. Basic validation
        if len(clusters) <= 1:
            return False, "Insufficient clusters"

        # 2. Check for identical tags
        unique_tags = set(sense_tags.values())
        if len(unique_tags) == 1 and not list(unique_tags)[0].startswith("sense_"):
            return False, "Identical domain tags across clusters"

        # 3. Calculate separation score
        separation_score = self._calculate_cluster_separation(term, clusters)
        
        # Get threshold based on level
        level_thresholds = {
            0: 0.5,  # College level
            1: 0.4,  # Department level
            2: 0.3,  # Research area level
            3: 0.2  # Conference topic level
        }
        threshold = level_thresholds[self.level]
        min_threshold = threshold * 0.5

        # 4. Early exit if separation too low
        if separation_score < min_threshold:
             return False, f"Insufficient separation ({separation_score:.2f} < required min {min_threshold:.2f})"

        # 5. Check tag distinctness
        tags_distinct, distinctness_reason = self._check_tags_distinctness(sense_tags.values())

        # 6. Make final decision
        if tags_distinct:
            tag_list = list(sense_tags.values())
            tags_info = ', '.join(tag_list)
            
            # Generate appropriate reason
            if distinctness_reason == "Tags represent distinct academic fields" or \
               distinctness_reason == "Using generic tags" or \
               distinctness_reason.startswith("No LLM available"):
                return True, f"Valid split: Fields distinct ({tags_info} represent distinct academic fields) and separation sufficient ({separation_score:.2f} >= {min_threshold:.2f})"
            else:
                return True, f"Valid split: Fields distinct ({distinctness_reason}) and separation sufficient ({separation_score:.2f} >= {min_threshold:.2f})"
        else:
            return False, f"Split rejected: Fields not distinct enough ({distinctness_reason})"

    def _check_tags_distinctness(self, tags: List[str]) -> Tuple[bool, str]:
        """
        Determine if the generated tags represent truly distinct academic fields.
        """
        # Convert dict_values to list if necessary
        tags_list = list(tags)
        
        # Skip check if we have generic sense_X tags (no meaningful comparison possible)
        if any(tag.startswith("sense_") for tag in tags_list):
            return True, "Using generic tags"
        
        # Convert underscore_tags to readable text for better LLM understanding
        readable_tags = [tag.replace('_', ' ') for tag in tags_list]
            
        # Check each pair of tags
        for i, tag1 in enumerate(tags_list):
            for j in range(i+1, len(tags_list)):
                tag2 = tags_list[j]
                
                # Check exact matches 
                if tag1 == tag2:
                    return False, f"Identical tags: '{tag1}' and '{tag2}'"
                
                # Check parent-child relationship
                if tag1 in tag2 or tag2 in tag1:
                    return False, f"Potential parent-child relationship: '{tag1}' and '{tag2}'"
                
                # Use LLM to determine if fields are distinct enough
                readable_tag1 = readable_tags[i]
                readable_tag2 = readable_tags[j]
                
                # Call the main field distinctness check method (which handles both structured and text responses)
                are_distinct, reason = self._check_field_distinctness_with_llm(readable_tag1, readable_tag2)
                if not are_distinct:
                    return False, reason
                elif reason and not reason.startswith("No LLM available") and not reason.startswith("Different concepts:"):
                    return True, reason
        
        # If we got here, all pairs were checked and no issues found
        return True, "Tags represent distinct academic fields"

    def _check_field_distinctness_with_llm(self, field1: str, field2: str) -> Tuple[bool, str]:
        """
        Use LLM to determine if two academic fields are distinct enough to warrant a split.
        First attempts a structured response using Pydantic model, then falls back to
        text-based response if needed.
        """
        if not self._llm or not self.using_real_llm:
            logger.info(f"No LLM available for field distinctness check: '{field1}' vs '{field2}'")
            return True, "No LLM available for distinctness check"
        
        # System prompt with clear instructions for both structured and text response
        system_prompt = """You are an expert taxonomist specializing in academic field classification. Your task is to determine if two academic fields represent GENUINELY DISTINCT CORE MEANINGS that warrant separate glossary entries.

EVALUATION CRITERIA:
1. TRUE DISTINCTION: Fields represent fundamentally different core concepts with distinct:
   - Academic foundations and intellectual traditions
   - Primary research methodologies
   - Core subject matter and phenomena of study
   - Levels of analysis (e.g., molecular vs. societal)

2. FALSE DISTINCTION (same field): Fields represent the same or highly related concepts with:
   - Different aspects/applications of the same concept
   - Different methodological approaches within the same domain
   - Hierarchical relationship (parent-child, subfield relationship)
   - Different terminology for the same underlying concepts
   - Overlapping boundaries with shared core objectives

IMPORTANT: Provide your answer in this EXACT FORMAT:
```
EXPLANATION: [Provide a SINGLE SENTENCE explanation for the verdict, comparing the fields based on core concepts, methodologies, and relationship (e.g., subfield, overlap, distinct domains).]
VERDICT: [DISTINCT or NOT_DISTINCT]
```

DISTINCT means the fields represent fundamentally different core meanings requiring separate entries.
NOT_DISTINCT means the fields represent different aspects/applications of the same underlying concept."""

        # User prompt with examples
        user_prompt = f"""Compare these two academic fields and determine if they represent GENUINELY DIFFERENT CORE MEANINGS:

FIELD 1: {field1}
FIELD 2: {field2}

EXAMPLES OF TRULY DISTINCT FIELDS (should be separate entries):
- "cell biology" vs. "prison cell studies" (different core concepts in biology vs. sociology)
- "mathematical programming" vs. "computer programming" (one is operations research, the other software development)
- "developmental psychology" vs. "developmental economics" (same qualifier but different domains)
- "network (sociology)" vs. "network theory in computer science" (same term, different domains)
- "signal processing" vs. "food processing" (completely unrelated domains that share a term)

EXAMPLES OF NON-DISTINCT FIELDS (should be single entries):
- "computational linguistics" vs. "natural language processing" (overlapping fields with shared objectives)
- "machine learning theory" vs. "machine learning applications" (theory vs. practice of same field)
- "cognitive neuroscience" vs. "behavioral neuroscience" (subfields of neuroscience)
- "quantum mechanics" vs. "quantum physics" (essentially the same field with different naming)
- "psychological stress" vs. "mental stress" (different terminology for same concept)

REMEMBER: Focus on core meaning, not just terminology. Subfields of the same discipline usually warrant a SINGLE entry with qualifiers, while truly different domains warrant SEPARATE entries.

Apply the evaluation criteria and provide your determination in the specified format."""

        try:
            # First attempt: Try structured response with Pydantic model
            try:
                response = self._llm.infer(
                    prompt=user_prompt, 
                    system_prompt=system_prompt,
                    temperature=0.1,
                    response_model=FieldDistinctnessAnalysis
                )
                
                if response and hasattr(response, 'text'):
                    # Extract the structured response
                    analysis_result = response.text # Rename for clarity
                    
                    # Make the decision based on the verdict
                    if analysis_result.verdict == "DISTINCT":
                        return True, analysis_result.explanation
                    else:
                        return False, analysis_result.explanation
            
            except Exception as e:
                # If structured response fails, log it and continue to text-based approach
                logger.warning(f"Structured response failed: {e}. Falling back to text response.")
            
            # Second attempt: Use text-based response as fallback
            response = self._llm.infer(
                prompt=user_prompt, 
                system_prompt=system_prompt,
                temperature=0.1
            )
            
            if response and hasattr(response, 'text'):
                answer_text = response.text.strip()
                
                # Extract decision using regex patterns
                verdict_match = re.search(r'VERDICT:\s*(DISTINCT|NOT_DISTINCT)', answer_text, re.IGNORECASE)
                explanation_match = re.search(r'EXPLANATION:\s*(.+?)(?=\nVERDICT:|$)', answer_text, re.DOTALL) # Look for EXPLANATION
                
                # Use the extracted explanation
                reason = explanation_match.group(1).strip() if explanation_match else f"Comparison between '{field1}' and '{field2}'"
                
                if verdict_match:
                    decision = verdict_match.group(1).upper()
                    
                    if decision == "DISTINCT":
                        return True, reason
                    elif decision == "NOT_DISTINCT":
                        return False, reason
                
                # Fallback check for keywords (less reliable now)
                answer_lower = answer_text.lower()
                if "verdict: not_distinct" in answer_lower or "verdict: not distinct" in answer_lower:
                    return False, reason # Return extracted reason even on keyword fallback
                elif "verdict: distinct" in answer_lower:
                    return True, reason # Return extracted reason even on keyword fallback
            
            # Default to conservative approach
            logger.warning(f"Ambiguous LLM response for field distinctness check")
            return False, "Ambiguous determination, defaulting to not distinct"
                
        except Exception as e:
            logger.error(f"Error in field distinctness check: {e}")
            return False, f"Error in distinctness check: {str(e)[:50]}"

    def _extract_domain_keywords_with_tfidf(self, term: str, cluster_resources: list[dict], all_resources: list[dict]) -> str:
        """
        Use TF-IDF to extract domain-specific keywords for a cluster.
        
        Args:
            term: The term being processed
            cluster_resources: Resources for a specific cluster
            all_resources: All resources for the term across clusters
            
        Returns:
            A domain tag string or None if extraction failed
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Prepare corpus: all snippets for the term
            all_snippets = [r.get('processed_content', '') for r in all_resources if r.get('processed_content')]
            cluster_snippets = [r.get('processed_content', '') for r in cluster_resources if r.get('processed_content')]
            
            if not all_snippets or not cluster_snippets:
                return None
                
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_df=0.85,           # Ignore terms that appear in >85% of documents
                min_df=2,              # Ignore terms that appear in fewer than 2 documents
                stop_words='english',  # Remove English stopwords
                max_features=1000,     # Limit vocabulary size
                ngram_range=(1, 2)     # Consider unigrams and bigrams
            )
            
            # Fit vectorizer on all snippets
            tfidf_matrix = vectorizer.fit_transform(all_snippets)
            
            # Transform cluster snippets
            cluster_matrix = vectorizer.transform(cluster_snippets)
            
            # Get average TF-IDF scores for cluster
            cluster_avg = cluster_matrix.mean(axis=0)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Convert to array and get top keywords
            cluster_avg_array = cluster_avg.toarray().flatten()
            top_indices = cluster_avg_array.argsort()[-10:][::-1]  # Get top 10 keywords
            top_keywords = [feature_names[i] for i in top_indices]
            
            # Filter for domain-specific keywords
            # This is a simple domain keyword list - in a real implementation, 
            # you might use a more comprehensive academic field taxonomy
            domain_keywords = [
                "biology", "physics", "chemistry", "mathematics", "computer science",
                "psychology", "sociology", "economics", "philosophy", "history",
                "engineering", "medicine", "law", "art", "literature", "music",
                "politics", "education", "environment", "technology", "data", 
                "research", "analysis", "theory", "study", "science", "humanities",
                "social", "natural", "applied", "theoretical", "computational",
                "quantum", "molecular", "neural", "cognitive", "behavioral",
                "statistical", "algebraic", "geometric", "evolutionary", "developmental",
                "clinical", "industrial", "environmental", "organic", "inorganic"
            ]
            
            # Find domain keywords in top keywords
            found_domains = []
            for keyword in top_keywords:
                if keyword in domain_keywords or any(kw in keyword for kw in domain_keywords):
                    found_domains.append(keyword)
                    
            if found_domains:
                # Use first domain keyword found
                tag = found_domains[0]
                logger.info(f"TF-IDF extracted domain tag '{tag}' for term '{term}'")
                return tag
                
            # If no domain keywords found, use top keyword
            if top_keywords:
                tag = top_keywords[0]
                logger.info(f"TF-IDF using top keyword '{tag}' for term '{term}'")
                return tag
                
            return None
                
        except Exception as e:
            logger.error(f"Error in TF-IDF keyword extraction for '{term}': {e}")
            return None

    def _extract_tag_from_parent_context(self, term: str, cluster_resources: list[dict]) -> str:
        """
        Extract a domain tag based on the parent context of the resources in a cluster.
        
        Args:
            term: The term being processed
            cluster_resources: Resources for a specific cluster
            
        Returns:
            A domain tag string or None if extraction failed
        """
        try:
            # Get the parent terms for the current term
            parents = self._get_term_parent_context(term)
            if not parents:
                return None
                
            # Track which resources are associated with which parents
            parent_resource_counts = defaultdict(int)
            
            # For each resource in the cluster, try to find its original parent
            for resource in cluster_resources:
                url = resource.get('url', '')
                if not url:
                    continue
                    
                # Look through parents to see which one's resources include this URL
                for parent in parents:
                    if parent not in self.term_details:
                        continue
                        
                    parent_resources = self.term_details[parent].get('resources', [])
                    parent_urls = [pr.get('url', '') for pr in parent_resources]
                    
                    if url in parent_urls:
                        parent_resource_counts[parent] += 1
                        
            # Find the parent with the most resources in this cluster
            if parent_resource_counts:
                dominant_parent = max(parent_resource_counts.items(), key=lambda x: x[1])[0]
                
                # Check if this is a significant majority (>60% of resources)
                total_matched = sum(parent_resource_counts.values())
                if total_matched > 0 and parent_resource_counts[dominant_parent] / total_matched >= 0.6:
                    # Get the level of the parent term
                    parent_level = self._get_term_level(dominant_parent)
                    
                    # For lower-level parents, use the parent term directly (it's more specific)
                    if parent_level is not None and parent_level > 0:
                        logger.info(f"Using dominant parent '{dominant_parent}' as tag for '{term}'")
                        return dominant_parent
                    
                    # For top-level parents, we might need the grandparent context
                    # since level 0 terms are too broad (e.g., "science", "humanities")
                    if parent_level == 0:
                        # Try to get a more specific context from the term itself
                        if "_" in term:  # If term has multiple words
                            parts = term.split("_")
                            # Use the dominant parent + qualifier from the term
                            context_tag = f"{dominant_parent}_{parts[-1]}"
                            logger.info(f"Created context tag '{context_tag}' for '{term}'")
                            return context_tag
                            
                        # Just use the dominant parent if we can't create a more specific tag
                        return dominant_parent
            
            return None
                
        except Exception as e:
            logger.error(f"Error in parent context analysis for '{term}': {e}")
            return None

    def _generate_sense_tags(self, term: str, grouped_resources: dict[int, list[dict]]) -> dict[int, str]:
        """
        Generates a disambiguating tag for each sense (cluster) of a term.
        Uses a multi-layered approach (LLM -> TF-IDF -> Parent Context -> Generic).

        Args:
            term: The term string.
            grouped_resources: Dictionary mapping cluster ID to list of resources for that cluster.

        Returns:
            Dictionary mapping cluster ID to its generated disambiguator tag string.
        """
        sense_tags = {}
        logger.debug(f"Generating tags for term '{term}' with {len(grouped_resources)} clusters.")
        
        # Get parent context for additional information
        parent_context = self._get_term_parent_context(term)
        
        # Get all resources for TF-IDF processing
        all_resources = []
        for resources in grouped_resources.values():
            all_resources.extend(resources)

        for cluster_id, resources in grouped_resources.items():
            tag = None
            resource_snippets = [r.get('processed_content', '') for r in resources if r.get('processed_content')]
            resource_snippets = [s for s in resource_snippets if len(s) > 10] # Ensure non-trivial snippets

            # 1. Attempt LLM Tagging (Real or Simulated)
            if self.use_llm_for_tags:
                tag = self._call_llm_for_tag(term, cluster_id, resource_snippets, parent_context)
                    
                if tag:
                    tag = self._standardize_tag(tag)
                    logger.info(f"Generated tag '{tag}' for term '{term}', cluster {cluster_id}")

            # 2. Attempt TF-IDF Tagging
            if not tag:
                logger.debug(f"LLM tagging failed/disabled for '{term}' cluster {cluster_id}. Trying TF-IDF.")
                tfidf_tag = self._extract_domain_keywords_with_tfidf(term, resources, all_resources)
                if tfidf_tag:
                    tag = self._standardize_tag(tfidf_tag)
                    logger.info(f"TF-IDF generated tag '{tag}' for term '{term}', cluster {cluster_id}")

            # 3. Attempt Parent Context Tagging
            if not tag:
                logger.debug(f"TF-IDF tagging failed for '{term}' cluster {cluster_id}. Trying parent context.")
                context_tag = self._extract_tag_from_parent_context(term, resources)
                if context_tag:
                    tag = self._standardize_tag(context_tag)
                    logger.info(f"Parent context generated tag '{tag}' for term '{term}', cluster {cluster_id}")

            # 4. Generic Fallback
            if not tag:
                logger.warning(f"All tagging methods failed for '{term}' cluster {cluster_id}. Using generic tag.")
                tag = f"sense_{cluster_id + 1}"

            sense_tags[cluster_id] = tag
            logger.debug(f"Assigned tag '{tag}' to cluster {cluster_id} for term '{term}'")

        return sense_tags

    def _group_resources_by_cluster(self, term: str) -> dict[int, list[dict]]:
        """
        Groups the resources of a candidate term based on their pre-calculated cluster labels.
        If detailed cluster information is available (from comprehensive format),
        uses that directly instead of reconstructing clusters.

        Args:
            term: The candidate term string.

        Returns:
            A dictionary mapping cluster ID (int >= 0) to a list of resource objects
            belonging to that cluster. Returns empty dict if term not found, not a candidate,
            has no resources, or no cluster results.
        """
        if term not in self.candidate_terms:
            logger.debug(f"Term '{term}' is not in the candidate list. Skipping grouping.")
            return {}
        
        # Check if we have detailed cluster info for this term (from comprehensive format)
        if self.detailed_cluster_info and term in self.detailed_cluster_info:
            detailed_clusters = self.detailed_cluster_info[term]
            
            # Convert string cluster IDs to ints and filter out noise cluster (-1)
            grouped_resources = {}
            for cluster_id_str, resources in detailed_clusters.items():
                cluster_id = int(cluster_id_str)
                if cluster_id >= 0:  # Skip noise cluster
                    grouped_resources[cluster_id] = resources
                    
            logger.debug(f"Using detailed cluster info for '{term}': Found {len(grouped_resources)} clusters "
                       f"with {[len(resources) for resources in grouped_resources.values()]} resources each.")
            return grouped_resources
            
        # Fall back to traditional method if no detailed info available
        if not self.term_details:
             logger.warning("Term details not loaded. Cannot group resources.")
             return {}
        if term not in self.term_details:
            logger.warning(f"Term '{term}' not found in hierarchy details. Skipping grouping.")
            return {}

        term_data = self.term_details[term]
        resources = term_data.get('resources', [])
        if not resources:
            logger.debug(f"Term '{term}' has no resources. Skipping grouping.")
            return {}

        # Retrieve the cluster labels for this term's resources
        resource_cluster_labels = self.cluster_results.get(term)
        if resource_cluster_labels is None:
            logger.warning(f"No cluster results found for candidate term '{term}'. Skipping grouping.")
            return {}

        # Basic check: ensure number of labels matches number of resources
        # Note: This assumes the detector only provides labels for resources it processed.
        # A more robust check might be needed depending on how cluster_results is generated.
        if len(resource_cluster_labels) != len(resources):
             logger.warning(f"Mismatch between resource count ({len(resources)}) "
                             f"and cluster label count ({len(resource_cluster_labels)}) for term '{term}'. Skipping grouping.")
             # TODO: Decide how to handle mismatches if they can occur.
             # For now, we skip. Could try aligning based on processed indices if available.
             return {}

        grouped_resources = defaultdict(list)
        noise_resources = [] # Keep track of noise points (-1) separately for now

        for i, resource in enumerate(resources):
            cluster_label = resource_cluster_labels[i]
            if cluster_label == -1:
                # Handle noise points (currently just collected, not added to core clusters)
                noise_resources.append(resource)
            elif cluster_label >= 0:
                # Add resource to its corresponding cluster group
                grouped_resources[cluster_label].append(resource)


        # Log summary for the term
        num_core_clusters = len(grouped_resources)
        num_noise_points = len(noise_resources)
        logger.debug(f"Grouped resources for '{term}': Found {num_core_clusters} core clusters "
                     f"({[len(v) for v in grouped_resources.values()]} resources each) and {num_noise_points} noise points.")

        # Apply level-specific filtering
        # At higher levels (0-1), require more evidence (more clusters, larger separation)
        if self.level <= 1 and num_core_clusters <= 1:
            logger.debug(f"Term '{term}' at level {self.level} needs stronger evidence for splitting. Skipping.")
            return {}

        return dict(grouped_resources) # Convert back to regular dict

    def generate_split_proposals(self) -> tuple[list[dict], list[dict]]:
        """
        Generates proposals for splitting ambiguous terms into distinct senses.
        Includes resource grouping and sense tagging.
        
        Returns:
            Tuple of (accepted_proposals, rejected_proposals)
        """
        if not self._load_hierarchy():
            logger.error("Failed to load hierarchy. Cannot generate proposals.")
            return [], []

        # Filter candidate terms to only include terms at the current level
        level_filtered_candidates = self._filter_candidate_terms_by_level()
        if not level_filtered_candidates:
            logger.warning(f"No candidate terms found at level {self.level}. Cannot generate proposals.")
            return [], []

        accepted_proposals = []
        rejected_proposals = []
        logger.info(f"Generating split proposals for {len(level_filtered_candidates)} candidate terms at level {self.level}...")

        processed_candidates = 0
        for term in level_filtered_candidates:
            processed_candidates += 1
            if processed_candidates % 50 == 0:
                 logger.info(f"Processing candidate {processed_candidates}/{len(level_filtered_candidates)}: '{term}'")
            # Step 1: Group resources by cluster
            grouped_resources = self._group_resources_by_cluster(term)

            # Add to rejected proposals if term wasn't clustered into multiple groups
            if not grouped_resources or len(grouped_resources) <= 1:
                rejection_reason = "Insufficient clusters" if not grouped_resources else "Only one cluster found"
                rejected_proposals.append({
                    "original_term": term,
                    "level": self.level,
                    "rejection_reason": rejection_reason,
                    "cluster_count": len(grouped_resources) if grouped_resources else 0
                })
                continue

            # Step 2: Generate Meaningful Tags
            sense_tags = self._generate_sense_tags(term, grouped_resources)

            # Step 3: Validate the split is justified
            should_split, split_reason = self._validate_split(term, grouped_resources, sense_tags)
            
            # Create a proposal structure regardless of acceptance/rejection
            proposal = {
                "original_term": term,
                "level": self.level,
                "cluster_count": len(grouped_resources),
                "proposed_senses": []
            }
            
            # Add detailed sense information
            for cluster_id, resources in grouped_resources.items():
                # Use the generated tag, falling back if somehow missing
                tag = sense_tags.get(cluster_id, f"sense_{cluster_id + 1}")
                sense_proposal = {
                    "sense_tag": tag,
                    "cluster_id": cluster_id,
                    "resource_count": len(resources),
                    "sample_resources": [
                        {
                            "url": r.get("url", ""),
                            "content": r.get("processed_content", "")[:200] + "..." if r.get("processed_content", "") else ""
                        } for r in resources[:3] # Include up to 3 sample resources
                    ]
                }
                proposal["proposed_senses"].append(sense_proposal)
            
            if should_split:
                # Add the reason for acceptance and add to accepted list
                proposal["split_reason"] = split_reason
                accepted_proposals.append(proposal)
                logger.info(f"Generated proposal for '{term}' with {len(proposal['proposed_senses'])} senses: Tags={list(sense_tags.values())}")
            else:
                # Add the reason for rejection and add to rejected list
                proposal["rejection_reason"] = split_reason
                rejected_proposals.append(proposal)
                logger.info(f"Rejected split for '{term}': {split_reason}")

        logger.info(f"Generated {len(accepted_proposals)} accepted split proposals and {len(rejected_proposals)} rejected cases (from {len(level_filtered_candidates)} candidates at level {self.level}).")
        return accepted_proposals, rejected_proposals

    def save_proposals(self, accepted_proposals: list[dict], rejected_proposals: list[dict], filename: Optional[str] = None) -> str:
        """
        Saves split proposals to a file.
        
        Args:
            accepted_proposals: List of accepted split proposals.
            rejected_proposals: List of rejected split cases.
            filename: Optional filename. If not provided, a default name will be used.
            
        Returns:
            Path to the saved file.
        """
        if not filename:
            # Use a simple default name without timestamp
            filename = f"split_proposals_level{self.level}.json"
            
        filepath = os.path.join(self.output_dir, filename)
        
        # Prepare the output data structure
        output_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "level": self.level,
            "parameters": {
                "embedding_model": self.embedding_model_name,
                "use_llm": self.use_llm_for_tags,
                "llm_provider": self.llm_provider if self.use_llm_for_tags else None,
                "llm_model": self.llm_model if self.use_llm_for_tags else None,
                "clustering": {
                    "algorithm": "dbscan",  # Currently only DBSCAN is used for final splitting
                    "eps": self.level_params["eps"],
                    "min_samples": self.level_params["min_samples"]
                }
            },
            "accepted_proposals": accepted_proposals,
            "rejected_proposals": rejected_proposals,
            "counts": {
                "accepted": len(accepted_proposals),
                "rejected": len(rejected_proposals),
                "total": len(accepted_proposals) + len(rejected_proposals)
            }
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"[SenseSplitter] Saved proposals to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"[SenseSplitter] Error saving proposals to {filepath}: {e}")
            return None

    def _load_cluster_results_from_file(self, cluster_results_file: str) -> bool:
        """
        Loads pre-computed cluster results from a file.
        Supports both standard results and comprehensive cluster details format.
        
        Args:
            cluster_results_file: Path to the cluster results JSON file
            
        Returns:
            True if loading was successful, False otherwise
        """
        try:
            with open(cluster_results_file, 'r') as f:
                data = json.load(f)
            
            # Determine the format - comprehensive format has "term_clusters"
            if "term_clusters" in data:
                logger.info(f"Detected comprehensive cluster details format in {cluster_results_file}")
                return self._process_comprehensive_cluster_details(data)
            else:
                logger.info(f"Detected standard cluster results format in {cluster_results_file}")
                
                # Update candidate terms list
                # We will filter by level later
                if self.candidate_terms is None or len(self.candidate_terms) == 0:
                    self.candidate_terms = data.get("ambiguous_terms", [])
                    logger.info(f"Loaded {len(self.candidate_terms)} candidate terms from {cluster_results_file}")
                
                # Update cluster results dict
                self.cluster_results = data.get("cluster_results", {})
                if not self.cluster_results:
                    logger.error(f"No cluster results found in {cluster_results_file}")
                    return False
                
                # Store metrics for potential use in split validation
                self.cluster_metrics = data.get("metrics", {})
                
                logger.info(f"Successfully loaded cluster results for {len(self.cluster_results)} terms from {cluster_results_file}")
                return True
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading cluster results from {cluster_results_file}: {e}")
            return False

    def _process_comprehensive_cluster_details(self, data: dict) -> bool:
        """
        Process comprehensive cluster details format.
        This format includes full resource details grouped by cluster,
        providing more information for the splitter.
        
        Args:
            data: The loaded JSON data in comprehensive format
            
        Returns:
            True if processing was successful, False otherwise
        """
        if not data or "term_clusters" not in data:
            logger.error("Invalid comprehensive cluster details format")
            return False
            
        term_clusters = data["term_clusters"]
        
        # Extract candidate terms and their cluster labels
        all_candidate_terms = list(term_clusters.keys())
        
        # Initialize storage
        cluster_results = {}
        cluster_details = {}
        detailed_metrics = {}
        
        # Process each term's cluster information
        for term, term_data in term_clusters.items():
            term_level = term_data.get("level")
            clusters = term_data.get("clusters", {})
            
            # Skip terms with no clusters or just noise
            valid_clusters = {cid: resources for cid, resources in clusters.items() if cid != "-1"}
            if not valid_clusters or len(valid_clusters) < 2:
                continue
                
            # Store detailed cluster information (all resources by cluster)
            cluster_details[term] = valid_clusters
            
            # Store metrics
            detailed_metrics[term] = term_data.get("metrics", {})
            
            # Extract and store cluster labels in the format expected by SenseSplitter
            # We need to recreate the full label array matching the original resource order
            all_resources = []
            resource_labels = {}
            
            # Collect resources from all clusters (including noise)
            for cluster_id, resources in clusters.items():
                for res in resources:
                    # Create a composite key from URL and content fragment to identify the resource
                    # This is needed since the original order is not preserved in the JSON
                    url = res.get("url", "")
                    content_fragment = res.get("processed_content", "")[:50]
                    res_key = f"{url}:{content_fragment}"
                    
                    # Store the resource and its cluster label
                    resource_labels[res_key] = int(cluster_id)
                    all_resources.append((res_key, res))
            
            # If we can't determine the original order, we create a new array
            # with the cluster labels in the order of the resources
            cluster_label_array = [resource_labels[res[0]] for res in all_resources]
            cluster_results[term] = cluster_label_array
        
        # Update instance variables
        self.candidate_terms = all_candidate_terms
        self.cluster_results = cluster_results
        self.cluster_metrics = detailed_metrics
        
        # Store the extra cluster details for enhanced processing
        self.detailed_cluster_info = cluster_details
        
        logger.info(f"Successfully processed comprehensive cluster details for {len(cluster_results)} terms")
        return True

    def run(self, save_output: bool = True, output_filename: Optional[str] = None) -> tuple[list[dict], list[dict], Optional[str]]:
        """
        Main execution method to generate and save split proposals.
        
        Two ways to use this class:
        1. Initialize with pre-computed cluster results and candidate terms
        2. Initialize with empty lists, then load a comprehensive cluster details file 
           using _load_cluster_results_from_file() method
        
        Example using comprehensive cluster details:
            splitter = SenseSplitter(
                hierarchy_file_path="data/hierarchy.json",
                candidate_terms_list=[],  # Will be populated from the file
                cluster_results={},       # Will be populated from the file
                level=2
            )
            splitter._load_cluster_results_from_file("data/ambiguity_detection_results/comprehensive_cluster_details_20250410_123456.json")
            accepted, rejected, output_path = splitter.run()
        
        Args:
            save_output: Whether to save the results to a file.
            output_filename: Optional filename. If None, a timestamped name will be used.
            
        Returns:
            Tuple of (accepted_proposals, rejected_proposals, output_path)
            where output_path is the path to the saved file if save_output is True, 
            otherwise None.
        """
        # Generate split proposals
        accepted_proposals, rejected_proposals = self.generate_split_proposals()
        
        output_path = None
        if save_output and (accepted_proposals or rejected_proposals):
            output_path = self.save_proposals(accepted_proposals, rejected_proposals, output_filename)
            
        return accepted_proposals, rejected_proposals, output_path


# --- Example Usage ---
if __name__ == '__main__':
    # Setting up paths
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    default_hierarchy_path = os.path.join(repo_root, "data", "hierarchy.json")
    
    # Directories for input and output
    detection_results_dir = os.path.join(repo_root, "data", "ambiguity_detection_results")
    output_dir = os.path.join(repo_root, "data", "sense_disambiguation_results")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find the latest cluster results file
    cluster_results_files = glob.glob(os.path.join(detection_results_dir, "cluster_results_eps*.json"))
    
    if not cluster_results_files:
        logger.error("No cluster results files found. Please run detector.py first.")
        print("No cluster results files found. Please run detector.py first to generate cluster results.")
        print(f"Expected directory: {detection_results_dir}")
        sys.exit(1)
    
    # Use the most recent file by default, or a specific epsilon value if preferred
    # For example, to use eps=0.4 results:
    eps02_files = [f for f in cluster_results_files if "eps0.2" in f]
    
    if eps02_files:
        cluster_results_file = max(eps02_files, key=os.path.getmtime)
    else:
        # Fallback to most recent
        cluster_results_file = max(cluster_results_files, key=os.path.getmtime)
    
    logger.info(f"Using cluster results from: {cluster_results_file}")
    print(f"Using cluster results from: {cluster_results_file}")
    
    # Run Splitter for each hierarchy level
    try:
        # Process each level with appropriate parameters
        for level in range(4):  # Levels 0-3
            logger.info(f"Processing terms at level {level}...")
            print(f"\nProcessing terms at level {level}...")
            
            # Initialize a SenseSplitter with empty candidates, we'll load them from file
            splitter = SenseSplitter(
                hierarchy_file_path=default_hierarchy_path,
                candidate_terms_list=[],  # Will be populated from loaded results
                cluster_results={},       # Will be populated from loaded results
                use_llm_for_tags=True,
                llm_provider=Provider.OPENAI,
                level=level,
                output_dir=output_dir
            )
            
            # Load cluster results from the detector output
            if not splitter._load_cluster_results_from_file(cluster_results_file):
                logger.error(f"Failed to load cluster results for level {level}. Skipping.")
                print(f"Failed to load cluster results for level {level}. Skipping.")
                continue
            
            # Run and save the results
            accepted_proposals, rejected_proposals, output_filepath = splitter.run(save_output=True)
            
            if accepted_proposals or rejected_proposals:
                print(f"Generated {len(accepted_proposals)} accepted and {len(rejected_proposals)} rejected split proposals for level {level}.")
                print(f"Results saved to: {output_filepath}")
            else:
                print(f"No split proposals generated for level {level}.")
                
    except Exception as e:
        logger.error(f"Error running SenseSplitter: {e}")
        print(f"Error running SenseSplitter: {e}")
        import traceback
        logger.error(traceback.format_exc()) 