"""
Parent context-based detector for identifying ambiguous terms.

This module implements the ParentContextDetector class that analyzes
the hierarchy to find terms with divergent parent contexts.
"""

import json
import glob
import os
import logging
import datetime # Added for timestamp
from collections import defaultdict
from typing import Optional, List, Set, Tuple, Dict, Any

# Import utility for numpy conversion just in case it's needed later, although unlikely here
from .utils import convert_numpy_types

# Setup logger if not already configured by parent module
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ParentContextDetector:
    """Detects potentially ambiguous terms by analyzing parent contexts in the hierarchy."""

    def __init__(self, hierarchy_file_path: str, final_term_files_pattern: str, level: Optional[int] = None):
        """Initializes the detector.

        Args:
            hierarchy_file_path: Path to the hierarchy.json file.
            final_term_files_pattern: Glob pattern for lv*_final.txt files.
            level: Optional hierarchy level (0-3) for targeted analysis.
        """
        self.hierarchy_file_path = hierarchy_file_path
        self.final_term_files_pattern = final_term_files_pattern
        self.level = level
        self.hierarchy_data = None
        self.canonical_terms = set()
        self.term_details = None
        self._ancestor_cache = {}
        self._loaded = False
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                     "sense_disambiguation/data", "ambiguity_detection_results", "parent_context") # Default subdir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Store detailed results internally
        self.detailed_results: Dict[str, Dict[str, Any]] = {}

    def _load_data(self):
        """Loads the hierarchy data and canonical term lists."""
        if self._loaded:
            return True

        # Load hierarchy
        logging.info(f"[ParentContextDetector] Loading hierarchy from {self.hierarchy_file_path}...")
        try:
            with open(self.hierarchy_file_path, 'r') as f:
                self.hierarchy_data = json.load(f)
            self.term_details = self.hierarchy_data.get('terms', {})
            if not self.term_details:
                logging.warning("[ParentContextDetector] Hierarchy file loaded, but 'terms' dictionary is missing or empty.")
                return False
            logging.info(f"[ParentContextDetector] Loaded {len(self.term_details)} terms from hierarchy.")
        except FileNotFoundError:
            logging.error(f"[ParentContextDetector] Hierarchy file not found: {self.hierarchy_file_path}")
            return False
        except json.JSONDecodeError:
            logging.error(f"[ParentContextDetector] Error decoding JSON from {self.hierarchy_file_path}")
            return False

        # Load canonical terms
        logging.info(f"[ParentContextDetector] Loading canonical terms from pattern: {self.final_term_files_pattern}")
        final_term_files = glob.glob(self.final_term_files_pattern)
        if not final_term_files:
            logging.warning(f"[ParentContextDetector] No files found matching pattern: {self.final_term_files_pattern}")

        for file_path in final_term_files:
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        term = line.strip()
                        if term:
                            self.canonical_terms.add(term)
            except FileNotFoundError:
                logging.warning(f"[ParentContextDetector] Final term file not found during glob: {file_path}")
            except Exception as e:
                logging.error(f"[ParentContextDetector] Error reading final term file {file_path}: {e}")

        logging.info(f"[ParentContextDetector] Loaded {len(self.canonical_terms)} unique canonical terms.")
        if not self.canonical_terms:
             logging.warning("[ParentContextDetector] No canonical terms were loaded. Ambiguity detection based on canonical terms will not yield results.")

        self._loaded = True
        return True

    def _get_ancestors(self, term: str, visited: set) -> tuple[str | None, str | None]:
        """Finds the Level 0 and Level 1 ancestors for a given term.

        Uses caching and prevents infinite loops.

        Args:
            term: The term string to find ancestors for.
            visited: A set of terms already visited in the current traversal path.

        Returns:
            A tuple (l0_ancestor, l1_ancestor). Returns (None, None) if not found or error.
        """
        if term in self._ancestor_cache:
            return self._ancestor_cache[term]

        if term not in self.term_details:
            # logging.debug(f"Term '{term}' not found in hierarchy details.")
            return None, None

        if term in visited:
            logging.warning(f"[ParentContextDetector] Cycle detected involving term '{term}'. Stopping traversal.")
            return None, None

        visited.add(term)

        term_data = self.term_details[term]
        level = term_data.get('level')
        parents = term_data.get('parents', [])

        l0_ancestor = None
        l1_ancestor = None

        if level == 0:
            l0_ancestor = term
        elif level == 1:
            l1_ancestor = term
            # Try to find L0 parent
            for parent in parents:
                parent_l0, _ = self._get_ancestors(parent, visited.copy())
                if parent_l0:
                    l0_ancestor = parent_l0
                    break # Assume first L0 parent found is the one

        elif level is not None and level > 1:
            # Look through parents
            for parent in parents:
                parent_l0, parent_l1 = self._get_ancestors(parent, visited.copy())
                if parent_l0 and not l0_ancestor:
                    l0_ancestor = parent_l0
                if parent_l1 and not l1_ancestor:
                    l1_ancestor = parent_l1
                # If we found both, we can potentially stop early
                if l0_ancestor and l1_ancestor:
                    break

        visited.remove(term)
        self._ancestor_cache[term] = (l0_ancestor, l1_ancestor)
        return l0_ancestor, l1_ancestor

    def detect_ambiguous_terms(self) -> list[str]:
        """Performs the ambiguity detection based on parent contexts.

        Returns:
            A list of canonical term strings identified as potentially ambiguous.
        """
        if not self._load_data():
            logging.error("[ParentContextDetector] Failed to load necessary data. Aborting detection.")
            return []

        if not self.term_details or not self.canonical_terms:
            logging.warning("[ParentContextDetector] Missing term details or canonical terms. Cannot perform detection.")
            return []

        level_info = f" for level {self.level}" if self.level is not None else ""
        logging.info(f"[ParentContextDetector] Starting ambiguity detection{level_info}...")
        processed_count = 0
        all_ambiguous_terms_details = {} # Store details for all levels before filtering

        for term, term_data in self.term_details.items():
            processed_count += 1
            if processed_count % 5000 == 0: # Log less frequently maybe
                logging.info(f"[ParentContextDetector] Processed {processed_count}/{len(self.term_details)} terms...")

            # 1. Check if canonical
            if term not in self.canonical_terms:
                continue

            # 2. Check for multiple parents
            parents = term_data.get('parents', [])
            if len(parents) <= 1:
                continue

            # 3. Collect parent contexts, handling potential None values
            parent_contexts = set()
            parent_details_for_term = {} # Collect parent L0/L1 info
            for parent_term in parents:
                l0_anc, l1_anc = self._get_ancestors(parent_term, set())
                # Handle None before converting to list for details
                l0_list = [l0_anc] if l0_anc is not None else []
                l1_list = [l1_anc] if l1_anc is not None else []
                parent_details_for_term[parent_term] = {"l0": l0_list, "l1": l1_list}
                
                # Use empty frozenset if ancestor is None for context comparison
                l0_fset = frozenset([l0_anc]) if l0_anc is not None else frozenset()
                l1_fset = frozenset([l1_anc]) if l1_anc is not None else frozenset()
                
                # Only add if at least one ancestor was found
                if l0_fset or l1_fset: 
                    parent_contexts.add((l0_fset, l1_fset))
            
            # 4. Proceed with ambiguity check using the collected contexts
            if len(parent_contexts) <= 1:
                continue # All parents map to the same context or couldn't be traced

            # Calculate unique non-empty L0 and L1 sets across all parent contexts
            unique_l0 = {l0 for ctx in parent_contexts for l0 in ctx[0]} 
            unique_l1 = {l1 for ctx in parent_contexts for l1 in ctx[1]}
            
            is_ambiguous = False
            # Ambiguous if more than one distinct L0 ancestor found
            if len(unique_l0) > 1:
                is_ambiguous = True
            # Or, if L0 is consistent (or absent), but more than one L1 is found
            elif len(unique_l0) <= 1 and len(unique_l1) > 1:
                # Check if all non-empty L0 sets are the same
                non_empty_l0_sets = [ctx[0] for ctx in parent_contexts if ctx[0]] # Get non-empty frozensets
                if len(set(non_empty_l0_sets)) <= 1:
                    is_ambiguous = True
            
            if is_ambiguous:
                # Store details
                all_ambiguous_terms_details[term] = {
                    "level": term_data.get('level'),
                    "parent_count": len(parents),
                    "ancestor_contexts": [[list(pair[0]), list(pair[1])] for pair in sorted(list(parent_contexts), key=lambda x: (str(x[0]), str(x[1])))],
                    "parents_details": parent_details_for_term 
                }

        # Filter by level and store final detailed results
        filtered_ambiguous_terms = []
        self.detailed_results = {} # Clear previous detailed results for this instance
        for term, details in all_ambiguous_terms_details.items():
            term_level_actual = details.get('level')
            if self.level is None or term_level_actual == self.level:
                filtered_ambiguous_terms.append(term)
                self.detailed_results[term] = details

        filtered_ambiguous_terms.sort()

        logging.info(f"[ParentContextDetector] Found {len(all_ambiguous_terms_details)} potentially ambiguous terms across all levels")
        logging.info(f"[ParentContextDetector] After level filtering, {len(filtered_ambiguous_terms)} terms remain at level {self.level}")
        logging.info(f"[ParentContextDetector] Ambiguity detection complete for level {self.level}. Found {len(filtered_ambiguous_terms)} potentially ambiguous terms.")
        
        return filtered_ambiguous_terms

    def save_detailed_results(self, filename: Optional[str] = None) -> str:
        """
        Saves the detailed ambiguity results (including ancestor contexts) to a JSON file.
        Should be called after detect_ambiguous_terms().
        
        Args:
            filename: Optional custom filename. If not provided, uses a default name.
            
        Returns:
            Path to the saved file, or empty string if error.
        """
        if not filename:
            level_str = f"_level{self.level}" if self.level is not None else "_all_levels"
            filename = f"parent_context_details{level_str}.json"
            
        # Use the output_dir potentially set by CLI
        effective_output_dir = getattr(self, 'cli_output_dir', self.output_dir)
        filepath = os.path.join(effective_output_dir, filename)
        
        # Prepare the data structure
        output_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "detector": "ParentContextDetector",
            "level_filter": self.level,
            "parameters": {},
            "ambiguous_terms_count": len(self.detailed_results),
            "detailed_results": self.detailed_results
        }
        
        # Convert numpy types just in case (though unlikely needed here)
        output_data_serializable = convert_numpy_types(output_data)

        try:
            # Ensure directory exists one last time before writing
            os.makedirs(effective_output_dir, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(output_data_serializable, f, indent=2)
            logging.info(f"[ParentContextDetector] Saved detailed results to {filepath}")
            return filepath
        except Exception as e:
            logging.error(f"[ParentContextDetector] Error saving detailed results to {filepath}: {e}")
            return "" 