"""
Hybrid detector that combines multiple detection approaches for ambiguity detection.

This module implements the HybridAmbiguityDetector which combines results from
multiple detection approaches to provide higher quality and confidence results.
"""

import os
import logging
import datetime
import json
from typing import Optional, Dict, List, Any, Tuple, Set
import warnings

# Import base classes for the unified API
from .base import EvidenceBuilder, merge_term_contexts, get_detector_version, TermContext

# Import all necessary detectors
from .parent_context import ParentContextDetector
from .resource_cluster import ResourceClusterDetector
from .radial_polysemy import RadialPolysemyDetector
from .utils import convert_numpy_types

# Setup logger if not already configured by parent module
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class HybridAmbiguityDetector:
    """
    Combines multiple ambiguity detection approaches for higher confidence scoring.
    
    Integrates signals from Parent Context, Resource Clustering (DBSCAN),
    and optionally Radial Polysemy.
    """
    
    def __init__(self, hierarchy_file_path: str, final_term_files_pattern: str,
                 model_name: str = 'all-MiniLM-L6-v2',
                 output_dir: Optional[str] = None,
                 min_resources: int = 5,
                 level: Optional[int] = None,
                 use_radial_detector: bool = True):
        """
        Initialize the hybrid detector with configurations for all sub-detectors.
        
        Args:
            hierarchy_file_path: Path to the hierarchy.json file
            final_term_files_pattern: Glob pattern for canonical term files
            model_name: Name of the embedding model to use
            output_dir: Directory to save results
            min_resources: Minimum resources needed for ResourceClusterDetector
            level: Optional hierarchy level (0-3) for targeted analysis
            use_radial_detector: Whether to include the RadialPolysemyDetector in the analysis
        """
        self.hierarchy_file_path = hierarchy_file_path
        self.final_term_files_pattern = final_term_files_pattern
        self.model_name = model_name
        self.min_resources = min_resources
        self.level = level
        self.use_radial_detector = use_radial_detector
        
        # Set output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            self.output_dir = os.path.join(repo_root, "sense_disambiguation/data", "ambiguity_detection_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize the sub-detectors (but don't run them yet)
        # 1. Parent Context Detector
        self.parent_context_detector = ParentContextDetector(
            hierarchy_file_path=hierarchy_file_path,
            final_term_files_pattern=final_term_files_pattern,
            level=level
        )
        # Set output dir for parent context detector to match hybrid's
        self.parent_context_detector.output_dir = self.output_dir
        
        # 2. Resource Cluster Detector (DBSCAN)
        self.dbscan_detector = ResourceClusterDetector(
            hierarchy_file_path=hierarchy_file_path,
            final_term_files_pattern=final_term_files_pattern,
            model_name=model_name,
            min_resources=min_resources,
            clustering_algorithm='dbscan', # Explicitly DBSCAN for this instance
            level=level,
            output_dir=output_dir
        )
        
        # 3. Radial Polysemy Detector (Optional)
        if self.use_radial_detector:
            self.radial_detector = RadialPolysemyDetector(
                hierarchy_file_path=hierarchy_file_path,
                final_term_files_pattern=final_term_files_pattern,
                model_name=model_name,
                context_window_size=10,
                min_contexts=10, # Keep the reduced min_contexts
                level=level,
                output_dir=output_dir
            )
        else:
            self.radial_detector = None
        
        # Results storage
        self.results = {}
        self.confidence_scores = {}
        
        # Cache for full analysis results (run once without level filtering)
        self._all_level_performed = False
        self._cached_parent_context_terms = None
        self._cached_parent_context_details = None
        self._cached_dbscan_terms = None
        self._cached_dbscan_metrics = None
        self._cached_radial_terms = None
        self._cached_radial_scores = None
    
    def detect_ambiguous_terms(self) -> Dict[str, Dict[str, Any]]:
        """
        Run all available detectors and combine their results with refined confidence scoring.
        
        DEPRECATED: This method will be removed in a future version.
        The recommended approach is to use detect() which returns term contexts conforming
        to the unified schema.
        
        Returns:
            Dictionary mapping from term strings to metadata dictionaries with 
            detection info and confidence scores.
        """
        warnings.warn(
            "HybridAmbiguityDetector.detect_ambiguous_terms() is deprecated; use detect() instead",
            DeprecationWarning, 
            stacklevel=2
        )
        # Check if we have already run all detectors (without level constraint)
        run_detectors = True
        if self.level is not None and self._all_level_performed:
            logger.info("[HybridAmbiguityDetector] Using cached results and filtering by level")
            run_detectors = False
            
            # Get cached results and filter them by the current level
            # Note: Filtering logic is handled within each sub-detector when level is set
            self.parent_context_detector.level = self.level
            parent_context_terms = set(self.parent_context_detector.detect_ambiguous_terms())
            parent_context_details = self.parent_context_detector.detailed_results
            
            self.dbscan_detector.level = self.level
            dbscan_terms = set(self.dbscan_detector.detect_ambiguous_terms())
            dbscan_metrics = self.dbscan_detector.get_cluster_metrics()
            
            radial_terms = set()
            radial_scores = {}
            if self.radial_detector and self._cached_radial_terms is not None:
                self.radial_detector.level = self.level
                radial_terms = set(self.radial_detector.detect_ambiguous_terms())
                radial_scores = self.radial_detector.polysemy_scores
        
        if run_detectors:
            # --- Run Detectors (without level constraint first) ---
            original_level = self.level
            
            # 1. Parent Context Detector
            logger.info("[HybridAmbiguityDetector] Running ParentContextDetector...")
            self.parent_context_detector.level = None # Run on all levels for caching
            parent_context_terms = set(self.parent_context_detector.detect_ambiguous_terms())
            parent_context_details = self.parent_context_detector.detailed_results # Get details
            self._cached_parent_context_terms = parent_context_terms.copy()
            self._cached_parent_context_details = parent_context_details.copy()
            logger.info(f"[HybridAmbiguityDetector] ParentContextDetector found {len(parent_context_terms)} ambiguous terms")
            
            # 2. Resource Cluster Detector (DBSCAN)
            logger.info("[HybridAmbiguityDetector] Running ResourceClusterDetector with DBSCAN...")
            self.dbscan_detector.level = None # Run on all levels for caching
            dbscan_terms = set(self.dbscan_detector.detect_ambiguous_terms())
            dbscan_metrics = self.dbscan_detector.get_cluster_metrics()
            self._cached_dbscan_terms = dbscan_terms.copy()
            self._cached_dbscan_metrics = dbscan_metrics.copy()
            logger.info(f"[HybridAmbiguityDetector] DBSCAN detector found {len(dbscan_terms)} ambiguous terms")
            
            # 3. Radial Polysemy Detector (Optional)
            radial_terms = set()
            radial_scores = {}
            if self.radial_detector:
                logger.info("[HybridAmbiguityDetector] Running RadialPolysemyDetector...")
                self.radial_detector.level = None # Run on all levels for caching
                radial_terms = set(self.radial_detector.detect_ambiguous_terms())
                radial_scores = self.radial_detector.polysemy_scores
                self._cached_radial_terms = radial_terms.copy()
                self._cached_radial_scores = radial_scores.copy()
                logger.info(f"[HybridAmbiguityDetector] RadialPolysemyDetector found {len(radial_terms)} ambiguous terms")
                
            # Restore original level for subsequent filtering if needed
            self.level = original_level
            self.parent_context_detector.level = original_level
            self.dbscan_detector.level = original_level
            if self.radial_detector: self.radial_detector.level = original_level
                
            # Mark that we have run all detectors
            self._all_level_performed = True

            # --- Apply level filtering if a specific level was requested ---
            if self.level is not None:
                 logger.info(f"[HybridAmbiguityDetector] Filtering cached results for level {self.level}")
                 # Re-run detectors with the specific level set to get filtered results
                 parent_context_terms = set(self.parent_context_detector.detect_ambiguous_terms())
                 parent_context_details = self.parent_context_detector.detailed_results
                 
                 dbscan_terms = set(self.dbscan_detector.detect_ambiguous_terms())
                 dbscan_metrics = self.dbscan_detector.get_cluster_metrics()
                 
                 radial_terms = set()
                 radial_scores = {}
                 if self.radial_detector:
                     radial_terms = set(self.radial_detector.detect_ambiguous_terms())
                     radial_scores = self.radial_detector.polysemy_scores

        # --- Combine and Score Results ---
        # Combine terms found by EITHER Parent Context OR DBSCAN Clustering
        all_candidate_terms = parent_context_terms.union(dbscan_terms)
        logger.info(f"[HybridAmbiguityDetector] Total candidate terms (ParentContext or DBSCAN): {len(all_candidate_terms)}")
        
        results = {}
        for term in all_candidate_terms:
            # Initialize result structure
            result = {
                "term": term,
                "level": self.dbscan_detector.term_details.get(term, {}).get("level"), # Get level from details
                "detected_by": {
                    "parent_context": term in parent_context_terms,
                    "dbscan": term in dbscan_terms,
                    "radial_polysemy": term in radial_terms # Still track if radial found it
                },
                "metrics": {},
                "confidence": 0.0 # Calculated below
            }
            
            # Extract relevant metrics/details for scoring and output
            term_dbscan_metrics = dbscan_metrics.get(term, {})
            term_parent_details = parent_context_details.get(term, {})
            term_radial_scores = radial_scores.get(term, {})
            
            # Populate metrics section for diagnostics
            if term_parent_details:
                 result["metrics"]["parent_context"] = {
                     "divergent": True, # Implicitly true if in parent_context_terms
                     "ancestor_contexts": term_parent_details.get("ancestor_contexts", []), # Already included
                     # Add any other relevant parent context details if needed
                     "distinct_ancestor_pairs_count": term_parent_details.get("distinct_ancestor_pairs_count", 0)
                 }
            if term_dbscan_metrics:
                result["metrics"]["dbscan"] = term_dbscan_metrics # Include all dbscan metrics
            if term_radial_scores:
                 result["metrics"]["radial_polysemy"] = {
                     "polysemy_index": term_radial_scores.get("polysemy_index", 0.0),
                     "context_count": term_radial_scores.get("context_count", 0),
                     # Add sample contexts if available and desired in the future
                 }

            # Calculate confidence score using the new logic
            result["confidence"] = self._calculate_confidence(
                term=term,
                detected_by=result["detected_by"],
                dbscan_metrics=term_dbscan_metrics,
                parent_details=term_parent_details,
                radial_scores=term_radial_scores
            )

            # Assign confidence level string for convenience
            conf_score = result["confidence"]
            if conf_score >= 0.8: result["confidence_level"] = "high"
            elif conf_score >= 0.5: result["confidence_level"] = "medium"
            else: result["confidence_level"] = "low"

            results[term] = result
            
        self.results = results # Store final results
        return self.results

    def detect(self) -> Dict[str, TermContext]:
        """
        Run all available detectors using the new unified API and merge their evidence.
        
        This is the recommended API that returns structured term contexts conforming
        to the unified schema for the splitter.
        
        Returns:
            Dictionary mapping term strings to TermContext objects containing all evidence.
        """
        logger.info("[HybridAmbiguityDetector] Running detectors with unified API...")
        
        # List to collect all evidence from all detectors
        all_evidence: List[EvidenceBuilder] = []
        
        # 1. Run ParentContextDetector
        logger.info("[HybridAmbiguityDetector] Running ParentContextDetector...")
        parent_context_evidence = self.parent_context_detector.detect()
        all_evidence.extend(parent_context_evidence)
        logger.info(f"[HybridAmbiguityDetector] ParentContextDetector found {len(parent_context_evidence)} potential terms")
        
        # 2. Run ResourceClusterDetector
        logger.info("[HybridAmbiguityDetector] Running ResourceClusterDetector...")
        dbscan_evidence = self.dbscan_detector.detect()
        all_evidence.extend(dbscan_evidence)
        logger.info(f"[HybridAmbiguityDetector] ResourceClusterDetector found {len(dbscan_evidence)} potential terms")
        
        # 3. Run RadialPolysemyDetector (if enabled)
        if self.radial_detector:
            logger.info("[HybridAmbiguityDetector] Running RadialPolysemyDetector...")
            radial_evidence = self.radial_detector.detect()
            all_evidence.extend(radial_evidence)
            logger.info(f"[HybridAmbiguityDetector] RadialPolysemyDetector found {len(radial_evidence)} potential terms")
        
        # Merge all evidence to create term contexts
        logger.info(f"[HybridAmbiguityDetector] Merging {len(all_evidence)} evidence blocks...")
        term_contexts = merge_term_contexts(*all_evidence)
        logger.info(f"[HybridAmbiguityDetector] Created {len(term_contexts)} unified term contexts")
        
        # Apply level filtering if needed
        if self.level is not None:
            filtered_contexts = {
                term: context for term, context in term_contexts.items()
                if context.level == self.level
            }
            logger.info(f"[HybridAmbiguityDetector] Filtered to {len(filtered_contexts)} terms at level {self.level}")
            return filtered_contexts
        
        return term_contexts
        
    def save_unified_context(self, term_contexts: Dict[str, TermContext], filename: Optional[str] = None) -> str:
        """
        Save the unified context to a JSON file.
        
        Args:
            term_contexts: Dictionary mapping terms to their TermContext objects
            filename: Optional custom filename
            
        Returns:
            Path to the saved file, or empty string if error
        """
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            level_str = f"_level{self.level}" if self.level is not None else ""
            filename = f"unified_context{level_str}_{timestamp}.json"
            
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            # Ensure the directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Convert term_contexts to dict for JSON serialization
            context_dict = {term: context.model_dump() for term, context in term_contexts.items()}
            
            # Add metadata
            output_data = {
                "schema_version": "v1",
                "timestamp": datetime.datetime.now().isoformat(),
                "detector_version": get_detector_version(),
                "parameters": {
                    "hierarchy_file": os.path.basename(self.hierarchy_file_path),
                    "model_name": self.model_name,
                    "level": self.level,
                    "use_radial_detector": self.use_radial_detector
                },
                "contexts": context_dict
            }
            
            # Convert any remaining numpy types for JSON serialization
            output_data_serializable = convert_numpy_types(output_data)
            
            with open(filepath, 'w') as f:
                json.dump(output_data_serializable, f, indent=2)
                
            logger.info(f"[HybridAmbiguityDetector] Saved unified context to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"[HybridAmbiguityDetector] Error saving unified context to {filepath}: {e}")
            return ""

    def detect_and_save(self, min_confidence: float = 0.3) -> Tuple[Dict[str, TermContext], str]:
        """
        Run detection and save the unified context to a file.
        
        Args:
            min_confidence: Minimum confidence threshold (0.0-1.0) for including terms
                           in the unified context file. Default is 0.3.
        
        Returns:
            Tuple of (term_contexts, output_filepath)
        """
        # Run detection to get all term contexts
        term_contexts = self.detect()
        
        # Filter out terms with confidence below the threshold
        filtered_contexts = {
            term: context for term, context in term_contexts.items()
            if context.overall_confidence >= min_confidence
        }
        
        # Log filtering results
        logger.info(f"[HybridAmbiguityDetector] Filtered {len(term_contexts)} terms to {len(filtered_contexts)} "
                   f"terms with confidence >= {min_confidence}")
        
        # Save the filtered contexts
        output_filepath = self.save_unified_context(filtered_contexts)
        return filtered_contexts, output_filepath

    def _calculate_confidence(self, term: str, detected_by: Dict[str, bool],
                              dbscan_metrics: Dict[str, Any],
                              parent_details: Dict[str, Any],
                              radial_scores: Dict[str, Any]) -> float:
        """
        Calculate a unified confidence score based on signals from multiple detectors.
        Iteration 1: Slightly relaxed clustering criteria.
        
        Args:
            term: The term being scored.
            detected_by: Dictionary indicating which detectors flagged the term.
            dbscan_metrics: Metrics from ResourceClusterDetector.
            parent_details: Details from ParentContextDetector.
            radial_scores: Scores from RadialPolysemyDetector.
            
        Returns:
            Confidence score between 0.0 and 1.0.
        """
        confidence = 0.0
        
        # --- Score from Resource Clustering (DBSCAN) ---
        dbscan_base = 0.0
        if detected_by["dbscan"]:
            cluster_count = 0
            if "cluster_labels" in dbscan_metrics:
                labels = dbscan_metrics["cluster_labels"]
                if labels:
                    non_noise_clusters = set(l for l in labels if l >= 0)
                    cluster_count = len(non_noise_clusters)
            
            separation = dbscan_metrics.get("separation_score", 0.0) 
            silhouette = dbscan_metrics.get("silhouette_score", -1.0)
            
            # Require multiple clusters AND minimum separation for base score
            if cluster_count > 1:
                dbscan_base = 0.2 # Small base score for >1 cluster
                
                # Boost only for *good* separation/silhouette metrics
                if separation > 0.45: # Higher threshold for separation boost
                    dbscan_base += min(0.2, (separation - 0.45) * 0.5) # Max boost 0.2
                
                if silhouette > 0.25: # Higher threshold for silhouette boost
                     dbscan_base += min(0.1, (silhouette - 0.25) * 0.3) # Max boost 0.1

        confidence += dbscan_base
        
        # --- Boost from Parent Context Divergence ---
        parent_boost = 0.0
        if detected_by["parent_context"]:
            # Check details if available (redundant check, but safe)
            if parent_details and parent_details.get("ancestor_contexts"):
                parent_boost = 0.5 # Increased boost for structural ambiguity
        
        confidence += parent_boost
        
        # --- Adjustments based on Resource Count (Kept) ---
        num_resources = dbscan_metrics.get("num_resources", 0)
        if num_resources > 0 and num_resources < 10:
            # Apply penalty if confidence is mainly from clustering with few resources
            # Adjusted threshold check for dbscan_base
            if dbscan_base > 0.2 and parent_boost < 0.1: 
                 penalty_factor = 0.7 + 0.3 * (num_resources / 10)
                 confidence *= penalty_factor
                 logger.debug(f"[HybridAmbiguityDetector] Applied resource penalty factor {penalty_factor:.2f} to '{term}'")

        # Cap confidence at 1.0
        final_confidence = min(1.0, confidence)
        
        # Log the components of the stricter score calculation
        logger.debug(f"[HybridAmbiguityDetector] Stricter Confidence for '{term}': Base(DBScan)={dbscan_base:.2f}, Boost(Parent)={parent_boost:.2f} -> Final={final_confidence:.2f}")
        
        return final_confidence

    def get_results_by_confidence(self, min_confidence: float = 0.0) -> Dict[str, list]:
        """
        Filter results by confidence level and categorize into high, medium, low.
        
        Args:
            min_confidence: Minimum confidence threshold to include results.
            
        Returns:
            Dictionary with lists of terms for 'high', 'medium', 'low' confidence.
        """
        categorized_results = {"high": [], "medium": [], "low": []}
        
        for term, data in self.results.items():
            confidence = data.get('confidence', 0.0)
            if confidence >= min_confidence:
                if confidence >= 0.8:
                    categorized_results["high"].append(term)
                elif confidence >= 0.5:
                    categorized_results["medium"].append(term)
                else:
                    categorized_results["low"].append(term)
                    
        # Sort lists alphabetically
        for key in categorized_results:
            categorized_results[key].sort()
            
        return categorized_results

    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save the combined detection results and confidence scores to a JSON file.
        
        DEPRECATED: This method will be removed in a future version.
        The recommended approach is to use detect() and save_unified_context() instead.
        
        Args:
            filename: Optional custom filename.
            
        Returns:
            Path to the saved file, or empty string if error.
        """
        warnings.warn(
            "HybridAmbiguityDetector.save_results() is deprecated; use detect() and save_unified_context() instead",
            DeprecationWarning, 
            stacklevel=2
        )
        
        if not filename:
            level_str = f"_level{self.level}" if self.level is not None else "_all_levels"
            filename = f"hybrid_detection_results{level_str}.json" # Consistent naming
            
        filepath = os.path.join(self.output_dir, filename)
        
        # Prepare the output data structure
        # Include parameters from sub-detectors for clarity
        output_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "detector": "HybridAmbiguityDetector (Improved)",
            "level_filter": self.level,
            "parameters": {
                "model_name": self.model_name,
                "min_resources": self.min_resources,
                "use_radial_detector": self.use_radial_detector,
                "dbscan_params": { # Include DBSCAN params used
                     "eps": getattr(self.dbscan_detector, 'dbscan_eps', 'N/A'),
                     "min_samples": getattr(self.dbscan_detector, 'dbscan_min_samples', 'N/A')
                 },
                "radial_params": { # Include Radial params if used
                    "context_window_size": getattr(self.radial_detector, 'context_window_size', 'N/A') if self.radial_detector else 'N/A',
                    "min_contexts": getattr(self.radial_detector, 'min_contexts', 'N/A') if self.radial_detector else 'N/A'
                 } if self.use_radial_detector else {}
            },
            "total_ambiguous_terms": len(self.results),
            "detailed_results": self.results # Already contains confidence and metrics
        }
        
        # Convert NumPy types to Python native types for JSON serialization
        output_data_serializable = convert_numpy_types(output_data)

        try:
            # Ensure the directory exists one last time before writing
            os.makedirs(self.output_dir, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(output_data_serializable, f, indent=2)
            logger.info(f"[HybridAmbiguityDetector] Saved detailed results to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"[HybridAmbiguityDetector] Error saving detailed results to {filepath}: {e}")
            return "" 