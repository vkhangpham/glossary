"""
Implements the radial polysemy detection approach for ambiguity detection.

Based on the approach described in "Automatic Ambiguity Detection"
by Richard Sproat and Jan van Santen (arXiv:1905.12065).
"""

import json
import glob
import os
import logging
import re
import datetime
import numpy as np
from collections import defaultdict
from typing import Optional, Dict, List, Any, Tuple, Set
from sentence_transformers import SentenceTransformer
import warnings

# Import base detector classes
from .base import EvidenceBuilder, get_detector_version

# Add NLTK for improved tokenization
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
    # Download necessary NLTK data if not already available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK package not found. Using basic tokenization instead. "
                  "For improved results, install NLTK: pip install nltk")

from .utils import convert_numpy_types

class RadialPolysemyDetector:
    """
    Implements the polysemy detection approach described in "Automatic Ambiguity Detection"
    by Richard Sproat and Jan van Santen (arXiv:1905.12065).
    
    This detector identifies potentially ambiguous terms by:
    1. Collecting contextual terms within a window of the target term
    2. Computing inter-term distances and reducing to 2D using dimensionality reduction
    3. Converting to radial coordinates (distance from origin, angle)
    4. Measuring deviation from a single-peak distribution
    5. Calculating a polysemy index based on regression analysis
    """
    
    def __init__(self, hierarchy_file_path: str, 
                 final_term_files_pattern: str,
                 model_name: str = 'all-MiniLM-L6-v2',
                 context_window_size: int = 10,
                 min_contexts: int = 20,
                 level: Optional[int] = None,
                 output_dir: Optional[str] = None,
                 remove_stopwords: bool = True):
        """
        Initialize the RadialPolysemyDetector.
        
        Args:
            hierarchy_file_path: Path to the hierarchy.json file
            final_term_files_pattern: Glob pattern for canonical term files
            model_name: Embedding model to use for semantic analysis
            context_window_size: Size of window around target term to collect context
            min_contexts: Minimum number of contexts needed for analysis
            level: Hierarchy level to analyze (0-3, or None for all levels)
            output_dir: Directory to save results
            remove_stopwords: Whether to remove stopwords from context terms
        """
        self.hierarchy_file_path = hierarchy_file_path
        self.final_term_files_pattern = final_term_files_pattern
        self.model_name = model_name
        self.context_window_size = context_window_size
        self.min_contexts = min_contexts
        self.level = level
        self.remove_stopwords = remove_stopwords
        
        # Set output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            self.output_dir = os.path.join(repo_root, "sense_disambiguation/data", "ambiguity_detection_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Internal state
        self.hierarchy_data = None
        self.term_details = None
        self.canonical_terms = set()
        self._embedding_model = None
        self.polysemy_scores = {}
        self._loaded = False
        
        # Initialize stopwords if using NLTK
        if NLTK_AVAILABLE and self.remove_stopwords:
            self.stopwords = set(stopwords.words('english'))
        else:
            self.stopwords = set()
        
    @property
    def embedding_model(self):
        """Lazily load the embedding model when needed."""
        if self._embedding_model is None:
            try:
                logging.info(f"Loading sentence transformer model '{self.model_name}'")
                self._embedding_model = SentenceTransformer(self.model_name)
            except Exception as e:
                logging.error(f"Failed to load sentence transformer model: {e}")
                return None
        return self._embedding_model
    
    def _load_data(self):
        """Load hierarchy and canonical terms."""
        if self._loaded:
            return True
            
        # Load hierarchy
        try:
            with open(self.hierarchy_file_path, 'r') as f:
                self.hierarchy_data = json.load(f)
            self.term_details = self.hierarchy_data.get('terms', {})
            if not self.term_details:
                logging.warning("Hierarchy file loaded, but 'terms' dictionary is empty or missing.")
                return False
        except Exception as e:
            logging.error(f"Error loading hierarchy: {e}")
            return False
            
        # Load canonical terms
        final_term_files = glob.glob(self.final_term_files_pattern)
        if not final_term_files:
            logging.warning(f"No files found matching pattern: {self.final_term_files_pattern}")
            
        for file_path in final_term_files:
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        term = line.strip()
                        if term:
                            self.canonical_terms.add(term)
            except Exception as e:
                logging.error(f"Error reading file {file_path}: {e}")
                
        self._loaded = True
        return len(self.canonical_terms) > 0
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into a list of tokens using improved tokenization.
        
        Args:
            text: Text string to tokenize
            
        Returns:
            List of token strings
        """
        if not text:
            return []
            
        # Convert to lowercase for consistent processing
        text = text.lower()
        
        if NLTK_AVAILABLE:
            # Use NLTK's more advanced tokenization
            tokens = word_tokenize(text)
            
            # Remove stopwords if requested
            if self.remove_stopwords:
                tokens = [token for token in tokens if token not in self.stopwords]
                
            # Filter to only include alphanumeric tokens
            tokens = [token for token in tokens if token.isalnum()]
        else:
            # Fallback to basic regex tokenization
            tokens = re.findall(r'\b\w+\b', text)
            
        return tokens
        
    def _extract_context_terms(self, term: str) -> List[List[str]]:
        """
        Extract context terms that appear within the context window of the target term.
        
        Args:
            term: The target term to analyze
            
        Returns:
            List of context windows, each containing a list of terms
        """
        if term not in self.term_details:
            return []
            
        contexts = []
        resources = self.term_details[term].get('resources', [])
        term_parts = term.lower().split('_')
        term_length = len(term_parts)
        
        for resource in resources:
            content = resource.get('processed_content', '')
            if not content or not isinstance(content, str):
                continue
            
            # Tokenize content using improved tokenization
            tokens = self._tokenize_text(content)
            
            if not tokens:
                continue
                
            # Find occurrences of the target term parts (respecting compound terms with underscores)
            i = 0
            while i <= len(tokens) - term_length:
                # Check if this position matches the term parts
                if all(tokens[i+j] == term_parts[j] for j in range(term_length)):
                    # Found the term at position i
                    # Extract the surrounding context window
                    start = max(0, i - self.context_window_size)
                    end = min(len(tokens), i + term_length + self.context_window_size)
                    
                    # Create context window without the target term itself
                    context = tokens[start:i] + tokens[i+term_length:end]
                    if context:
                        contexts.append(context)
                    
                    # Skip past this occurrence
                    i += term_length
                else:
                    i += 1
        
        return contexts
    
    def _compute_radial_distribution(self, contexts: List[List[str]]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute the radial distribution of context terms as described in the paper.
        
        Steps:
        1. Embed all unique context terms
        2. Compute inter-term distances
        3. Reduce to 2D using t-SNE or similar
        4. Convert to radial coordinates (distance from origin, angle)
        5. Analyze the distribution using regression
        
        Args:
            contexts: List of context windows
            
        Returns:
            Tuple of (radial_coordinates, distribution_metrics)
        """
        if not self.embedding_model:
            return np.array([]), {}
            
        # Extract unique terms from all contexts
        all_terms = set()
        for context in contexts:
            all_terms.update(context)
        
        # Skip if too few unique terms
        if len(all_terms) < 5:
            return np.array([]), {}
            
        # Embed the unique terms
        unique_terms = list(all_terms)
        embeddings = self.embedding_model.encode(unique_terms, show_progress_bar=False)
        
        # Reduce dimensionality to 2D
        try:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(unique_terms)//5)), 
                        random_state=42, max_iter=1000)
            reduced_embeddings = tsne.fit_transform(embeddings)
        except Exception as e:
            logging.error(f"Error in dimensionality reduction: {e}")
            return np.array([]), {}
        
        # Convert to radial coordinates
        # First normalize embeddings around the origin
        center = np.mean(reduced_embeddings, axis=0)
        centered_embeddings = reduced_embeddings - center
        
        # Calculate radius (distance from origin) and angle (theta)
        radius = np.sqrt(np.sum(centered_embeddings**2, axis=1))
        theta = np.arctan2(centered_embeddings[:, 1], centered_embeddings[:, 0])
        
        # Create the radial coordinates array
        radial_coords = np.column_stack((radius, theta))
        
        # Sort by theta for regression analysis
        sorted_indices = np.argsort(theta)
        sorted_radius = radius[sorted_indices]
        
        # Analyze the radial distribution patterns
        metrics = self._analyze_radial_distribution(sorted_radius)
        
        return radial_coords, metrics
    
    def _analyze_radial_distribution(self, sorted_radius: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the radial distribution to determine if it exhibits multiple peaks.
        
        Uses isotonic/antitonic regression as mentioned in the paper to measure
        the deviation from a single-peak model.
        
        Args:
            sorted_radius: Array of radius values sorted by angle
            
        Returns:
            Dictionary of distribution metrics including the polysemy index
        """
        try:
            from sklearn.isotonic import IsotonicRegression
            
            # Normalize radius values
            max_radius = sorted_radius.max()
            if max_radius > 0:
                norm_radius = sorted_radius / max_radius
            else:
                norm_radius = sorted_radius
                
            # Create X coordinates (normalized)
            x = np.linspace(0, 1, len(norm_radius))
            
            # Fit isotonic regression (non-decreasing)
            ir_inc = IsotonicRegression(increasing=True)
            y_inc = ir_inc.fit_transform(x, norm_radius)
            
            # Fit antitonic regression (non-increasing)
            ir_dec = IsotonicRegression(increasing=False)
            y_dec = ir_dec.fit_transform(x, norm_radius)
            
            # Calculate mean squared errors for different models
            mse_inc = np.mean((norm_radius - y_inc)**2)
            mse_dec = np.mean((norm_radius - y_dec)**2)
            
            # Find the best single-peak model
            # (increasing then decreasing, with peak at different positions)
            min_mse = float('inf')
            best_peak_pos = 0.5  # Default middle
            for i in range(1, len(x)-1):
                # First part increasing
                ir_first = IsotonicRegression(increasing=True)
                y_first = ir_first.fit_transform(x[:i], norm_radius[:i])
                
                # Second part decreasing
                ir_second = IsotonicRegression(increasing=False)
                y_second = ir_second.fit_transform(x[i:], norm_radius[i:])
                
                # Combined model
                y_combined = np.concatenate([y_first, y_second])
                mse = np.mean((norm_radius - y_combined)**2)
                
                if mse < min_mse:
                    min_mse = mse
                    best_peak_pos = i / len(x)
            
            # Calculate deviation from single-peak model (the polysemy index)
            # Higher deviation indicates multiple peaks, suggesting polysemy
            polysemy_index = 1.0 - (min_mse / max(0.0001, max(mse_inc, mse_dec)))
            
            # Additional metrics
            variance = np.var(norm_radius)
            peak_count_estimate = self._estimate_peak_count(norm_radius)
            
            return {
                "polysemy_index": polysemy_index,
                "variance": variance,
                "peak_count_estimate": peak_count_estimate,
                "best_peak_position": best_peak_pos
            }
            
        except Exception as e:
            logging.error(f"Error in radial distribution analysis: {e}")
            return {"polysemy_index": 0.0, "error": str(e)}
    
    def _estimate_peak_count(self, values: np.ndarray, smoothing: int = 3) -> int:
        """
        Estimate the number of peaks in the radial distribution.
        
        Args:
            values: Array of values (radius sorted by angle)
            smoothing: Window size for smoothing
            
        Returns:
            Estimated number of peaks
        """
        if len(values) < 2 * smoothing + 1:
            return 1  # Too few points for reliable peak detection
            
        # Apply simple moving average smoothing
        smoothed = np.zeros_like(values)
        for i in range(len(values)):
            window_start = max(0, i - smoothing)
            window_end = min(len(values), i + smoothing + 1)
            smoothed[i] = np.mean(values[window_start:window_end])
        
        # Find peaks (local maxima)
        peaks = []
        for i in range(1, len(smoothed)-1):
            if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
                peaks.append(i)
        
        # Filter insignificant peaks (height difference should be significant)
        if len(peaks) <= 1:
            return len(peaks)
            
        significant_peaks = [peaks[0]]
        threshold = 0.1 * (np.max(smoothed) - np.min(smoothed))
        
        for i in range(1, len(peaks)):
            # Find minimum between consecutive peaks
            valley = np.min(smoothed[peaks[i-1]:peaks[i]+1])
            peak_height = smoothed[peaks[i]]
            prev_peak_height = smoothed[peaks[i-1]]
            
            # Check if the peaks are significantly higher than the valley
            if (peak_height - valley > threshold) and (prev_peak_height - valley > threshold):
                significant_peaks.append(peaks[i])
        
        return len(significant_peaks)
    
    def detect_ambiguous_terms(self) -> List[str]:
        """
        Detect potentially ambiguous terms using the radial polysemy approach.
        
        Returns:
            List of ambiguous terms sorted by polysemy index
        """
        if not self._load_data():
            logging.error("Failed to load data. Aborting detection.")
            return []
            
        if not self.embedding_model:
            logging.error("Embedding model not available. Aborting detection.")
            return []
        
        # Process all canonical terms or filter by level
        terms_to_process = []
        for term in self.canonical_terms:
            if term not in self.term_details:
                continue
                
            if self.level is not None:
                term_level = self.term_details[term].get('level')
                if term_level != self.level:
                    continue
                    
            terms_to_process.append(term)
        
        logging.info(f"[RadialPolysemyDetector] Analyzing {len(terms_to_process)} terms")
        
        # Process each term
        self.polysemy_scores = {}
        
        for idx, term in enumerate(terms_to_process):
            if idx % 100 == 0:
                logging.info(f"[RadialPolysemyDetector] Processing term {idx+1}/{len(terms_to_process)}: {term}")
                
            # Extract context terms
            contexts = self._extract_context_terms(term)
            if len(contexts) < self.min_contexts:
                continue
                
            # Compute radial distribution
            _, metrics = self._compute_radial_distribution(contexts)
            if not metrics or "polysemy_index" not in metrics:
                continue
                
            # Store the polysemy index and metrics
            polysemy_index = metrics["polysemy_index"]
            self.polysemy_scores[term] = {
                "polysemy_index": polysemy_index,
                "metrics": metrics,
                "context_count": len(contexts)
            }
            
            if polysemy_index >= 0.25:  # Match the reduced threshold
                logging.info(f"Term '{term}': polysemy_index={polysemy_index:.3f}, contexts={len(contexts)}")
        
        # Find terms with high polysemy index
        polysemy_threshold = 0.25  # Reduced from 0.35 to flag more potential ambiguous terms
        ambiguous_terms = [(term, data["polysemy_index"]) 
                           for term, data in self.polysemy_scores.items() 
                           if data["polysemy_index"] >= polysemy_threshold]
        
        # Sort by polysemy index (highest first)
        ambiguous_terms.sort(key=lambda x: x[1], reverse=True)
        
        logging.info(f"[RadialPolysemyDetector] Found {len(ambiguous_terms)} potentially ambiguous terms")
        
        return [term for term, _ in ambiguous_terms]
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save detection results to a JSON file.
        
        DEPRECATED: This method will be removed in a future version.
        The recommended approach is to get evidence blocks via detect() and store them in a unified context file.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to the saved file
        """
        import warnings
        warnings.warn(
            "RadialPolysemyDetector.save_results() is deprecated; direct file writes will be removed in future",
            DeprecationWarning, 
            stacklevel=2
        )
        
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            level_str = f"_level{self.level}" if self.level is not None else ""
            filename = f"radial_polysemy_results{level_str}_{timestamp}.json"
            
        filepath = os.path.join(self.output_dir, filename)
        
        # Prepare output data
        output_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "parameters": {
                "model_name": self.model_name,
                "context_window_size": self.context_window_size,
                "min_contexts": self.min_contexts,
                "level": self.level
            },
            "scores": self.polysemy_scores
        }
        
        # Convert NumPy types to Python native types for JSON serialization
        output_data_serializable = convert_numpy_types(output_data)
        
        try:
            # Ensure the directory exists before writing
            os.makedirs(self.output_dir, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(output_data_serializable, f, indent=2)
            logging.info(f"[RadialPolysemyDetector] Saved detailed scores to {filepath}")
            return filepath
        except Exception as e:
            logging.error(f"[RadialPolysemyDetector] Error saving detailed scores: {e}")
            return ""

    def save_summary_results(self, ambiguous_terms: List[str], filename: Optional[str] = None) -> str:
        """
        Saves the final list of ambiguous terms to a simple text file.
        
        DEPRECATED: This method will be removed in a future version.
        The recommended approach is to get evidence blocks via detect() and store them in a unified context file.
        
        Args:
            ambiguous_terms: List of terms identified as ambiguous.
            filename: Optional custom filename. If not provided, uses a default name.
            
        Returns:
            Path to the saved file, or empty string if error.
        """
        import warnings
        warnings.warn(
            "RadialPolysemyDetector.save_summary_results() is deprecated; direct file writes will be removed in future",
            DeprecationWarning, 
            stacklevel=2
        )
        
        if not filename:
            level_str = f"_level{self.level}" if self.level is not None else "_all_levels"
            filename = f"radial_polysemy_terms{level_str}.txt"
            
        # Use the output_dir potentially set by CLI
        effective_output_dir = getattr(self, 'cli_output_dir', self.output_dir)
        filepath = os.path.join(effective_output_dir, filename)

        try:
            # Ensure directory exists one last time before writing
            os.makedirs(effective_output_dir, exist_ok=True)
            with open(filepath, 'w') as f:
                for term in sorted(ambiguous_terms):
                    f.write(f"{term}\n")
            logging.info(f"[RadialPolysemyDetector] Saved summary results (term list) to {filepath}")
            return filepath
        except Exception as e:
            logging.error(f"[RadialPolysemyDetector] Error saving summary results to {filepath}: {e}")
            return "" 

    def detect(self) -> List[EvidenceBuilder]:
        """
        Detects ambiguous terms based on radial polysemy and returns evidence builders.
        
        This is the new preferred API that returns structured evidence blocks for the splitter.
        
        Returns:
            List of EvidenceBuilder objects for detected ambiguous terms.
        """
        # Run detection to populate polysemy_scores
        # This approach maintains backward compatibility while exposing the new API
        detected_terms = self.detect_ambiguous_terms()
        
        # Get the detector version
        version = get_detector_version()
        
        # Number of sample contexts to include (capped)
        max_sample_contexts = 5
        max_context_length = 300  # chars per sample context
        
        # Convert to evidence builders
        evidence_builders = []
        
        # Process terms with polysemy scores
        for term, score_data in self.polysemy_scores.items():
            # Get polysemy metrics
            polysemy_index = score_data.get("polysemy_index", 0.0)
            context_count = score_data.get("context_count", 0)
            metrics = score_data.get("metrics", {})
            
            # Skip terms with very low polysemy index
            if polysemy_index < 0.05:  # Ultra-low confidence threshold for detection
                continue
                
            # Calculate confidence using logistic function on polysemy_index
            # Uses scaled logistic: 1/(1+e^(-10*(x-0.25))) 
            # This gives ~0.5 confidence at the threshold of 0.25
            # and approaches 1.0 as polysemy_index increases
            confidence = 1.0 / (1.0 + np.exp(-10 * (polysemy_index - 0.25)))
            
            # Ensure confidence is in valid range
            confidence = max(0.0, min(1.0, confidence))
            
            # Get the term level
            level = None
            if term in self.term_details:
                level = self.term_details[term].get("level")
            
            # Prepare metrics dictionary
            api_metrics = {
                "polysemy_index": polysemy_index,
                "context_count": context_count,
                "peak_count_estimate": metrics.get("peak_count_estimate", 1),
                "variance": metrics.get("variance", 0.0)
            }
            
            # Extract contexts for the term to include in payload
            sample_contexts = []
            
            # Re-extract contexts to include in payload
            # We can't reuse easily because the contexts are processed during detection
            contexts = self._extract_context_terms(term)
            
            # Select a representative subset of contexts
            # Since we're doing semantic modeling, it's better to take evenly spaced samples
            # rather than the first N contexts, which might all be from the same resource
            if contexts and len(contexts) > 0:
                # Take evenly spaced samples
                step = max(1, len(contexts) // max_sample_contexts)
                for i in range(0, len(contexts), step):
                    if len(sample_contexts) >= max_sample_contexts:
                        break
                        
                    # Convert token list to readable string
                    context_str = " ".join(contexts[i])
                    
                    # Truncate if too long
                    if len(context_str) > max_context_length:
                        context_str = context_str[:max_context_length] + "..."
                        
                    sample_contexts.append(context_str)
            
            # Prepare payload dictionary
            payload = {
                "polysemy_index": polysemy_index,
                "context_count": context_count,
                "sample_contexts": sample_contexts
            }
            
            # Create the evidence builder
            builder = EvidenceBuilder.create(
                term=term,
                level=level,
                source="radial_polysemy",
                detector_version=version,
                confidence=confidence,
                metrics=api_metrics,
                payload=payload
            )
            
            evidence_builders.append(builder)
        
        # Mark legacy file writing methods as deprecated
        warnings.warn(
            "RadialPolysemyDetector.save_results() and save_summary_results() are deprecated; "
            "use detect() and unified context files instead",
            DeprecationWarning, 
            stacklevel=2
        )
        
        return evidence_builders 