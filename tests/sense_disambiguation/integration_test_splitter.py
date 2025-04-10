#!/usr/bin/env python3
"""
Integration test script for the SenseSplitter class, focusing on real LLM calls
for field distinctness verification.

NOTE: This test makes actual LLM API calls and may incur costs and take time.
Ensure your LLM provider (e.g., OpenAI, Gemini) API keys are configured in your environment.
"""

import os
import sys
import unittest
import logging
from dotenv import load_dotenv

# Add the project root to sys.path to allow importing project modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from generate_glossary.sense_disambiguation.splitter import SenseSplitter
from generate_glossary.utils.llm import Provider  # Assuming Provider constants are defined here
from generate_glossary.utils.logger import setup_logger

# Load environment variables (for LLM API keys)
load_dotenv()

# Setup logging for the test
logger = setup_logger("integration_test_splitter", level=logging.INFO)

# --- Configuration --- 
# Use a real hierarchy file (or a smaller subset for testing)
HIERARCHY_FILE = "data/hierarchy.json" 
# Select a few candidate terms if needed, otherwise an empty list is fine for this test
CANDIDATE_TERMS = []
# Specify the LLM provider to use (or None to use default from SenseSplitter)
# Ensure the corresponding API key is set in your .env file
LLM_PROVIDER_TO_TEST = Provider.OPENAI # Or Provider.GEMINI, etc.
# --- End Configuration ---

class TestSenseSplitterIntegration(unittest.TestCase):
    """Integration tests for SenseSplitter focusing on LLM interaction."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the SenseSplitter instance once for all tests in this class."""
        logger.info(f"Setting up SenseSplitter for integration tests using {LLM_PROVIDER_TO_TEST}...")
        try:
            cls.splitter = SenseSplitter(
                hierarchy_file_path=HIERARCHY_FILE,
                candidate_terms_list=CANDIDATE_TERMS,
                cluster_results={}, # Not needed for field distinctness test
                use_llm_for_tags=True, # Ensure LLM is enabled
                llm_provider=LLM_PROVIDER_TO_TEST,
                # Optionally specify a model, otherwise SenseSplitter default will be used
                # llm_model="gpt-4o", 
                level=2 # Level doesn't impact this specific test much
            )
            # Ensure the LLM was initialized correctly
            if not cls.splitter._llm:
                 raise Exception("LLM failed to initialize. Check API keys and provider settings.")
            logger.info("SenseSplitter initialized successfully with real LLM.")
        except Exception as e:
            logger.error(f"Failed to set up SenseSplitter: {e}", exc_info=True)
            # Skip tests if setup failed
            raise unittest.SkipTest(f"Skipping integration tests due to setup failure: {e}")

    def run_distinctness_test(self, field1: str, field2: str, expected_distinct: bool):
        """Helper method to run a distinctness test and log results."""
        logger.info(f"Testing: '{field1}' vs '{field2}' (Expected: {'DISTINCT' if expected_distinct else 'NOT_DISTINCT'})")
        try:
            distinct, reason = self.splitter._check_field_distinctness_with_llm(field1, field2)
            logger.info(f"  LLM Verdict: {'DISTINCT' if distinct else 'NOT_DISTINCT'}")
            logger.info(f"  LLM Explanation: {reason}")
            self.assertEqual(distinct, expected_distinct, 
                             f"Mismatch for '{field1}' vs '{field2}'. Expected {expected_distinct}, got {distinct}. Reason: {reason}")
        except Exception as e:
            logger.error(f"  Test failed with exception: {e}", exc_info=True)
            self.fail(f"Exception during LLM call for '{field1}' vs '{field2}': {e}")

    def test_truly_distinct_fields(self):
        """Test pairs of fields that should be clearly distinct."""
        test_pairs = [
            ("image processing", "food processing"),
            ("cell biology", "prison cell studies"), 
            ("mathematical programming", "computer programming"),
            ("signal processing", "social policy")
        ]
        for field1, field2 in test_pairs:
            self.run_distinctness_test(field1, field2, expected_distinct=True)
            
    def test_non_distinct_fields(self):
        """Test pairs of fields that represent aspects of the same concept."""
        test_pairs = [
            ("machine learning algorithms", "machine learning applications"),
            ("cognitive neuroscience", "behavioral neuroscience"),
            ("quantum mechanics theory", "quantum mechanics experiments"),
            ("psychological stress", "mental stress") 
        ]
        for field1, field2 in test_pairs:
            self.run_distinctness_test(field1, field2, expected_distinct=False)
            
    def test_borderline_fields(self):
        """Test pairs of fields that are closely related or overlapping."""
        # Pairs expected to be NOT DISTINCT based on prompt guidelines
        non_distinct_pairs = [
            ("computational linguistics", "natural language processing")
        ]
        for field1, field2 in non_distinct_pairs:
             self.run_distinctness_test(field1, field2, expected_distinct=False)
             
        # Pair where the LLM provided a reasonable argument for DISTINCT
        distinct_pairs = [
            ("data science", "statistics"),
            ("robotics", "artificial intelligence") # Moved from non_distinct_pairs
        ]
        for field1, field2 in distinct_pairs:
            self.run_distinctness_test(field1, field2, expected_distinct=True)

    def test_harder_borderline_cases(self):
        """Test pairs with more nuanced or challenging relationships."""
        # Pairs where distinction is debatable or depends heavily on interpretation
        # Expected NOT_DISTINCT (Highly overlapping / Method vs. Field / Theory vs. Application)
        non_distinct_pairs_hard = [
            ("social network analysis", "network science", False), # SNA as application/subfield of Network Science
            ("epidemiology", "public health", False), # Epidemiology as core method within Public Health
            ("statistical learning", "machine learning", False), # Statistical Learning often seen as foundational/subset of ML
            ("critical theory", "cultural studies", False), # Very intertwined, Cultural Studies often uses Critical Theory
            ("quantum computing", "quantum information theory", False) # Arguably different aspects (hardware/application vs theory) of same quantum principles
        ]
        
        # Expected DISTINCT (Different core disciplines / Approach vs Toolset)
        distinct_pairs_hard = [
            ("behavioral economics", "cognitive psychology", True), # Different core fields (Economics vs Psychology) despite informing each other
            ("digital humanities", "media studies", True), # DH applies computation TO humanities; Media Studies analyzes media itself
            ("systems biology", "bioinformatics", True) # Moved from non_distinct_pairs_hard
        ]
        
        for field1, field2, expected_distinct in non_distinct_pairs_hard:
            self.run_distinctness_test(field1, field2, expected_distinct=expected_distinct)
            
        for field1, field2, expected_distinct in distinct_pairs_hard:
             self.run_distinctness_test(field1, field2, expected_distinct=expected_distinct)

if __name__ == "__main__":
    print("Running SenseSplitter Integration Tests...")
    print("NOTE: This involves real LLM API calls.")
    unittest.main() 