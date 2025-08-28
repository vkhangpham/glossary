#!/usr/bin/env python3
"""
Test the SenseSplitter against the ground truth test dataset.

This test ensures that:
1. Positive (ambiguous) terms get split into multiple senses
2. Negative (unambiguous) terms do not get split
3. The unified context format works correctly with the splitter
"""

import pytest
import json
import os
import sys
from pathlib import Path

# Add the project root to sys.path for imports
test_dir = Path(__file__).parent
project_root = test_dir.parent.parent
sys.path.insert(0, str(project_root))

from sense_disambiguation.splitter import SenseSplitter

# Test dataset paths
TEST_DATASET_ROOT = test_dir.parent / "data" / "test_dataset"
HIERARCHY_FILE = TEST_DATASET_ROOT / "hierarchy.json"
UNIFIED_CONTEXT_FILE = TEST_DATASET_ROOT / "unified_context_ground_truth.json"

# Expected terms
POSITIVE_TERMS = [
    "transformers", "interface", "modeling", "fragmentation", "clustering",
    "stress", "regression", "cell", "network", "bond"
]

NEGATIVE_TERMS = [
    "artificial intelligence", "mathematics", "engineering", "geology",
    "astrophysics", "botany", "microbiology", "cryptography"
]


class TestDatasetIntegration:
    """Test the complete dataset integration with the splitter."""

    @pytest.fixture(scope="class")
    def splitter(self):
        """Create a SenseSplitter instance using the test dataset."""
        return SenseSplitter(
            hierarchy_file_path=str(HIERARCHY_FILE),
            context_file=str(UNIFIED_CONTEXT_FILE),
            level=2,
            use_llm_for_tags=False,  # Use simulation to avoid API dependencies
            output_dir=str(TEST_DATASET_ROOT / "test_results")
        )

    def test_files_exist(self):
        """Test that all required dataset files exist."""
        assert HIERARCHY_FILE.exists(), f"Hierarchy file not found: {HIERARCHY_FILE}"
        assert UNIFIED_CONTEXT_FILE.exists(), f"Unified context file not found: {UNIFIED_CONTEXT_FILE}"

    def test_hierarchy_structure(self):
        """Test that the hierarchy file has the correct structure."""
        with open(HIERARCHY_FILE, 'r') as f:
            hierarchy = json.load(f)
        
        assert "terms" in hierarchy, "Hierarchy missing 'terms' key"
        terms = hierarchy["terms"]
        
        # Check all expected terms are present
        all_terms = POSITIVE_TERMS + NEGATIVE_TERMS
        for term in all_terms:
            assert term in terms, f"Term '{term}' missing from hierarchy"
            
            # Check required fields
            term_data = terms[term]
            assert "level" in term_data, f"Term '{term}' missing 'level'"
            assert "parents" in term_data, f"Term '{term}' missing 'parents'"
            assert "resources" in term_data, f"Term '{term}' missing 'resources'"
            assert term_data["level"] == 2, f"Term '{term}' has wrong level"
            assert len(term_data["resources"]) >= 7, f"Term '{term}' has insufficient resources"

    def test_unified_context_structure(self):
        """Test that the unified context file has the correct structure."""
        with open(UNIFIED_CONTEXT_FILE, 'r') as f:
            context_data = json.load(f)
        
        assert "contexts" in context_data, "Unified context missing 'contexts' key"
        contexts = context_data["contexts"]
        
        # Check all expected terms are present
        all_terms = POSITIVE_TERMS + NEGATIVE_TERMS
        for term in all_terms:
            assert term in contexts, f"Term '{term}' missing from unified context"
            
            # Check required fields
            term_context = contexts[term]
            assert "canonical_name" in term_context, f"Term '{term}' missing 'canonical_name'"
            assert "level" in term_context, f"Term '{term}' missing 'level'"
            assert "overall_confidence" in term_context, f"Term '{term}' missing 'overall_confidence'"
            assert "evidence" in term_context, f"Term '{term}' missing 'evidence'"
            assert term_context["level"] == 2, f"Term '{term}' has wrong level"

    def test_positive_terms_high_confidence(self):
        """Test that positive (ambiguous) terms have high confidence."""
        with open(UNIFIED_CONTEXT_FILE, 'r') as f:
            context_data = json.load(f)
        
        contexts = context_data["contexts"]
        
        for term in POSITIVE_TERMS:
            confidence = contexts[term]["overall_confidence"]
            assert confidence >= 0.7, f"Positive term '{term}' has low confidence: {confidence}"

    def test_negative_terms_low_confidence(self):
        """Test that negative (unambiguous) terms have low confidence."""
        with open(UNIFIED_CONTEXT_FILE, 'r') as f:
            context_data = json.load(f)
        
        contexts = context_data["contexts"]
        
        for term in NEGATIVE_TERMS:
            confidence = contexts[term]["overall_confidence"]
            assert confidence <= 0.3, f"Negative term '{term}' has high confidence: {confidence}"

    def test_splitter_initialization(self, splitter):
        """Test that the splitter initializes correctly with the test dataset."""
        assert splitter is not None
        assert splitter.hierarchy_file_path == str(HIERARCHY_FILE)
        assert splitter.context_file == str(UNIFIED_CONTEXT_FILE)
        assert splitter.level == 2

    def test_splitter_loads_context(self, splitter):
        """Test that the splitter correctly loads the unified context."""
        # This should trigger loading of the unified context
        assert hasattr(splitter, 'term_contexts')
        
        # Check that all terms are loaded
        all_terms = POSITIVE_TERMS + NEGATIVE_TERMS
        for term in all_terms:
            assert term in splitter.term_contexts, f"Term '{term}' not loaded in splitter context"

    def test_positive_terms_get_split(self, splitter):
        """Test that positive (ambiguous) terms get split into multiple senses."""
        accepted_proposals, rejected_proposals, _ = splitter.run(save_output=False)
        
        # Check that at least some positive terms are accepted for splitting
        accepted_terms = [proposal["original_term"] for proposal in accepted_proposals]
        positive_accepted = [term for term in accepted_terms if term in POSITIVE_TERMS]
        
        # We expect at least 50% of positive terms to be correctly identified as ambiguous
        expected_minimum = len(POSITIVE_TERMS) // 2
        assert len(positive_accepted) >= expected_minimum, \
            f"Expected at least {expected_minimum} positive terms to be split, got {len(positive_accepted)}: {positive_accepted}"
        
        # Check that accepted positive terms have multiple senses
        for proposal in accepted_proposals:
            if proposal["original_term"] in POSITIVE_TERMS:
                senses = proposal["proposed_senses"]
                assert len(senses) >= 2, \
                    f"Positive term '{proposal['original_term']}' should have at least 2 senses, got {len(senses)}"

    def test_negative_terms_not_split(self, splitter):
        """Test that negative (unambiguous) terms do not get split."""
        accepted_proposals, rejected_proposals, _ = splitter.run(save_output=False)
        
        # Check that negative terms are predominantly rejected
        accepted_terms = [proposal["original_term"] for proposal in accepted_proposals]
        negative_accepted = [term for term in accepted_terms if term in NEGATIVE_TERMS]
        
        # We expect at most 20% of negative terms to be incorrectly split
        max_allowed_splits = len(NEGATIVE_TERMS) // 5
        assert len(negative_accepted) <= max_allowed_splits, \
            f"Too many negative terms split: {negative_accepted} (max allowed: {max_allowed_splits})"

    def test_evidence_sources_work(self, splitter):
        """Test that different evidence sources are correctly utilized."""
        # Check that we can get evidence for different terms
        test_term = POSITIVE_TERMS[0]  # "transformers"
        
        # Test resource cluster evidence
        resource_evidence = splitter._get_resource_cluster_evidence(test_term)
        assert resource_evidence is not None, f"No resource cluster evidence for '{test_term}'"
        
        # Test parent context evidence
        parent_evidence = splitter._get_parent_context_evidence(test_term)
        assert parent_evidence is not None, f"No parent context evidence for '{test_term}'"

    @pytest.mark.parametrize("term", POSITIVE_TERMS)
    def test_individual_positive_terms(self, splitter, term):
        """Test each positive term individually to ensure it has proper evidence structure."""
        # Check that term exists in contexts
        assert term in splitter.term_contexts, f"Term '{term}' not in contexts"
        
        context = splitter.term_contexts[term]
        
        # Check confidence
        assert context["overall_confidence"] >= 0.7, \
            f"Term '{term}' has low confidence: {context['overall_confidence']}"
        
        # Check evidence structure
        assert "evidence" in context, f"Term '{term}' missing evidence"
        evidence_sources = [e["source"] for e in context["evidence"]]
        assert "resource_cluster" in evidence_sources, f"Term '{term}' missing resource_cluster evidence"

    @pytest.mark.parametrize("term", NEGATIVE_TERMS)
    def test_individual_negative_terms(self, splitter, term):
        """Test each negative term individually to ensure it has proper low-confidence structure."""
        # Check that term exists in contexts
        assert term in splitter.term_contexts, f"Term '{term}' not in contexts"
        
        context = splitter.term_contexts[term]
        
        # Check confidence
        assert context["overall_confidence"] <= 0.3, \
            f"Term '{term}' has high confidence: {context['overall_confidence']}"
        
        # Check evidence structure
        assert "evidence" in context, f"Term '{term}' missing evidence"
        evidence_sources = [e["source"] for e in context["evidence"]]
        assert "resource_cluster" in evidence_sources, f"Term '{term}' missing resource_cluster evidence"


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"]) 