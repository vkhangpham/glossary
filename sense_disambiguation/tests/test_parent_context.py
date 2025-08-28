"""Tests for the ParentContextDetector using the new API."""

import json
import os
import pytest
from unittest.mock import patch, MagicMock

from sense_disambiguation.detector.parent_context import ParentContextDetector
from sense_disambiguation.detector.base import EvidenceBuilder

class TestParentContextDetector:
    """Test for the ParentContextDetector class."""
    
    @patch('sense_disambiguation.detector.parent_context.ParentContextDetector.detect_ambiguous_terms')
    def test_detect_method(self, mock_detect_ambiguous_terms):
        """Test the new detect method returns valid EvidenceBuilder objects."""
        # Setup detector
        detector = ParentContextDetector(
            hierarchy_file_path="dummy_path.json",
            final_term_files_pattern="dummy_pattern.txt"
        )
        
        # Mock the legacy detect_ambiguous_terms to return a list of terms
        mock_detect_ambiguous_terms.return_value = ["machine_learning", "graph_theory"]
        
        # Mock detailed_results that would be populated by the legacy method
        detector.detailed_results = {
            "machine_learning": {
                "level": 2,
                "parent_count": 2,
                "ancestor_contexts": [
                    [["computer_science"], ["artificial_intelligence"]],
                    [["statistics"], ["data_science"]]
                ],
                "parents_details": {
                    "ai": {"l0": ["computer_science"], "l1": ["artificial_intelligence"]},
                    "data_analysis": {"l0": ["statistics"], "l1": ["data_science"]}
                }
            },
            "graph_theory": {
                "level": 3,
                "parent_count": 2,
                "ancestor_contexts": [
                    [["mathematics"], ["discrete_math"]],
                    [["computer_science"], ["algorithms"]]
                ],
                "parents_details": {
                    "discrete_mathematics": {"l0": ["mathematics"], "l1": ["discrete_math"]},
                    "algorithms": {"l0": ["computer_science"], "l1": ["algorithms"]}
                }
            }
        }
        
        # Call the new detect method
        evidence_builders = detector.detect()
        
        # Basic checks
        assert len(evidence_builders) == 2
        assert all(isinstance(eb, EvidenceBuilder) for eb in evidence_builders)
        
        # Check specific fields for each evidence builder
        ml_evidence = next(eb for eb in evidence_builders if eb.term == "machine_learning")
        assert ml_evidence.level == 2
        assert ml_evidence.evidence.source == "parent_context"
        assert ml_evidence.evidence.confidence > 0
        assert ml_evidence.evidence.metrics["distinct_ancestor_pairs_count"] == 2
        assert ml_evidence.evidence.payload["divergent"] is True
        assert len(ml_evidence.evidence.payload["parents"]) == 2
        
        # Check JSON serialization
        evidence_json = ml_evidence.evidence.model_dump_json()
        evidence_data = json.loads(evidence_json)
        assert "source" in evidence_data
        assert "detector_version" in evidence_data
        assert "confidence" in evidence_data
        assert "metrics" in evidence_data
        assert "payload" in evidence_data 