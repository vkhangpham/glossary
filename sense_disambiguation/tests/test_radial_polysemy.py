"""Tests for the RadialPolysemyDetector using the new API."""

import json
import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from sense_disambiguation.detector.radial_polysemy import RadialPolysemyDetector
from sense_disambiguation.detector.base import EvidenceBuilder

class TestRadialPolysemyDetector:
    """Tests for the RadialPolysemyDetector class."""
    
    @patch('sense_disambiguation.detector.radial_polysemy.RadialPolysemyDetector.detect_ambiguous_terms')
    @patch('sense_disambiguation.detector.radial_polysemy.RadialPolysemyDetector._load_data')
    def test_detect_method(self, mock_load_data, mock_detect_ambiguous_terms):
        """Test the new detect method returns valid EvidenceBuilder objects."""
        # Setup detector
        detector = RadialPolysemyDetector(
            hierarchy_file_path="dummy_path.json",
            final_term_files_pattern="dummy_pattern.txt",
            model_name="all-MiniLM-L6-v2"
        )
        
        # Mock return values
        mock_load_data.return_value = True
        mock_detect_ambiguous_terms.return_value = ["neural_network", "machine_learning"]
        
        # Mock polysemy_scores
        detector.polysemy_scores = {
            "neural_network": {
                "polysemy_index": 0.70,
                "context_count": 50,
                "metrics": {
                    "peak_count_estimate": 3,
                    "variance": 0.25,
                    "best_peak_position": 0.33
                }
            },
            "machine_learning": {
                "polysemy_index": 0.45,
                "context_count": 35,
                "metrics": {
                    "peak_count_estimate": 2,
                    "variance": 0.18,
                    "best_peak_position": 0.45
                }
            },
            "low_confidence_term": {
                "polysemy_index": 0.04,  # Below threshold, should be excluded
                "context_count": 25,
                "metrics": {
                    "peak_count_estimate": 1,
                    "variance": 0.05
                }
            }
        }
        
        # Mock term_details
        detector.term_details = {
            "neural_network": {
                "level": 2
            },
            "machine_learning": {
                "level": 1
            },
            "low_confidence_term": {
                "level": 3
            }
        }
        
        # Mock _extract_context_terms method to return sample contexts
        def mock_extract_contexts(term):
            if term == "neural_network":
                return [
                    ["neural", "networks", "artificial", "intelligence", "deep", "learning"],
                    ["biological", "inspiration", "neural", "networks", "brain"],
                    ["convolutional", "neural", "networks", "image", "recognition"]
                ]
            elif term == "machine_learning":
                return [
                    ["supervised", "learning", "algorithms", "training", "data"],
                    ["machine", "learning", "models", "classification", "regression"]
                ]
            return []
            
        detector._extract_context_terms = mock_extract_contexts
        
        # Call the new detect method
        evidence_builders = detector.detect()
        
        # Basic checks
        assert len(evidence_builders) == 2  # Should exclude the low confidence term
        assert all(isinstance(eb, EvidenceBuilder) for eb in evidence_builders)
        
        # Check specific fields for neural_network evidence
        nn_evidence = next(eb for eb in evidence_builders if eb.term == "neural_network")
        assert nn_evidence.level == 2
        assert nn_evidence.evidence.source == "radial_polysemy"
        assert nn_evidence.evidence.confidence > 0.9  # Should be high based on polysemy_index of 0.7
        assert nn_evidence.evidence.metrics["polysemy_index"] == 0.70
        assert nn_evidence.evidence.metrics["context_count"] == 50
        assert nn_evidence.evidence.metrics["peak_count_estimate"] == 3
        assert len(nn_evidence.evidence.payload["sample_contexts"]) > 0
        
        # Check machine_learning evidence
        ml_evidence = next(eb for eb in evidence_builders if eb.term == "machine_learning")
        assert ml_evidence.level == 1
        assert ml_evidence.evidence.confidence > 0.6  # Should be moderate based on polysemy_index of 0.45
        assert ml_evidence.evidence.payload["context_count"] == 35
        assert len(ml_evidence.evidence.payload["sample_contexts"]) > 0
        
        # Check JSON serialization
        evidence_json = nn_evidence.evidence.model_dump_json()
        evidence_data = json.loads(evidence_json)
        assert "source" in evidence_data
        assert "detector_version" in evidence_data
        assert "confidence" in evidence_data
        assert "metrics" in evidence_data
        assert "payload" in evidence_data
        assert "sample_contexts" in evidence_data["payload"] 