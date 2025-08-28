"""Tests for the HybridAmbiguityDetector using the new unified API."""

import json
import os
import pytest
from unittest.mock import patch, MagicMock

from sense_disambiguation.detector.hybrid import HybridAmbiguityDetector
from sense_disambiguation.detector.base import EvidenceBuilder, TermContext

class TestHybridAmbiguityDetector:
    """Tests for the HybridAmbiguityDetector class."""
    
    @patch('sense_disambiguation.detector.hybrid.ParentContextDetector.detect')
    @patch('sense_disambiguation.detector.hybrid.ResourceClusterDetector.detect')
    @patch('sense_disambiguation.detector.hybrid.RadialPolysemyDetector.detect')
    def test_detect_method(self, mock_radial_detect, mock_dbscan_detect, mock_parent_detect):
        """Test the new detect method correctly merges evidence from all detectors."""
        # Setup detector
        detector = HybridAmbiguityDetector(
            hierarchy_file_path="dummy_path.json",
            final_term_files_pattern="dummy_pattern.txt",
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create mock evidence builders for each detector
        # Parent context detector evidence
        parent_evidence = [
            EvidenceBuilder.create(
                term="neural_network",
                level=2,
                source="parent_context",
                detector_version="2025.05.20",
                confidence=0.8,
                metrics={"distinct_ancestor_pairs_count": 4},
                payload={"distinct_ancestors": [["ai", "deep_learning"], ["biology", "neuroscience"]]}
            ),
            EvidenceBuilder.create(
                term="tree",
                level=1,
                source="parent_context",
                detector_version="2025.05.20",
                confidence=0.6,
                metrics={"distinct_ancestor_pairs_count": 3},
                payload={"distinct_ancestors": [["botany", "plants"], ["computer_science", "data_structures"]]}
            )
        ]
        
        # Resource cluster detector evidence
        dbscan_evidence = [
            EvidenceBuilder.create(
                term="neural_network",
                level=2,
                source="resource_cluster",
                detector_version="2025.05.20",
                confidence=0.75,
                metrics={"separation_score": 0.7, "silhouette_score": 0.6, "num_clusters": 2},
                payload={"cluster_labels": [0, 1, 0, 1, 2, 0]}
            ),
            EvidenceBuilder.create(
                term="machine_learning",
                level=2,
                source="resource_cluster",
                detector_version="2025.05.20",
                confidence=0.6,
                metrics={"separation_score": 0.5, "silhouette_score": 0.4, "num_clusters": 2},
                payload={"cluster_labels": [0, 1, 0, 1]}
            )
        ]
        
        # Radial polysemy detector evidence
        radial_evidence = [
            EvidenceBuilder.create(
                term="neural_network",
                level=2,
                source="radial_polysemy",
                detector_version="2025.05.20",
                confidence=0.7,
                metrics={"polysemy_index": 0.6, "context_count": 40},
                payload={"sample_contexts": ["neural networks in deep learning", "biological neural networks"]}
            ),
            EvidenceBuilder.create(
                term="bank",
                level=1,
                source="radial_polysemy",
                detector_version="2025.05.20",
                confidence=0.85,
                metrics={"polysemy_index": 0.8, "context_count": 60},
                payload={"sample_contexts": ["bank account", "river bank"]}
            )
        ]
        
        # Set up mock returns
        mock_parent_detect.return_value = parent_evidence
        mock_dbscan_detect.return_value = dbscan_evidence
        mock_radial_detect.return_value = radial_evidence
        
        # Call the new detect method
        term_contexts = detector.detect()
        
        # Check that we got the right terms
        expected_terms = {"neural_network", "tree", "machine_learning", "bank"}
        assert set(term_contexts.keys()) == expected_terms
        
        # Check neural_network which should have evidence from all detectors
        nn_context = term_contexts["neural_network"]
        assert nn_context.level == 2
        assert len(nn_context.evidence) == 3  # One from each detector
        assert nn_context.overall_confidence > 0.9  # Should be high due to multiple signals
        
        # Check sources represented in the evidence
        evidence_sources = {e.source for e in nn_context.evidence}
        assert evidence_sources == {"parent_context", "resource_cluster", "radial_polysemy"}
        
        # Check bank which only has radial evidence
        bank_context = term_contexts["bank"]
        assert len(bank_context.evidence) == 1
        assert bank_context.evidence[0].source == "radial_polysemy"
        assert bank_context.overall_confidence == 0.85  # Equal to the only evidence's confidence
        
    @patch('sense_disambiguation.detector.hybrid.HybridAmbiguityDetector.detect')
    def test_save_unified_context(self, mock_detect):
        """Test saving the unified context to a file."""
        # Setup detector and temporary directory for output
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        detector = HybridAmbiguityDetector(
            hierarchy_file_path="dummy_path.json",
            final_term_files_pattern="dummy_pattern.txt",
            model_name="all-MiniLM-L6-v2",
            output_dir=temp_dir
        )
        
        # Create mock term contexts to save
        term_contexts = {
            "neural_network": TermContext(
                canonical_name="neural_network",
                level=2,
                overall_confidence=0.95,
                evidence=[
                    EvidenceBuilder.create(
                        term="neural_network",
                        level=2,
                        source="parent_context",
                        detector_version="2025.05.20",
                        confidence=0.8,
                        metrics={"distinct_ancestor_pairs_count": 4},
                        payload={"distinct_ancestors": [["ai", "deep_learning"], ["biology", "neuroscience"]]}
                    ).evidence
                ]
            )
        }
        
        # Set up mock return
        mock_detect.return_value = term_contexts
        
        # Call detect_and_save
        _, output_path = detector.detect_and_save()
        
        # Check the file was created and contains valid JSON
        assert os.path.exists(output_path)
        
        with open(output_path, 'r') as f:
            saved_data = json.load(f)
        
        # Check that the output contains the expected structure
        assert "schema_version" in saved_data
        assert "contexts" in saved_data
        assert "neural_network" in saved_data["contexts"]
        assert saved_data["contexts"]["neural_network"]["overall_confidence"] == 0.95
        
        # Clean up the temporary file
        os.remove(output_path)
        os.rmdir(temp_dir) 