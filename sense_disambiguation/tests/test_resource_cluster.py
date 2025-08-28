"""Tests for the ResourceClusterDetector using the new API."""

import json
import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from sense_disambiguation.detector.resource_cluster import ResourceClusterDetector
from sense_disambiguation.detector.base import EvidenceBuilder

class TestResourceClusterDetector:
    """Tests for the ResourceClusterDetector class."""
    
    @patch('sense_disambiguation.detector.resource_cluster.ResourceClusterDetector.detect_ambiguous_terms')
    @patch('sense_disambiguation.detector.resource_cluster.ResourceClusterDetector._load_data')
    def test_detect_method(self, mock_load_data, mock_detect_ambiguous_terms):
        """Test the new detect method returns valid EvidenceBuilder objects."""
        # Setup detector
        detector = ResourceClusterDetector(
            hierarchy_file_path="dummy_path.json",
            final_term_files_pattern="dummy_pattern.txt",
            model_name="all-MiniLM-L6-v2"
        )
        
        # Mock return values
        mock_load_data.return_value = True
        mock_detect_ambiguous_terms.return_value = ["machine_learning", "graph_theory"]
        
        # Mock cluster_results and cluster_metrics
        detector.cluster_results = {
            "machine_learning": [0, 1, 0, 2, 1],
            "graph_theory": [0, 1, 2, 0]
        }
        
        detector.cluster_metrics = {
            "machine_learning": {
                "level": 2,
                "num_clusters": 3,
                "separation_score": 0.7,
                "silhouette_score": 0.6,
                "noise_points": 0,
                "valid_snippets": 5,
                "tfidf_confidence": 0.8
            },
            "graph_theory": {
                "level": 3,
                "num_clusters": 3,
                "separation_score": 0.6,
                "silhouette_score": 0.5,
                "noise_points": 0,
                "valid_snippets": 4,
                "tfidf_confidence": 0.7
            }
        }
        
        # Mock term_details
        detector.term_details = {
            "machine_learning": {
                "level": 2,
                "resources": [
                    {"url": "https://example.com/ml1", "title": "ML Article 1", "processed_content": "Machine learning content 1"},
                    {"url": "https://example.com/ml2", "title": "ML Article 2", "processed_content": "Machine learning content 2"},
                    {"url": "https://example.com/ml3", "title": "ML Article 3", "processed_content": "Machine learning content 3"},
                    {"url": "https://example.com/ml4", "title": "ML Article 4", "processed_content": "Machine learning content 4"},
                    {"url": "https://example.com/ml5", "title": "ML Article 5", "processed_content": "Machine learning content 5"}
                ]
            },
            "graph_theory": {
                "level": 3,
                "resources": [
                    {"url": "https://example.com/gt1", "title": "GT Article 1", "processed_content": "Graph theory content 1"},
                    {"url": "https://example.com/gt2", "title": "GT Article 2", "processed_content": "Graph theory content 2"},
                    {"url": "https://example.com/gt3", "title": "GT Article 3", "processed_content": "Graph theory content 3"},
                    {"url": "https://example.com/gt4", "title": "GT Article 4", "processed_content": "Graph theory content 4"}
                ]
            }
        }
        
        # Call the new detect method
        evidence_builders = detector.detect()
        
        # Basic checks
        assert len(evidence_builders) == 2
        assert all(isinstance(eb, EvidenceBuilder) for eb in evidence_builders)
        
        # Check specific fields for machine_learning evidence
        ml_evidence = next(eb for eb in evidence_builders if eb.term == "machine_learning")
        assert ml_evidence.level == 2
        assert ml_evidence.evidence.source == "resource_cluster"
        assert ml_evidence.evidence.confidence > 0.6  # Should be high based on our mock metrics
        assert ml_evidence.evidence.metrics["separation_score"] == 0.7
        assert ml_evidence.evidence.metrics["num_clusters"] == 3
        assert ml_evidence.evidence.payload["cluster_labels"] == [0, 1, 0, 2, 1]
        
        # Check graph_theory evidence
        gt_evidence = next(eb for eb in evidence_builders if eb.term == "graph_theory")
        assert gt_evidence.level == 3
        assert gt_evidence.evidence.payload["cluster_labels"] == [0, 1, 2, 0]
        
        # Check JSON serialization
        evidence_json = ml_evidence.evidence.model_dump_json()
        evidence_data = json.loads(evidence_json)
        assert "source" in evidence_data
        assert "detector_version" in evidence_data
        assert "confidence" in evidence_data
        assert "metrics" in evidence_data
        assert "payload" in evidence_data
        assert "cluster_labels" in evidence_data["payload"] 