"""Tests for the SenseSplitter using the new unified context API."""

import json
import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock

from sense_disambiguation.splitter import SenseSplitter

class TestSenseSplitter:
    """Tests for the SenseSplitter class using the new unified context API."""
    
    def test_init_with_context_file(self):
        """Test initializing the splitter with a context file."""
        # Create a temporary unified context file
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w+", delete=False) as f:
            # Write example unified context content
            context_data = {
                "schema_version": "v1",
                "timestamp": "2025-05-22T12:34:56",
                "contexts": {
                    "neural_network": {
                        "canonical_name": "neural_network",
                        "level": 2,
                        "overall_confidence": 0.95,
                        "evidence": [
                            {
                                "source": "resource_cluster",
                                "detector_version": "2025.05.22",
                                "confidence": 0.75,
                                "metrics": {
                                    "separation_score": 0.7,
                                    "silhouette_score": 0.6,
                                    "num_clusters": 2
                                },
                                "payload": {
                                    "cluster_labels": [0, 1, 0, 1, 2, 0],
                                    "eps": 0.4,
                                    "min_samples": 2
                                }
                            },
                            {
                                "source": "parent_context",
                                "detector_version": "2025.05.22",
                                "confidence": 0.8,
                                "metrics": {
                                    "distinct_ancestor_pairs_count": 2
                                },
                                "payload": {
                                    "divergent": True,
                                    "distinct_ancestors": [
                                        ["computer_science", "ai", "deep_learning"],
                                        ["biology", "neuroscience"]
                                    ]
                                }
                            }
                        ]
                    },
                    "tree": {
                        "canonical_name": "tree",
                        "level": 1,
                        "overall_confidence": 0.85,
                        "evidence": [
                            {
                                "source": "parent_context",
                                "detector_version": "2025.05.22",
                                "confidence": 0.85,
                                "metrics": {
                                    "distinct_ancestor_pairs_count": 2
                                },
                                "payload": {
                                    "divergent": True,
                                    "distinct_ancestors": [
                                        ["botany", "plants"],
                                        ["computer_science", "data_structures"]
                                    ]
                                }
                            }
                        ]
                    }
                }
            }
            json.dump(context_data, f)
            f.flush()
            
            # Initialize splitter with the context file
            try:
                splitter = SenseSplitter(
                    hierarchy_file_path="dummy_path.json",
                    context_file=f.name,
                    level=2
                )
                
                # Check that the context was loaded properly
                assert "neural_network" in splitter.term_contexts
                assert "tree" in splitter.term_contexts
                
                # Check that the candidate terms were extracted
                assert len(splitter.candidate_terms) == 2
                assert "neural_network" in splitter.candidate_terms
                assert "tree" in splitter.candidate_terms
                
                # Check cluster results were extracted
                assert "neural_network" in splitter.cluster_results
                assert splitter.cluster_results["neural_network"] == [0, 1, 0, 1, 2, 0]
                
                # Check metrics were extracted
                assert "neural_network" in splitter.cluster_metrics
                assert splitter.cluster_metrics["neural_network"]["separation_score"] == 0.7
                
            finally:
                # Clean up the temporary file
                os.unlink(f.name)
    
    @patch("sense_disambiguation.splitter.SenseSplitter._load_hierarchy")
    def test_filter_candidate_terms_by_level(self, mock_load_hierarchy):
        """Test filtering terms by level using the context data."""
        mock_load_hierarchy.return_value = True
        
        # Create a splitter with mock context data
        splitter = SenseSplitter(
            hierarchy_file_path="dummy_path.json",
            level=2
        )
        
        # Set up mock term contexts with different levels
        splitter.term_contexts = {
            "neural_network": {
                "level": 2,
                "evidence": []
            },
            "tree": {
                "level": 1,
                "evidence": []
            },
            "machine_learning": {
                "level": 2,
                "evidence": []
            },
            "computer": {
                "level": 3,
                "evidence": []
            }
        }
        
        # Test filtering by level
        filtered_terms = splitter._filter_candidate_terms_by_level()
        
        # Should get only the level 2 terms
        assert len(filtered_terms) == 2
        assert "neural_network" in filtered_terms
        assert "machine_learning" in filtered_terms
        assert "tree" not in filtered_terms
        assert "computer" not in filtered_terms
    
    @patch("sense_disambiguation.splitter.SenseSplitter._load_hierarchy")
    def test_get_parent_context_evidence(self, mock_load_hierarchy):
        """Test getting parent context evidence from the context data."""
        mock_load_hierarchy.return_value = True
        
        # Create a splitter with mock context data
        splitter = SenseSplitter(
            hierarchy_file_path="dummy_path.json",
            level=2
        )
        
        # Set up mock term contexts
        splitter.term_contexts = {
            "neural_network": {
                "level": 2,
                "evidence": [
                    {
                        "source": "parent_context",
                        "confidence": 0.8,
                        "payload": {
                            "divergent": True,
                            "distinct_ancestors": [
                                ["ai", "deep_learning"],
                                ["biology", "neuroscience"]
                            ]
                        }
                    }
                ]
            }
        }
        
        # Test getting evidence
        evidence = splitter._get_parent_context_evidence("neural_network")
        
        # Check evidence was retrieved correctly
        assert evidence is not None
        assert evidence["source"] == "parent_context"
        assert evidence["confidence"] == 0.8
        assert evidence["payload"]["divergent"] is True
        
        # Test getting evidence for non-existent term
        evidence = splitter._get_parent_context_evidence("nonexistent_term")
        assert evidence is None 