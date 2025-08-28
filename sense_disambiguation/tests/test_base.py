"""Tests for the base abstractions used in the detector pipeline."""

import json
import pytest
from sense_disambiguation.detector.base import (
    EvidenceBlock, 
    TermContext, 
    EvidenceBuilder,
    merge_term_contexts,
    get_detector_version
)

def test_evidence_block_creation():
    """Test that EvidenceBlock can be created with valid data."""
    evidence = EvidenceBlock(
        source="resource_cluster",
        detector_version="2025.05.22",
        confidence=0.8,
        metrics={"separation_score": 0.7},
        payload={"cluster_labels": [0, 1, 0, 2, 1]}
    )
    
    assert evidence.source == "resource_cluster"
    assert evidence.confidence == 0.8
    assert evidence.metrics["separation_score"] == 0.7
    assert evidence.payload["cluster_labels"] == [0, 1, 0, 2, 1]

def test_evidence_serialization():
    """Test that EvidenceBlock serializes to JSON properly."""
    evidence = EvidenceBlock(
        source="resource_cluster",
        detector_version="2025.05.22",
        confidence=0.8,
        metrics={"separation_score": 0.7},
        payload={"cluster_labels": [0, 1, 0, 2, 1]}
    )
    
    # Serialize to JSON
    json_str = evidence.model_dump_json()
    data = json.loads(json_str)
    
    # Check that fields are present
    assert "source" in data
    assert "confidence" in data
    assert "metrics" in data
    assert "payload" in data
    
    # Check that values match
    assert data["source"] == "resource_cluster"
    assert data["confidence"] == 0.8
    assert data["metrics"]["separation_score"] == 0.7
    assert data["payload"]["cluster_labels"] == [0, 1, 0, 2, 1]
    
    # Deserialize and check that it's the same
    evidence2 = EvidenceBlock.model_validate_json(json_str)
    assert evidence2.source == evidence.source
    assert evidence2.confidence == evidence.confidence
    assert evidence2.metrics == evidence.metrics
    assert evidence2.payload == evidence.payload

def test_term_context_serialization():
    """Test that TermContext serializes to JSON properly."""
    context = TermContext(
        canonical_name="machine_learning",
        level=2,
        overall_confidence=0.8,
        evidence=[
            EvidenceBlock(
                source="resource_cluster",
                detector_version="2025.05.22",
                confidence=0.8,
                metrics={"separation_score": 0.7},
                payload={"cluster_labels": [0, 1, 0, 2, 1]}
            ),
            EvidenceBlock(
                source="parent_context",
                detector_version="2025.05.22",
                confidence=0.6,
                metrics={"distinct_ancestors_count": 2},
                payload={"parents": ["ai", "algorithms"]}
            )
        ]
    )
    
    # Serialize to JSON
    json_str = context.model_dump_json()
    data = json.loads(json_str)
    
    # Check that fields are present
    assert "canonical_name" in data
    assert "level" in data
    assert "overall_confidence" in data
    assert "evidence" in data
    
    # Check that values match
    assert data["canonical_name"] == "machine_learning"
    assert data["level"] == 2
    assert data["overall_confidence"] == 0.8
    assert len(data["evidence"]) == 2
    
    # Check evidence blocks
    assert data["evidence"][0]["source"] == "resource_cluster"
    assert data["evidence"][1]["source"] == "parent_context"
    
    # Deserialize and check that it's the same
    context2 = TermContext.model_validate_json(json_str)
    assert context2.canonical_name == context.canonical_name
    assert context2.level == context.level
    assert context2.overall_confidence == context.overall_confidence
    assert len(context2.evidence) == len(context.evidence)
    
def test_evidence_builder():
    """Test that EvidenceBuilder works as expected."""
    builder = EvidenceBuilder.create(
        term="machine_learning",
        level=2,
        source="resource_cluster",
        detector_version="2025.05.22",
        confidence=0.8,
        metrics={"separation_score": 0.7},
        payload={"cluster_labels": [0, 1, 0, 2, 1]}
    )
    
    assert builder.term == "machine_learning"
    assert builder.level == 2
    assert builder.evidence.source == "resource_cluster"
    assert builder.evidence.confidence == 0.8
    
def test_merge_term_contexts():
    """Test that merge_term_contexts works as expected."""
    builders = [
        EvidenceBuilder.create(
            term="machine_learning",
            level=2,
            source="resource_cluster",
            detector_version="2025.05.22",
            confidence=0.8,
            metrics={"separation_score": 0.7},
            payload={"cluster_labels": [0, 1, 0, 2, 1]}
        ),
        EvidenceBuilder.create(
            term="machine_learning",
            level=2,
            source="parent_context",
            detector_version="2025.05.22",
            confidence=0.6,
            metrics={"distinct_ancestors_count": 2},
            payload={"parents": ["ai", "algorithms"]}
        ),
        EvidenceBuilder.create(
            term="deep_learning",
            level=3,
            source="resource_cluster",
            detector_version="2025.05.22",
            confidence=0.9,
            metrics={"separation_score": 0.8},
            payload={"cluster_labels": [0, 1, 0, 1]}
        )
    ]
    
    merged = merge_term_contexts(*builders)
    
    # Check that we have two terms
    assert len(merged) == 2
    assert "machine_learning" in merged
    assert "deep_learning" in merged
    
    # Check machine_learning term
    ml_context = merged["machine_learning"]
    assert ml_context.canonical_name == "machine_learning"
    assert ml_context.level == 2
    # Confidence should be higher than either individual source
    assert ml_context.overall_confidence > 0.8
    assert len(ml_context.evidence) == 2
    
    # Check deep_learning term
    dl_context = merged["deep_learning"]
    assert dl_context.canonical_name == "deep_learning"
    assert dl_context.level == 3
    assert dl_context.overall_confidence == 0.9  # Only one source
    assert len(dl_context.evidence) == 1
    
def test_empty_merge():
    """Test that merge_term_contexts handles empty input properly."""
    merged = merge_term_contexts()
    assert merged == {} 