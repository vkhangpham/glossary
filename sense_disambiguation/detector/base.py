"""
Base abstractions for ambiguity detectors in the sense disambiguation pipeline.

This module defines the core data models and interfaces for all detectors
to ensure consistent outputs and a unified schema for interacting with the splitter.
"""

from typing import Literal, Dict, List, Any, TypeVar, Optional, NamedTuple
import datetime
from pydantic import BaseModel, Field

# Type definitions for sources and payloads
EvidenceSource = Literal["resource_cluster", "parent_context", "radial_polysemy"]
EvidencePayload = Dict[str, Any]  # Type-erased but detector-specific payload

class EvidenceBlock(BaseModel):
    """
    Evidence block provided by a detector for an ambiguous term.
    Each detector contributes one or more evidence blocks per detected term.
    """
    source: EvidenceSource = Field(
        description="Source detector that produced this evidence"
    )
    detector_version: str = Field(
        description="Version string of the detector"
    )
    confidence: float = Field(
        description="Confidence score [0.0-1.0] for this evidence",
        ge=0.0, 
        le=1.0
    )
    metrics: Dict[str, Any] = Field(
        description="Numeric/boolean metrics for explaining the detection",
        default_factory=dict
    )
    payload: EvidencePayload = Field(
        description="Source-specific payload data. Contents depend on evidence source.",
        default_factory=dict
    )

class TermContext(BaseModel):
    """
    Complete context for an ambiguous term, aggregating evidence from all detectors.
    The splitter consumes this data structure to make sense-splitting decisions.
    """
    canonical_name: str = Field(
        description="Primary canonical form of the term"
    )
    level: Optional[int] = Field(
        description="Hierarchy level (0-3) if available",
        ge=0,
        le=3,
        default=None
    )
    overall_confidence: float = Field(
        description="Aggregated confidence [0.0-1.0] from all evidence sources",
        ge=0.0,
        le=1.0
    )
    evidence: List[EvidenceBlock] = Field(
        description="Evidence blocks from all contributing detectors",
        default_factory=list
    )

class EvidenceBuilder(NamedTuple):
    """Helper for building evidence for detected terms."""
    term: str
    level: Optional[int]
    evidence: EvidenceBlock
    
    @classmethod
    def create(cls, 
               term: str, 
               level: Optional[int], 
               source: EvidenceSource,
               detector_version: str,
               confidence: float,
               metrics: Optional[Dict[str, Any]] = None,
               payload: Optional[EvidencePayload] = None) -> 'EvidenceBuilder':
        """Factory method to create an EvidenceBuilder with complete evidence block."""
        evidence = EvidenceBlock(
            source=source,
            detector_version=detector_version,
            confidence=confidence,
            metrics=metrics or {},
            payload=payload or {}
        )
        return cls(term=term, level=level, evidence=evidence)

# Type variable for the generic function
T = TypeVar('T', bound=TermContext)

def merge_term_contexts(*blocks: EvidenceBuilder) -> Dict[str, TermContext]:
    """
    Merge multiple EvidenceBuilder instances into a dictionary of TermContext objects.
    
    Args:
        *blocks: Variable number of EvidenceBuilder namedtuples
        
    Returns:
        Dictionary mapping term strings to their aggregated TermContext
    """
    if not blocks:
        return {}
        
    # Group by term
    term_to_blocks: Dict[str, List[EvidenceBuilder]] = {}
    for block in blocks:
        if block.term not in term_to_blocks:
            term_to_blocks[block.term] = []
        term_to_blocks[block.term].append(block)
    
    # Merge evidence for each term
    result: Dict[str, TermContext] = {}
    for term, term_blocks in term_to_blocks.items():
        # Use first non-None level, or None if all are None
        level = next((b.level for b in term_blocks if b.level is not None), None)
        
        # Calculate overall confidence using noisy-OR model: 1 - ∏(1-conf_i)
        confidence_values = [b.evidence.confidence for b in term_blocks]
        if not confidence_values:
            overall_confidence = 0.0
        else:
            # Noisy-OR aggregation: P(A or B) = 1 - (1-P(A)) * (1-P(B))
            # This gives higher weight when multiple detectors agree
            overall_confidence = 1.0 - (
                (1.0 - confidence_values[0]) if len(confidence_values) == 1 
                else 
                # Product of (1-confidence) values
                # End formula: 1 - ∏(1-conf_i)
                prod([1.0 - conf for conf in confidence_values])
            )
            
        # Collect all evidence blocks
        all_evidence = [b.evidence for b in term_blocks]
        
        # Create the merged context
        context = TermContext(
            canonical_name=term,
            level=level,
            overall_confidence=overall_confidence,
            evidence=all_evidence
        )
        result[term] = context
    
    return result

def prod(values: List[float]) -> float:
    """Calculate the product of a list of values."""
    result = 1.0
    for v in values:
        result *= v
    return result

def get_detector_version() -> str:
    """Return the current detector version as a date-based string."""
    return datetime.datetime.now().strftime("%Y.%m.%d")

# For tests and debug
def create_test_term_context(term: str, level: int = 2) -> TermContext:
    """Create a test TermContext with minimal evidence for unit testing."""
    return TermContext(
        canonical_name=term,
        level=level,
        overall_confidence=0.8,
        evidence=[
            EvidenceBlock(
                source="resource_cluster",
                detector_version=get_detector_version(),
                confidence=0.8,
                metrics={"separation_score": 0.7},
                payload={"cluster_labels": [0, 1, 0, 2, 1]}
            )
        ]
    ) 