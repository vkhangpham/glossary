#!/usr/bin/env python3
"""
Test Dataset Build and Validation Script

This script builds and validates the synthetic test dataset used for testing
the detector → unified-context → splitter pipeline.

Usage:
    python build_test_dataset.py [--validate-only] [--rebuild]
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Add the project root to sys.path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

try:
    from sense_disambiguation.detector.base import TermContext, EvidenceBlock
    from pydantic import ValidationError
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please install the required dependencies: pip install -r requirements.txt")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dataset configuration
DATASET_ROOT = project_root / "sense_disambiguation" / "data" / "test_dataset"
HIERARCHY_FILE = DATASET_ROOT / "hierarchy.json"
UNIFIED_CONTEXT_FILE = DATASET_ROOT / "unified_context_ground_truth.json"
RAW_RESOURCES_DIR = DATASET_ROOT / "raw_resources"

# Expected terms (ground truth)
POSITIVE_TERMS = [
    "transformers", "interface", "modeling", "fragmentation", "clustering",
    "stress", "regression", "cell", "network", "bond"
]

NEGATIVE_TERMS = [
    "artificial intelligence", "mathematics", "engineering", "geology",
    "astrophysics", "botany", "microbiology", "cryptography"
]

ALL_TERMS = POSITIVE_TERMS + NEGATIVE_TERMS

# Expected structure
EXPECTED_POSITIVE_CONFIDENCE_THRESHOLD = 0.7
EXPECTED_NEGATIVE_CONFIDENCE_THRESHOLD = 0.3


class DatasetValidator:
    """Validates the test dataset structure and content."""
    
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root
        self.errors = []
        self.warnings = []
        
    def validate_all(self) -> bool:
        """Run all validation checks."""
        logger.info("Starting comprehensive dataset validation...")
        
        # Check basic structure
        if not self._validate_directory_structure():
            return False
            
        # Validate hierarchy file
        if not self._validate_hierarchy():
            return False
            
        # Validate unified context file  
        if not self._validate_unified_context():
            return False
            
        # Cross-validation checks
        if not self._validate_cross_references():
            return False
            
        # Report results
        self._report_results()
        return len(self.errors) == 0
        
    def _validate_directory_structure(self) -> bool:
        """Check that required directories and files exist."""
        logger.info("Validating directory structure...")
        
        required_paths = [
            self.dataset_root,
            HIERARCHY_FILE,
            UNIFIED_CONTEXT_FILE,
            RAW_RESOURCES_DIR
        ]
        
        for path in required_paths:
            if not path.exists():
                self.errors.append(f"Missing required path: {path}")
                
        return len(self.errors) == 0
        
    def _validate_hierarchy(self) -> bool:
        """Validate hierarchy.json structure and content."""
        logger.info("Validating hierarchy.json...")
        
        try:
            with open(HIERARCHY_FILE, 'r') as f:
                hierarchy = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.errors.append(f"Error loading hierarchy.json: {e}")
            return False
            
        # Check top-level structure
        if 'terms' not in hierarchy:
            self.errors.append("hierarchy.json missing 'terms' key")
            return False
            
        terms = hierarchy['terms']
        
        # Check all expected terms are present
        missing_terms = set(ALL_TERMS) - set(terms.keys())
        if missing_terms:
            self.errors.append(f"Missing terms in hierarchy: {missing_terms}")
            
        # Validate each term structure
        for term_name in ALL_TERMS:
            if term_name not in terms:
                continue
                
            term_data = terms[term_name]
            
            # Check required fields
            required_fields = ['level', 'parents', 'resources']
            for field in required_fields:
                if field not in term_data:
                    self.errors.append(f"Term '{term_name}' missing field '{field}'")
                    
            # Check level is 2
            if term_data.get('level') != 2:
                self.errors.append(f"Term '{term_name}' has incorrect level: {term_data.get('level')}")
                
            # Check resources count
            resources = term_data.get('resources', [])
            if len(resources) < 7:
                self.warnings.append(f"Term '{term_name}' has only {len(resources)} resources (minimum 7 recommended)")
                
            # Validate resource structure
            for i, resource in enumerate(resources):
                required_res_fields = ['url', 'title', 'processed_content']
                for field in required_res_fields:
                    if field not in resource:
                        self.errors.append(f"Term '{term_name}' resource {i} missing field '{field}'")
                        
        return len(self.errors) == 0
        
    def _validate_unified_context(self) -> bool:
        """Validate unified_context_ground_truth.json using Pydantic models."""
        logger.info("Validating unified context file...")
        
        try:
            with open(UNIFIED_CONTEXT_FILE, 'r') as f:
                context_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.errors.append(f"Error loading unified context: {e}")
            return False
            
        # Check top-level structure
        if 'contexts' not in context_data:
            self.errors.append("unified_context_ground_truth.json missing 'contexts' key")
            return False
            
        contexts = context_data['contexts']
        
        # Check all expected terms are present
        missing_terms = set(ALL_TERMS) - set(contexts.keys())
        if missing_terms:
            self.errors.append(f"Missing terms in unified context: {missing_terms}")
            
        # Validate each term context using Pydantic
        for term_name in ALL_TERMS:
            if term_name not in contexts:
                continue
                
            try:
                # Validate using Pydantic model
                term_context = TermContext(**contexts[term_name])
                
                # Additional semantic validations
                self._validate_term_context_semantics(term_name, term_context)
                
            except ValidationError as e:
                self.errors.append(f"Validation error for term '{term_name}': {e}")
                
        return len(self.errors) == 0
        
    def _validate_term_context_semantics(self, term_name: str, context: TermContext):
        """Validate semantic correctness of term context."""
        
        # Check confidence levels match expected ambiguity
        if term_name in POSITIVE_TERMS:
            if context.overall_confidence < EXPECTED_POSITIVE_CONFIDENCE_THRESHOLD:
                self.warnings.append(f"Positive term '{term_name}' has low confidence: {context.overall_confidence}")
        else:
            if context.overall_confidence > EXPECTED_NEGATIVE_CONFIDENCE_THRESHOLD:
                self.warnings.append(f"Negative term '{term_name}' has high confidence: {context.overall_confidence}")
                
        # Check evidence consistency
        has_resource_cluster = False
        has_parent_context = False
        
        for evidence in context.evidence:
            if evidence.source == "resource_cluster":
                has_resource_cluster = True
                
                # For positive terms, should have multiple clusters
                if term_name in POSITIVE_TERMS:
                    cluster_count = evidence.metrics.get('cluster_count', 0)
                    if cluster_count < 2:
                        self.warnings.append(f"Positive term '{term_name}' has {cluster_count} clusters (expected ≥2)")
                        
            elif evidence.source == "parent_context":
                has_parent_context = True
                
                # For positive terms, should show divergence
                if term_name in POSITIVE_TERMS:
                    divergent = evidence.payload.get('divergent', False)
                    if not divergent:
                        self.warnings.append(f"Positive term '{term_name}' parent context not divergent")
                        
        # All terms should have resource cluster evidence
        if not has_resource_cluster:
            self.errors.append(f"Term '{term_name}' missing resource_cluster evidence")
            
    def _validate_cross_references(self) -> bool:
        """Validate cross-references between hierarchy and unified context."""
        logger.info("Validating cross-references...")
        
        try:
            with open(HIERARCHY_FILE, 'r') as f:
                hierarchy = json.load(f)
            with open(UNIFIED_CONTEXT_FILE, 'r') as f:
                context_data = json.load(f)
        except Exception as e:
            self.errors.append(f"Error loading files for cross-validation: {e}")
            return False
            
        contexts = context_data.get('contexts', {})
        terms = hierarchy.get('terms', {})
        
        # Check term consistency
        hierarchy_terms = set(terms.keys())
        context_terms = set(contexts.keys())
        
        if hierarchy_terms != context_terms:
            missing_in_context = hierarchy_terms - context_terms
            missing_in_hierarchy = context_terms - hierarchy_terms
            
            if missing_in_context:
                self.errors.append(f"Terms in hierarchy but not in context: {missing_in_context}")
            if missing_in_hierarchy:
                self.errors.append(f"Terms in context but not in hierarchy: {missing_in_hierarchy}")
                
        # Check resource count consistency
        for term_name in hierarchy_terms & context_terms:
            hierarchy_resources = len(terms[term_name].get('resources', []))
            
            # Get resource cluster evidence
            context_term = contexts[term_name]
            resource_evidence = None
            for evidence in context_term.get('evidence', []):
                if evidence.get('source') == 'resource_cluster':
                    resource_evidence = evidence
                    break
                    
            if resource_evidence:
                cluster_labels = resource_evidence.get('payload', {}).get('cluster_labels', [])
                context_resources = len(cluster_labels)
                
                if hierarchy_resources != context_resources:
                    self.warnings.append(f"Resource count mismatch for '{term_name}': hierarchy={hierarchy_resources}, context={context_resources}")
                    
        return len(self.errors) == 0
        
    def _report_results(self):
        """Report validation results."""
        if self.errors:
            logger.error(f"Validation failed with {len(self.errors)} errors:")
            for error in self.errors:
                logger.error(f"  ❌ {error}")
                
        if self.warnings:
            logger.warning(f"Validation completed with {len(self.warnings)} warnings:")
            for warning in self.warnings:
                logger.warning(f"  ⚠️  {warning}")
                
        if not self.errors and not self.warnings:
            logger.info("✅ All validations passed!")
        elif not self.errors:
            logger.info(f"✅ Validation passed with {len(self.warnings)} warnings")


class DatasetBuilder:
    """Builds/rebuilds the test dataset structure."""
    
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root
        
    def build(self) -> bool:
        """Build the complete dataset structure."""
        logger.info("Building test dataset structure...")
        
        # Create directories
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        RAW_RESOURCES_DIR.mkdir(exist_ok=True)
        
        # Create minimal hierarchy if it doesn't exist
        if not HIERARCHY_FILE.exists():
            self._create_minimal_hierarchy()
            
        # Create minimal unified context if it doesn't exist
        if not UNIFIED_CONTEXT_FILE.exists():
            self._create_minimal_unified_context()
            
        logger.info("✅ Dataset structure built successfully")
        return True
        
    def _create_minimal_hierarchy(self):
        """Create a minimal hierarchy.json file."""
        logger.info("Creating minimal hierarchy.json...")
        
        hierarchy = {
            "metadata": {
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "description": "Test dataset hierarchy for sense disambiguation",
                "level": 2
            },
            "terms": {}
        }
        
        # Add minimal term entries
        for term in ALL_TERMS:
            hierarchy["terms"][term] = {
                "level": 2,
                "parents": ["science"] if term in POSITIVE_TERMS else ["science"],
                "resources": []
            }
            
        with open(HIERARCHY_FILE, 'w') as f:
            json.dump(hierarchy, f, indent=2)
            
    def _create_minimal_unified_context(self):
        """Create a minimal unified_context_ground_truth.json file."""
        logger.info("Creating minimal unified context...")
        
        context_data = {
            "metadata": {
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "description": "Minimal ground truth unified context for test dataset",
                "schema_version": "unified_context_v1",
                "total_terms": len(ALL_TERMS),
                "positive_terms": len(POSITIVE_TERMS),
                "negative_terms": len(NEGATIVE_TERMS)
            },
            "contexts": {}
        }
        
        # Add minimal context entries
        for term in ALL_TERMS:
            is_positive = term in POSITIVE_TERMS
            context_data["contexts"][term] = {
                "canonical_name": term,
                "level": 2,
                "overall_confidence": 0.8 if is_positive else 0.2,
                "evidence": [
                    {
                        "source": "resource_cluster",
                        "detector_version": "1.0.0",
                        "confidence": 0.8 if is_positive else 0.2,
                        "metrics": {
                            "cluster_count": 2 if is_positive else 1,
                            "separation_score": 0.7 if is_positive else 0.1
                        },
                        "payload": {
                            "cluster_labels": [0, 1] if is_positive else [0],
                            "cluster_details": {"0": [], "1": []} if is_positive else {"0": []}
                        }
                    }
                ]
            }
            
        with open(UNIFIED_CONTEXT_FILE, 'w') as f:
            json.dump(context_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Build and validate test dataset")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only run validation, don't build")
    parser.add_argument("--rebuild", action="store_true",
                       help="Rebuild dataset structure (creates minimal files)")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT,
                       help="Path to dataset root directory")
    
    args = parser.parse_args()
    
    # Validate if requested
    if not args.rebuild or args.validate_only:
        validator = DatasetValidator(args.dataset_root)
        validation_passed = validator.validate_all()
        
        if args.validate_only:
            return 0 if validation_passed else 1
            
        if not validation_passed and not args.rebuild:
            logger.error("Validation failed. Use --rebuild to recreate minimal structure.")
            return 1
    
    # Build if requested
    if args.rebuild:
        builder = DatasetBuilder(args.dataset_root)
        if not builder.build():
            return 1
            
        # Re-validate after building
        validator = DatasetValidator(args.dataset_root)
        if not validator.validate_all():
            logger.error("Validation failed after building")
            return 1
    
    logger.info("🎉 Test dataset is ready!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 