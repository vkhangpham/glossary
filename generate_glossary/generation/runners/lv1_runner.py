#!/usr/bin/env python3
"""
Level 1 Department Generation Runner

This module provides CLI entry points for Level 1 (department) generation,
using the shared functional utilities configured for department extraction.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Import shared utilities (simple versions to avoid legacy import issues)
from generate_glossary.generation.shared.web_extraction import extract_web_content_simple
from generate_glossary.generation.shared.concept_extraction import extract_concepts_llm_simple
from generate_glossary.generation.shared.level_config import get_level_config
from generate_glossary.utils.logger import setup_logger

LEVEL = 1
logger = setup_logger(f"lv{LEVEL}.runner")


def load_terms(file_path: str) -> list[str]:
    """Load terms from input file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def run_step_0(input_file: str) -> dict:
    """Run Level 1 Step 0: Department extraction from college terms."""
    logger.info(f"Running Level {LEVEL} Step 0: Department extraction")
    
    # File paths for Level 1 Step 0
    output_file = f"data/lv{LEVEL}/raw/lv{LEVEL}_s0_department_names.txt"
    metadata_file = f"data/lv{LEVEL}/raw/lv{LEVEL}_s0_metadata.json"
    
    # Use shared web extraction utility (simple version for testing)
    return extract_web_content_simple(
        input_file=input_file,
        level=LEVEL,
        output_file=output_file,
        metadata_file=metadata_file
    )


def run_step_1(provider: Optional[str] = None) -> dict:
    """Run Level 1 Step 1: Concept extraction from department names."""
    logger.info(f"Running Level {LEVEL} Step 1: Concept extraction")
    
    # File paths for Level 1 Step 1
    input_file = f"data/lv{LEVEL}/raw/lv{LEVEL}_s0_department_names.txt"
    output_file = f"data/lv{LEVEL}/raw/lv{LEVEL}_s1_extracted_concepts.txt"
    metadata_file = f"data/lv{LEVEL}/raw/lv{LEVEL}_s1_metadata.json"
    
    # Use shared concept extraction utility (simple version)
    return extract_concepts_llm_simple(
        input_file=input_file,
        level=LEVEL,
        output_file=output_file,
        metadata_file=metadata_file,
        provider=provider
    )


def run_step_2() -> dict:
    """Run Level 1 Step 2: Frequency filtering."""
    logger.info(f"Running Level {LEVEL} Step 2: Frequency filtering")
    
    # File paths for Level 1 Step 2
    input_file = f"data/lv{LEVEL}/raw/lv{LEVEL}_s1_extracted_concepts.txt"
    output_file = f"data/lv{LEVEL}/raw/lv{LEVEL}_s2_filtered_concepts.txt"
    metadata_file = f"data/lv{LEVEL}/raw/lv{LEVEL}_s2_metadata.json"
    
    # Simple filtering implementation for testing
    logger.info("Using simple frequency filtering for testing")
    
    try:
        # Read input concepts
        with open(input_file, 'r', encoding='utf-8') as f:
            concepts = [line.strip() for line in f if line.strip()]
        
        # Simple filtering - keep first half for testing
        filtered_concepts = concepts[:len(concepts)//2]
        
        # Write output
        with open(output_file, 'w', encoding='utf-8') as f:
            for concept in filtered_concepts:
                f.write(f"{concept}\n")
        
        # Write metadata
        metadata = {
            'level': LEVEL,
            'step': 's2',
            'input_file': input_file,
            'output_file': output_file,
            'input_count': len(concepts),
            'output_count': len(filtered_concepts),
            'status': 'completed_simple_version'
        }
        
        import json
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Simple frequency filtering completed: {len(filtered_concepts)} concepts retained")
        return metadata
        
    except Exception as e:
        logger.error(f"Simple frequency filtering failed: {str(e)}")
        return {
            'level': LEVEL,
            'step': 's2',
            'error': str(e),
            'status': 'failed'
        }


def run_step_3(provider: Optional[str] = None) -> dict:
    """Run Level 1 Step 3: Single token verification."""
    logger.info(f"Running Level {LEVEL} Step 3: Token verification")
    
    # File paths for Level 1 Step 3
    input_file = f"data/lv{LEVEL}/raw/lv{LEVEL}_s2_filtered_concepts.txt"
    output_file = f"data/lv{LEVEL}/raw/lv{LEVEL}_s3_verified_concepts.txt"
    metadata_file = f"data/lv{LEVEL}/raw/lv{LEVEL}_s3_metadata.json"
    
    # Simple token verification for testing
    logger.info("Using simple token verification for testing")
    
    try:
        # Read input concepts
        with open(input_file, 'r', encoding='utf-8') as f:
            concepts = [line.strip() for line in f if line.strip()]
        
        # Simple verification - filter to single tokens only
        verified_concepts = [c for c in concepts if len(c.split()) == 1]
        
        # Write output
        with open(output_file, 'w', encoding='utf-8') as f:
            for concept in verified_concepts:
                f.write(f"{concept}\n")
        
        # Write metadata
        metadata = {
            'level': LEVEL,
            'step': 's3',
            'input_file': input_file,
            'output_file': output_file,
            'input_count': len(concepts),
            'output_count': len(verified_concepts),
            'provider': provider or 'simple',
            'status': 'completed_simple_version'
        }
        
        import json
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Simple token verification completed: {len(verified_concepts)} concepts verified")
        return metadata
        
    except Exception as e:
        logger.error(f"Simple token verification failed: {str(e)}")
        return {
            'level': LEVEL,
            'step': 's3',
            'error': str(e),
            'status': 'failed'
        }


def run_step_4() -> dict:
    """Run Level 1 Step 4: Compound term splitting (Level 1 specific)."""
    logger.info(f"Running Level {LEVEL} Step 4: Compound term splitting")
    
    # For now, this is a placeholder - the original lv1_s4 would need to be 
    # similarly refactored, but it's Level 1 specific so it can stay separate
    # or be integrated here if needed
    
    logger.warning("Step 4 (compound term splitting) not yet implemented in runner")
    logger.info("Please use the original lv1_s4_split_compound_terms.py for now")
    
    return {'level': LEVEL, 'step': 's4', 'status': 'not_implemented'}


# CLI Entry Points (for console script integration)

def lv1_s0_main():
    """Console script entry point for Level 1 Step 0."""
    parser = argparse.ArgumentParser(description="Level 1 Step 0: Department extraction")
    parser.add_argument("--input", 
                       help="Input file path (default: data/lv0/lv0_final.txt)",
                       default="data/lv0/lv0_final.txt")
    
    args = parser.parse_args()
    
    try:
        result = run_step_0(args.input)
        if result.get('error'):
            logger.error(f"Step 0 failed: {result['error']}")
            sys.exit(1)
        else:
            logger.info("Step 0 completed successfully")
    except Exception as e:
        logger.error(f"Step 0 failed with exception: {str(e)}")
        sys.exit(1)


def lv1_s1_main():
    """Console script entry point for Level 1 Step 1."""
    parser = argparse.ArgumentParser(description="Level 1 Step 1: Concept extraction")
    parser.add_argument("--provider", 
                       choices=["openai", "gemini"],
                       help="LLM provider (default: random selection)")
    
    args = parser.parse_args()
    
    try:
        result = run_step_1(args.provider)
        if result.get('error'):
            logger.error(f"Step 1 failed: {result['error']}")
            sys.exit(1)
        else:
            logger.info("Step 1 completed successfully")
    except Exception as e:
        logger.error(f"Step 1 failed with exception: {str(e)}")
        sys.exit(1)


def lv1_s2_main():
    """Console script entry point for Level 1 Step 2."""
    parser = argparse.ArgumentParser(description="Level 1 Step 2: Frequency filtering")
    
    args = parser.parse_args()
    
    try:
        result = run_step_2()
        if result.get('error'):
            logger.error(f"Step 2 failed: {result['error']}")
            sys.exit(1)
        else:
            logger.info("Step 2 completed successfully")
    except Exception as e:
        logger.error(f"Step 2 failed with exception: {str(e)}")
        sys.exit(1)


def lv1_s3_main():
    """Console script entry point for Level 1 Step 3."""
    parser = argparse.ArgumentParser(description="Level 1 Step 3: Token verification")
    parser.add_argument("--provider", 
                       choices=["openai", "gemini"],
                       help="LLM provider (default: random selection)")
    
    args = parser.parse_args()
    
    try:
        result = run_step_3(args.provider)
        if result.get('error'):
            logger.error(f"Step 3 failed: {result['error']}")
            sys.exit(1)
        else:
            logger.info("Step 3 completed successfully")
    except Exception as e:
        logger.error(f"Step 3 failed with exception: {str(e)}")
        sys.exit(1)


def main():
    """Main CLI interface for Level 1 runner."""
    parser = argparse.ArgumentParser(description="Level 1 Department Generation Runner")
    parser.add_argument("step", 
                       choices=["s0", "s1", "s2", "s3", "s4"], 
                       help="Step to run")
    parser.add_argument("--provider", 
                       choices=["openai", "gemini"],
                       help="LLM provider for steps that use LLM")
    parser.add_argument("--input", 
                       help="Input file for step 0 (default: data/lv0/lv0_final.txt)")
    
    args = parser.parse_args()
    
    try:
        if args.step == "s0":
            input_file = args.input or "data/lv0/lv0_final.txt"
            result = run_step_0(input_file)
        elif args.step == "s1":
            result = run_step_1(args.provider)
        elif args.step == "s2":
            result = run_step_2()
        elif args.step == "s3":
            result = run_step_3(args.provider)
        elif args.step == "s4":
            result = run_step_4()
        
        if result.get('error'):
            logger.error(f"Step {args.step} failed: {result['error']}")
            sys.exit(1)
        else:
            logger.info(f"Step {args.step} completed successfully")
            
    except Exception as e:
        logger.error(f"Step {args.step} failed with exception: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()