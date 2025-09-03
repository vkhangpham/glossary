#!/usr/bin/env python3
"""
Unified CLI command for running generation steps.

Usage:
    uv run generate -l 0 -s 0  # Run Level 0, Step 0
    uv run generate -l 1 -s 2  # Run Level 1, Step 2
"""

import sys
import argparse
from typing import Optional
from pathlib import Path

from generate_glossary.utils.logger import setup_logger

logger = setup_logger("cli.generate")


def run_level_0_step(step: int, **kwargs) -> int:
    """Run a Level 0 generation step."""
    if step == 0:
        from generate_glossary.generation.lv0.lv0_s0_get_college_names import main
        logger.info("Running Level 0, Step 0: Get college names from Excel")
        main()  # Uses default Excel path
        return 0
        
    elif step == 1:
        from generate_glossary.generation.lv0.lv0_s1_extract_concepts import main
        logger.info("Running Level 0, Step 1: Extract concepts via LLM")
        main()
        return 0
        
    elif step == 2:
        from generate_glossary.generation.lv0.lv0_s2_filter_by_institution_freq import main
        logger.info("Running Level 0, Step 2: Filter by institution frequency")
        main()
        return 0
        
    elif step == 3:
        from generate_glossary.generation.lv0.lv0_s3_verify_single_token import main
        logger.info("Running Level 0, Step 3: Verify single tokens")
        main()
        return 0
        
    else:
        logger.error(f"Invalid step {step} for Level 0. Valid steps are 0-3.")
        return 1


def run_level_1_step(step: int, **kwargs) -> int:
    """Run a Level 1 generation step."""
    from generate_glossary.generation.runners.lv1_runner import (
        run_step_0, run_step_1, run_step_2, run_step_3
    )
    
    # Get input file for step 0 if provided
    input_file = kwargs.get('input_file', 'data/lv0/lv0_final.txt')
    provider = kwargs.get('provider')
    
    if step == 0:
        logger.info("Running Level 1, Step 0: Web extraction for departments")
        run_step_0(input_file)
        return 0
        
    elif step == 1:
        logger.info("Running Level 1, Step 1: Extract department concepts")
        run_step_1(provider=provider)
        return 0
        
    elif step == 2:
        logger.info("Running Level 1, Step 2: Frequency filtering")
        run_step_2()
        return 0
        
    elif step == 3:
        logger.info("Running Level 1, Step 3: Token verification")
        run_step_3(provider=provider)
        return 0
        
    else:
        logger.error(f"Invalid step {step} for Level 1. Valid steps are 0-3.")
        return 1


def run_level_2_step(step: int, **kwargs) -> int:
    """Run a Level 2 generation step."""
    from generate_glossary.generation.runners.lv2_runner import (
        run_step_0, run_step_1, run_step_2, run_step_3
    )
    
    input_file = kwargs.get('input_file', 'data/lv1/lv1_final.txt')
    provider = kwargs.get('provider')
    
    if step == 0:
        logger.info("Running Level 2, Step 0: Extract research areas")
        run_step_0(input_file)
        return 0
        
    elif step == 1:
        logger.info("Running Level 2, Step 1: Extract research concepts")
        run_step_1(provider=provider)
        return 0
        
    elif step == 2:
        logger.info("Running Level 2, Step 2: Frequency filtering")
        run_step_2()
        return 0
        
    elif step == 3:
        logger.info("Running Level 2, Step 3: Token verification")
        run_step_3(provider=provider)
        return 0
        
    else:
        logger.error(f"Invalid step {step} for Level 2. Valid steps are 0-3.")
        return 1


def run_level_3_step(step: int, **kwargs) -> int:
    """Run a Level 3 generation step."""
    from generate_glossary.generation.runners.lv3_runner import (
        run_step_0, run_step_1, run_step_2, run_step_3
    )
    
    input_file = kwargs.get('input_file', 'data/lv2/lv2_final.txt')
    provider = kwargs.get('provider')
    
    if step == 0:
        logger.info("Running Level 3, Step 0: Extract conference topics")
        run_step_0(input_file)
        return 0
        
    elif step == 1:
        logger.info("Running Level 3, Step 1: Extract topic concepts")
        run_step_1(provider=provider)
        return 0
        
    elif step == 2:
        logger.info("Running Level 3, Step 2: Frequency filtering")
        run_step_2()
        return 0
        
    elif step == 3:
        logger.info("Running Level 3, Step 3: Token verification")
        run_step_3(provider=provider)
        return 0
        
    else:
        logger.error(f"Invalid step {step} for Level 3. Valid steps are 0-3.")
        return 1


def main():
    """Main entry point for the generate command."""
    parser = argparse.ArgumentParser(
        description="Run glossary generation steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run generate -l 0 -s 0    # Run Level 0, Step 0
  uv run generate -l 1 -s 2    # Run Level 1, Step 2
  uv run generate -l 2 -s 1 --provider gemini  # Use Gemini for LLM
  
Levels:
  0: College/School extraction (from Excel)
  1: Department extraction (from Level 0)
  2: Research area extraction (from Level 1)
  3: Conference topic extraction (from Level 2)
  
Steps (same for all levels):
  0: Data extraction (web mining or Excel parsing)
  1: LLM concept extraction
  2: Frequency-based filtering
  3: Single-token verification
        """
    )
    
    parser.add_argument(
        '-l', '--level',
        type=int,
        required=True,
        choices=[0, 1, 2, 3],
        help='Generation level (0-3)'
    )
    
    parser.add_argument(
        '-s', '--step',
        type=int,
        required=True,
        choices=[0, 1, 2, 3],
        help='Generation step (0-3)'
    )
    
    parser.add_argument(
        '--provider',
        type=str,
        choices=['openai', 'gemini'],
        help='LLM provider for steps 1 and 3 (default: openai)'
    )
    
    parser.add_argument(
        '--input-file',
        type=str,
        help='Input file for levels 1-3, step 0 (default: previous level final output)'
    )
    
    args = parser.parse_args()
    
    # Dispatch to appropriate level handler
    level_handlers = {
        0: run_level_0_step,
        1: run_level_1_step,
        2: run_level_2_step,
        3: run_level_3_step
    }
    
    handler = level_handlers.get(args.level)
    if not handler:
        logger.error(f"Invalid level: {args.level}")
        return 1
    
    # Run the step
    try:
        kwargs = {
            'provider': args.provider,
            'input_file': args.input_file
        }
        return handler(args.step, **kwargs)
        
    except Exception as e:
        logger.error(f"Error running Level {args.level}, Step {args.step}: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())