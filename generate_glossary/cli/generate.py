#!/usr/bin/env python3
"""
Unified CLI command for running generation steps with dynamic step discovery.

This modernized CLI eliminates hardcoded step mappings and uses dynamic
discovery to automatically adapt to changes in the step modules.

Usage:
    uv run generate -l 0 -s 0  # Run Level 0, Step 0
    uv run generate -l 1 -s 2  # Run Level 1, Step 2
    uv run generate --list      # List all available steps
    uv run generate --validate -l 1  # Validate all Level 1 steps
"""

import sys
import argparse

from generate_glossary.utils.logger import get_logger
from generate_glossary.generation.wrapper_utils import (
    create_generic_step_runner, 
    validate_step_dependencies,
    get_step_parameters
)
from generate_glossary.generation.step_discovery import (
    discover_available_steps,
    get_step_metadata, 
    validate_all_steps,
    get_validation_summary,
    list_all_steps
)

logger = get_logger("cli.generate")


def run_step(level: int, step: int, **kwargs) -> int:
    """
    Run any generation step using the dynamic step runner.
    
    This unified function replaces both run_level_0_step and run_generic_level_step
    with a single implementation that works for all levels and steps.
    """
    try:
        # Get step metadata
        step_metadata = get_step_metadata(level, step)
        if 'error' in step_metadata:
            logger.error(f"Cannot run Level {level} Step {step}: {step_metadata['error']}")
            return 1
        
        # Log what we're running
        description = step_metadata['description']
        test_mode = kwargs.get('test', False)
        mode_text = " [TEST MODE]" if test_mode else ""
        logger.info(f"Running Level {level}, Step {step}: {description}{mode_text}")
        
        # Create and run the step
        step_runner = create_generic_step_runner(level, step)
        return step_runner(**kwargs)
        
    except Exception as e:
        logger.error(f"Failed to run Level {level} Step {step}: {str(e)}")
        logger.exception("Full traceback:")
        return 1


def handle_list_command() -> int:
    """Handle the --list command to show all available steps."""
    try:
        all_steps = list_all_steps()
        
        print("Available Generation Steps:")
        print("=" * 50)
        
        for level in sorted(all_steps.keys()):
            level_names = {
                0: "College/School",
                1: "Department", 
                2: "Research Area",
                3: "Conference Topic"
            }
            
            print(f"\nLevel {level}: {level_names.get(level, f'Level {level}')} Generation")
            print("-" * 30)
            
            for step in sorted(all_steps[level].keys()):
                description = all_steps[level][step]
                print(f"  Step {step}: {description}")
        
        print("\nUsage Examples:")
        print("  uv run generate -l 0 -s 0    # Run Level 0, Step 0")
        print("  uv run generate -l 1 -s 2 --test    # Run Level 1, Step 2 in test mode")
        print("  uv run generate --validate -l 1     # Validate Level 1 steps")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to list steps: {str(e)}")
        return 1


def handle_validate_command(level: int = None) -> int:
    """Handle the --validate command to validate steps."""
    try:
        if level is not None:
            # Validate specific level
            validation_report = validate_all_steps(level)
            
            print(f"Validation Report for Level {level}")
            print("=" * 40)
            print(f"Overall Status: {'✅ READY' if validation_report['overall_valid'] else '❌ NOT READY'}")
            print(f"Valid Steps: {validation_report['summary']['valid_steps']}/{validation_report['summary']['total_steps']}")
            
            if validation_report['summary']['steps_with_warnings'] > 0:
                print(f"Steps with Warnings: {validation_report['summary']['steps_with_warnings']}")
            
            print("\nStep Details:")
            print("-" * 20)
            
            for step_num in sorted(validation_report['steps'].keys()):
                summary = get_validation_summary(level, step_num)
                print(summary)
                print()
            
            # Show recommendations
            if validation_report.get('recommendations'):
                print("Recommendations:")
                print("-" * 15)
                for rec in validation_report['recommendations']:
                    print(f"  • {rec}")
                print()
        
        else:
            # Validate all levels
            print("Validation Report for All Levels")
            print("=" * 35)
            
            overall_valid = True
            for level_num in range(4):
                validation_report = validate_all_steps(level_num)
                status = "✅" if validation_report['overall_valid'] else "❌"
                valid_count = validation_report['summary']['valid_steps']
                total_count = validation_report['summary']['total_steps']
                
                print(f"Level {level_num}: {status} {valid_count}/{total_count} steps ready")
                
                if not validation_report['overall_valid']:
                    overall_valid = False
            
            print(f"\nOverall Status: {'✅ ALL READY' if overall_valid else '❌ ISSUES FOUND'}")
            print("Use --validate -l <level> for detailed validation of specific levels")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to validate steps: {str(e)}")
        return 1


def main():
    """Main entry point for the modernized generate command with dynamic discovery."""
    parser = argparse.ArgumentParser(
        description="Run glossary generation steps with dynamic discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run generate -l 0 -s 0                    # Run Level 0, Step 0
  uv run generate -l 1 -s 2 --test             # Run Level 1, Step 2 in test mode
  uv run generate -l 2 -s 1 --provider gemini  # Use Gemini for LLM steps
  uv run generate --list                       # List all available steps
  uv run generate --validate                   # Validate all levels
  uv run generate --validate -l 1              # Validate only Level 1
  
Levels:
  0: College/School extraction (from Excel)
  1: Department extraction (from Level 0)
  2: Research area extraction (from Level 1)
  3: Conference topic extraction (from Level 2)
  
Steps (dynamically discovered):
  0: Data extraction (web mining or Excel parsing)
  1: LLM concept extraction
  2: Frequency-based filtering
  3: Single-token verification
        """,
    )

    # Special commands group
    special_group = parser.add_mutually_exclusive_group()
    special_group.add_argument(
        "--list",
        action="store_true",
        help="List all available generation steps"
    )
    special_group.add_argument(
        "--validate",
        action="store_true",
        help="Validate step dependencies and requirements"
    )

    # Standard execution arguments
    parser.add_argument(
        "-l",
        "--level",
        type=int,
        choices=[0, 1, 2, 3],
        help="Generation level (0-3). Required unless using --list"
    )

    parser.add_argument(
        "-s",
        "--step",
        type=int,
        choices=[0, 1, 2, 3],
        help="Generation step (0-3). Required unless using special commands"
    )

    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "gemini"],
        help="LLM provider for steps 1 and 3 (default: openai)"
    )

    parser.add_argument(
        "--input-file",
        type=str,
        help="Input file for step 0 (default: auto-determined from level config)"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (uses test data paths)"
    )

    args = parser.parse_args()

    try:
        # Handle special commands
        if args.list:
            return handle_list_command()
        
        if args.validate:
            return handle_validate_command(args.level)
        
        # Validate required arguments for normal execution
        if args.level is None:
            parser.error("--level is required unless using --list or --validate")
        
        if args.step is None:
            parser.error("--step is required unless using special commands")
        
        # Prepare kwargs for step execution
        kwargs = {
            "provider": args.provider or "openai",
            "test": args.test
        }
        
        # Handle input file for step 0
        if args.step == 0 and args.input_file:
            kwargs["input_file"] = args.input_file
        
        # Run the step using the unified runner
        return run_step(args.level, args.step, **kwargs)

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
