#!/usr/bin/env python3
"""
Unified CLI command for running generation steps.

Usage:
    uv run generate -l 0 -s 0  # Run Level 0, Step 0
    uv run generate -l 1 -s 2  # Run Level 1, Step 2
"""

import sys
import argparse

from generate_glossary.utils.logger import setup_logger

logger = setup_logger("cli.generate")


def run_level_0_step(step: int, **kwargs) -> int:
    """Run a Level 0 generation step."""

    steps = {
        0: ("lv0_s0_get_college_names", "Get college names from Excel"),
        1: ("lv0_s1_extract_concepts", "Extract concepts via LLM"),
        2: ("lv0_s2_filter_by_institution_freq", "Filter by institution frequency"),
        3: ("lv0_s3_verify_single_token", "Verify single tokens"),
    }

    if step not in steps:
        logger.error(f"Invalid step {step} for Level 0. Valid steps are 0-3.")
        return 1

    module_name, description = steps[step]
    test_mode = kwargs.get("test", False)

    module = __import__(
        f"generate_glossary.generation.lv0.{module_name}", fromlist=["test", "main"]
    )

    if test_mode:
        logger.info(f"Running Level 0, Step {step}: {description} [TEST MODE]")
        func = getattr(module, "test")
    else:
        logger.info(f"Running Level 0, Step {step}: {description}")
        func = getattr(module, "main")

    # Step 3 needs provider argument
    if step == 3:
        provider = kwargs.get("provider")
        func(provider=provider)
    else:
        func()

    return 0


def run_generic_level_step(level: int, step: int, **kwargs) -> int:
    """Run a generic level step using runners."""

    level_configs = {
        1: {
            "module": "lv1_runner",
            "descriptions": {
                0: "Web extraction for departments",
                1: "Extract department concepts",
                2: "Frequency filtering",
                3: "Token verification",
            },
            "default_input": "data/lv0/lv0_final.txt",
        },
        2: {
            "module": "lv2_runner",
            "descriptions": {
                0: "Extract research areas",
                1: "Extract research concepts",
                2: "Frequency filtering",
                3: "Token verification",
            },
            "default_input": "data/lv1/lv1_final.txt",
        },
        3: {
            "module": "lv3_runner",
            "descriptions": {
                0: "Extract conference topics",
                1: "Extract topic concepts",
                2: "Frequency filtering",
                3: "Token verification",
            },
            "default_input": "data/lv2/lv2_final.txt",
        },
    }

    if level not in level_configs:
        logger.error(f"Invalid level {level}")
        return 1

    config = level_configs[level]

    if step not in config["descriptions"]:
        logger.error(f"Invalid step {step} for Level {level}. Valid steps are 0-3.")
        return 1

    module = __import__(
        f"generate_glossary.generation.runners.{config['module']}",
        fromlist=[f"run_step_{step}"],
    )
    run_func = getattr(module, f"run_step_{step}")

    logger.info(f"Running Level {level}, Step {step}: {config['descriptions'][step]}")

    if step == 0:
        input_file = kwargs.get("input_file", config["default_input"])
        run_func(input_file)
    elif step in [1, 3]:
        provider = kwargs.get("provider")
        run_func(provider=provider)
    else:
        run_func()

    return 0


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
        """,
    )

    parser.add_argument(
        "-l",
        "--level",
        type=int,
        required=True,
        choices=[0, 1, 2, 3],
        help="Generation level (0-3)",
    )

    parser.add_argument(
        "-s",
        "--step",
        type=int,
        required=True,
        choices=[0, 1, 2, 3],
        help="Generation step (0-3)",
    )

    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "gemini"],
        help="LLM provider for steps 1 and 3 (default: openai)",
    )

    parser.add_argument(
        "--input-file",
        type=str,
        help="Input file for levels 1-3, step 0 (default: previous level final output)",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with 10%% sample (saves to data/generation/tests/)",
    )

    args = parser.parse_args()

    try:
        kwargs = {
            "provider": args.provider,
            "input_file": args.input_file,
            "test": args.test,
        }

        if args.level == 0:
            return run_level_0_step(args.step, **kwargs)
        elif args.level in [1, 2, 3]:
            return run_generic_level_step(args.level, args.step, **kwargs)
        else:
            logger.error(f"Invalid level: {args.level}. Valid levels are 0-3.")
            return 1

    except Exception as e:
        logger.error(f"Error running Level {args.level}, Step {args.step}: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
