#!/usr/bin/env python
"""CLI for prompt optimization using DSPy GEPA."""

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def get_available_optimizers() -> list[str]:
    """Get list of available optimizer modules."""
    optimizers_dir = Path(__file__).parent / "optimizers"
    optimizers = []

    for file in optimizers_dir.glob("*.py"):
        if file.stem not in ["__init__", "common", "template"]:
            optimizers.append(file.stem)

    return sorted(optimizers)


def run_optimizer(
    name: str,
    verbose: bool = False,
    auto: str = "light",
    max_full_evals: Optional[int] = None,
    max_metric_calls: Optional[int] = None,
) -> None:
    """Run a specific optimizer by name.

    Args:
        name: Name of the optimizer (e.g., 'lv0_s1', 'lv0_s3')
        verbose: Whether to show verbose output
        auto: GEPA optimization level ('light', 'medium', 'heavy')
        max_full_evals: Maximum number of full evaluations (overrides auto)
        max_metric_calls: Maximum number of metric calls (overrides auto)
    """
    available = get_available_optimizers()
    if name not in available:
        print(f"Error: Optimizer '{name}' not found.")
        print(f"Available optimizers: {', '.join(available)}")
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it with: export OPENAI_API_KEY='your-key'")
        sys.exit(1)

    try:
        module_name = f"prompt_optimization.optimizers.{name}"
        if verbose:
            print(f"Loading optimizer module: {module_name}")

        module = importlib.import_module(module_name)

        if not hasattr(module, "optimize_prompts"):
            print(
                f"Error: Optimizer '{name}' does not have an optimize_prompts function."
            )
            sys.exit(1)

        if max_metric_calls is not None:
            os.environ["GEPA_MAX_METRIC_CALLS"] = str(max_metric_calls)
            if verbose:
                print(f"Using max_metric_calls: {max_metric_calls}")
        elif max_full_evals is not None:
            os.environ["GEPA_MAX_FULL_EVALS"] = str(max_full_evals)
            if verbose:
                print(f"Using max_full_evals: {max_full_evals}")
        else:
            os.environ["GEPA_AUTO"] = auto
            if verbose:
                print(f"Using auto level: {auto}")

        print(f"\n{'=' * 60}")
        print(f"Running prompt optimization for: {name}")
        if max_metric_calls is not None:
            print(f"Budget: {max_metric_calls} metric calls")
        elif max_full_evals is not None:
            print(f"Budget: {max_full_evals} full evaluations")
        else:
            print(f"Optimization level: {auto}")
        print(f"{'=' * 60}\n")

        result = module.optimize_prompts()

        print(f"\n{'=' * 60}")
        print(f"✅ Optimization complete for {name}!")
        print(f"{'=' * 60}")

        return result

    except ImportError as e:
        print(f"Error importing optimizer '{name}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error running optimizer '{name}': {e}")
        import traceback

        if verbose:
            traceback.print_exc()
        sys.exit(1)


def main():
    """Main CLI entry point."""
    # Properly handle asyncio event loop cleanup
    import asyncio
    import atexit
    
    def cleanup_event_loop():
        """Clean up any existing event loops to prevent warnings."""
        try:
            loop = asyncio.get_event_loop()
            if loop and not loop.is_closed():
                loop.close()
        except RuntimeError:
            pass  # No event loop exists
    
    # Register cleanup function
    atexit.register(cleanup_event_loop)
    
    parser = argparse.ArgumentParser(
        description="Optimize prompts using DSPy GEPA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize lv0_s1 prompts (light/fast mode)
  uv run optimize-prompt --name lv0_s1 --model gpt-5-nano --reflection-model gpt-5
  
  # Optimize lv0_s3 prompts with thorough optimization
  uv run optimize-prompt --name lv0_s3 --model gpt-5-nano --reflection-model gpt-5 --auto heavy
  
  # Optimize with exactly 10 full dataset evaluations
  uv run optimize-prompt --name lv0_s1 --model gpt-4o-mini --reflection-model gpt-4o --max-full-evals 10
  
  # Optimize with exactly 500 metric calls
  uv run optimize-prompt --name lv0_s1 --model gpt-5-nano --reflection-model gpt-5 --max-metric-calls 500
  
  # Optimize with verbose output and medium optimization
  uv run optimize-prompt --name lv0_s1 --model gpt-5-nano --reflection-model gpt-5 --verbose --auto medium
  
  # List available optimizers
  uv run optimize-prompt --list

Environment Variables:
  OPENAI_API_KEY - Required: Your OpenAI API key
        """,
    )

    parser.add_argument(
        "--name",
        "-n",
        type=str,
        help="Name of the prompt optimizer to run (e.g., lv0_s1, lv0_s3)",
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available optimizers",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Generation model (e.g., gpt-5-nano, gpt-4o-mini)",
    )

    parser.add_argument(
        "--reflection-model",
        "-r",
        type=str,
        required=True,
        help="Reflection model (e.g., gpt-5, gpt-4o)",
    )

    parser.add_argument(
        "--auto",
        "-a",
        type=str,
        choices=["light", "medium", "heavy"],
        default="light",
        help="GEPA optimization level: light (fast), medium (balanced), heavy (thorough) [default: light]",
    )

    parser.add_argument(
        "--max-full-evals",
        type=int,
        help="Maximum number of full evaluations over the dataset (overrides --auto)",
    )

    parser.add_argument(
        "--max-metric-calls",
        type=int,
        help="Maximum number of individual metric calls (overrides --auto and --max-full-evals)",
    )

    args = parser.parse_args()

    if args.list:
        available = get_available_optimizers()
        print("Available optimizers:")
        for optimizer in available:
            print(f"  - {optimizer}")
        return

    if not args.name:
        parser.print_help()
        sys.exit(1)

    # Set required model environment variables from CLI args
    os.environ["GEPA_GEN_MODEL"] = args.model
    os.environ["GEPA_REFLECTION_MODEL"] = args.reflection_model
    
    if args.verbose:
        print(f"Using generation model: {args.model}")
        print(f"Using reflection model: {args.reflection_model}")

    if args.max_metric_calls and args.max_full_evals:
        print("Error: Cannot specify both --max-metric-calls and --max-full-evals")
        print(
            "Choose one budget control method: --auto, --max-full-evals, or --max-metric-calls"
        )
        sys.exit(1)

    run_optimizer(
        args.name,
        verbose=args.verbose,
        auto=args.auto,
        max_full_evals=args.max_full_evals,
        max_metric_calls=args.max_metric_calls,
    )


if __name__ == "__main__":
    main()
