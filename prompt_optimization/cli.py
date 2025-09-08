#!/usr/bin/env python3
"""CLI for running prompt optimization with GEPA

Usage:
    uv run optimize-prompt --prompt lv0_s1 --auto medium --max-calls 100
"""

import click
import os
import importlib
from pathlib import Path
from typing import Optional

import dspy


@click.command()
@click.option(
    "--prompt", required=True, help="Prompt to optimize (e.g., lv0_s1, lv1_s2)"
)
@click.option(
    "--auto",
    default="medium",
    type=click.Choice(["light", "medium", "heavy"]),
    help="Optimization level: light (~5min), medium (~15min), heavy (~30min+)",
)
@click.option("--max-calls", type=int, help="Maximum LLM calls (budget limit)")
@click.option("--num-threads", default=8, type=int, help="Number of parallel threads")
@click.option(
    "--task-model", default="openai/gpt-5-nano", help="Model for task execution"
)
@click.option(
    "--reflection-model",
    default="openai/gpt-5",
    help="Model for reflection/optimization",
)
@click.option(
    "--train-split", default=0.7, type=float, help="Train/validation split ratio"
)
@click.option(
    "--batch-size", default=5, type=int, help="Training batch size (production uses 20)"
)
@click.option("--verbose", is_flag=True, help="Enable verbose output")
def optimize_prompt(
    prompt: str,
    auto: str,
    max_calls: Optional[int],
    num_threads: int,
    task_model: str,
    reflection_model: str,
    train_split: float,
    batch_size: int,
    verbose: bool,
):
    """Run GEPA optimization for a specific prompt.

    Examples:
        uv run optimize-prompt --prompt lv0_s1 --auto light
        uv run optimize-prompt --prompt lv1_s2 --auto heavy --max-calls 500
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        click.echo("Error: Please set OPENAI_API_KEY environment variable", err=True)
        return 1

    try:
        module_name = f"prompt_optimization.optimizers.{prompt}"
        optimizer_module = importlib.import_module(module_name)
    except ImportError:
        click.echo(f"Error: Optimizer not found for '{prompt}'", err=True)
        click.echo(
            f"Expected file: prompt_optimization/optimizers/{prompt}.py", err=True
        )

        optimizers_dir = Path("prompt_optimization/optimizers")
        available = [
            f.stem
            for f in optimizers_dir.glob("*.py")
            if f.stem not in ["__init__", "template"]
        ]
        if available:
            click.echo(f"\nAvailable optimizers: {', '.join(available)}")
        return 1

    required_funcs = [
        "create_training_data",
        "prepare_dspy_examples",
        "metric_with_feedback",
        "get_program",
    ]
    missing = [f for f in required_funcs if not hasattr(optimizer_module, f)]
    if missing:
        click.echo(
            f"Error: Optimizer {prompt} is missing required functions: {missing}",
            err=True,
        )
        click.echo(
            "Please ensure the optimizer follows the template structure", err=True
        )
        return 1

    click.echo(f"{'='*60}")
    click.echo(f"Prompt Optimization: {prompt}")
    click.echo(f"{'='*60}")
    click.echo("Configuration:")
    click.echo(f"  - Optimization level: {auto}")
    click.echo(f"  - Task model: {task_model}")
    click.echo(f"  - Reflection model: {reflection_model}")
    click.echo(f"  - Threads: {num_threads}")
    if max_calls:
        click.echo(f"  - Max LLM calls: {max_calls}")
    click.echo(f"{'='*60}\n")

    if "gpt-5" in task_model:
        task_lm = dspy.LM(
            task_model, api_key=api_key, temperature=1.0, max_tokens=16000
        )
    else:
        task_lm = dspy.LM(task_model, api_key=api_key, temperature=1.0)

    if "gpt-5" in reflection_model:
        reflection_lm = dspy.LM(
            reflection_model, api_key=api_key, temperature=1.0, max_tokens=16000
        )
    else:
        reflection_lm = dspy.LM(reflection_model, api_key=api_key, temperature=1.0)
    dspy.configure_cache(
        enable_disk_cache=False,  # Disable to avoid SQLite errors
        enable_memory_cache=True,  # Keep memory cache for performance
    )
    dspy.configure(lm=task_lm)

    click.echo("Loading training data...")
    import inspect

    sig = inspect.signature(optimizer_module.create_training_data)
    if "batch_size" in sig.parameters:
        training_data = optimizer_module.create_training_data(batch_size=batch_size)
        click.echo(f"  Using batch size: {batch_size} (production uses 20)")
    else:
        training_data = optimizer_module.create_training_data()

    dspy_examples = optimizer_module.prepare_dspy_examples(training_data)

    split_idx = int(len(dspy_examples) * train_split)
    trainset = dspy_examples[:split_idx]
    valset = dspy_examples[split_idx:]

    click.echo(
        f"Using {len(trainset)} training examples and {len(valset)} validation examples\n"
    )

    program = optimizer_module.get_program()

    click.echo("Starting GEPA optimization...")
    click.echo("This may take several minutes...\n")

    optimizer_kwargs = {
        "metric": optimizer_module.metric_with_feedback,
        "num_threads": num_threads,
        "track_stats": True,
        "reflection_lm": reflection_lm,
        "use_merge": True,
        "add_format_failure_as_feedback": True,
        "reflection_minibatch_size": 3,
    }

    if max_calls:
        optimizer_kwargs["max_metric_calls"] = max_calls
    else:
        optimizer_kwargs["auto"] = auto

    optimizer = dspy.GEPA(**optimizer_kwargs)

    optimized_program = optimizer.compile(
        student=program, trainset=trainset, valset=valset
    )

    click.echo("\n" + "=" * 60)
    click.echo("OPTIMIZATION COMPLETE")
    click.echo("=" * 60)

    if hasattr(optimizer_module, "save_optimized_prompts"):
        saved_paths = optimizer_module.save_optimized_prompts(
            optimized_program, trainset, valset, task_model, reflection_model, auto
        )
        click.echo("\nSaved prompts:")
        for path in saved_paths:
            click.echo(f"  - {path}")
    else:
        click.echo("\nWarning: No save_optimized_prompts function found")
        click.echo("Optimized program not saved")

    if hasattr(optimized_program, "detailed_results"):
        stats = optimized_program.detailed_results
        click.echo("\nOptimization Statistics:")
        if hasattr(stats, "get"):
            click.echo(f"  - Best score: {stats.get('best_score', 'N/A')}")
            click.echo(
                f"  - Candidates evaluated: {stats.get('candidates_evaluated', 'N/A')}"
            )

    click.echo(f"\n{'='*60}")
    click.echo("Optimization complete!")

    return 0


if __name__ == "__main__":
    exit(optimize_prompt())

