"""
Main optimizer interface for prompt optimization using GEPA.
Provides high-level functions for optimizing prompts.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import time

import gepa

from ..registry import get_prompt, register_prompt
from .concept_extraction_adapter import ConceptExtractionAdapter
from generate_glossary.utils.logger import setup_logger

logger = setup_logger("prompt_optimizer")


def optimize_prompt(
    prompt_key: str,
    training_data: List[Dict[str, Any]],
    validation_data: Optional[List[Dict[str, Any]]] = None,
    level: int = 0,
    task_model: str = "openai/gpt-4o-mini",
    reflection_model: str = "openai/gpt-4",
    max_metric_calls: int = 100,
    save_results: bool = True,
    output_dir: Optional[Path] = None,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Optimize a prompt using GEPA.

    Args:
        prompt_key: Key of prompt to optimize (e.g., "extraction.level0")
        training_data: Training examples with inputs and optional ground truth
        validation_data: Validation set (uses training if not provided)
        level: Extraction level (0=colleges, 1=departments, etc.)
        task_model: Model being optimized
        reflection_model: Model for reflection/improvement
        max_metric_calls: Budget for optimization
        save_results: Whether to save optimized prompt to registry
        output_dir: Directory for saving results

    Returns:
        Tuple of (optimized_prompts, metrics)
    """
    logger.info(f"Starting optimization for {prompt_key}")

    try:
        system_prompt = get_prompt(f"{prompt_key}_system")
        user_template = get_prompt(f"{prompt_key}_user_template")
    except ValueError:
        logger.warning(f"Could not load {prompt_key}, using defaults")
        system_prompt = "Extract concepts from the input text."
        user_template = "Process: {sources}"

    seed_candidate = {"system_prompt": system_prompt, "user_template": user_template}

    if validation_data is None:
        split_idx = int(len(training_data) * 0.8)
        train_set = training_data[:split_idx]
        val_set = training_data[split_idx:]
    else:
        train_set = training_data
        val_set = validation_data

    logger.info(f"Training set: {len(train_set)} examples")
    logger.info(f"Validation set: {len(val_set)} examples")

    adapter = ConceptExtractionAdapter(level=level, task_model=task_model)

    start_time = time.time()

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=train_set,
        valset=val_set,
        adapter=adapter,
        task_lm=task_model,
        reflection_lm=reflection_model,
        max_metric_calls=max_metric_calls,
        candidate_selection_strategy="pareto",
        reflection_minibatch_size=3,
        display_progress_bar=True,
    )

    elapsed_time = time.time() - start_time

    # Extract best candidate and metrics
    best_candidate = result.best_candidate
    metrics = {
        "optimization_time": elapsed_time,
        "num_iterations": (
            len(result.iteration_scores) if hasattr(result, "iteration_scores") else 0
        ),
        "initial_score": result.seed_score if hasattr(result, "seed_score") else 0,
        "final_score": result.best_score if hasattr(result, "best_score") else 0,
        "improvement": (
            (result.best_score - result.seed_score)
            if hasattr(result, "best_score") and hasattr(result, "seed_score")
            else 0
        ),
    }

    logger.info(f"Optimization complete in {elapsed_time:.2f}s")
    logger.info(
        f"Score improved from {metrics['initial_score']:.3f} to {metrics['final_score']:.3f}"
    )

    # Save optimized prompts to registry
    if save_results:
        # Register system prompt
        if "system_prompt" in best_candidate:
            version = register_prompt(
                f"{prompt_key}_system_optimized",
                best_candidate["system_prompt"],
                metadata={
                    "optimizer": "GEPA",
                    "score": metrics["final_score"],
                    "improvement": metrics["improvement"],
                    "task_model": task_model,
                    "reflection_model": reflection_model,
                },
            )
            logger.info(f"Registered optimized system prompt: {version[:8]}")

        # Register user template
        if "user_template" in best_candidate:
            version = register_prompt(
                f"{prompt_key}_user_template_optimized",
                best_candidate["user_template"],
                metadata={
                    "optimizer": "GEPA",
                    "score": metrics["final_score"],
                    "improvement": metrics["improvement"],
                    "task_model": task_model,
                    "reflection_model": reflection_model,
                },
            )
            logger.info(f"Registered optimized user template: {version[:8]}")

    # Save detailed results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save prompts
        prompts_file = output_dir / f"{prompt_key}_optimized.json"
        with open(prompts_file, "w") as f:
            json.dump(
                {
                    "original": seed_candidate,
                    "optimized": best_candidate,
                    "metrics": metrics,
                },
                f,
                indent=2,
            )

        logger.info(f"Saved results to {prompts_file}")

    return best_candidate, metrics


def batch_optimize_prompts(
    prompt_keys: List[str],
    dataset_path: Path,
    level: int = 0,
    max_metric_calls: int = 100,
    output_dir: Optional[Path] = None,
) -> Dict[str, Tuple[Dict[str, str], Dict[str, Any]]]:
    """
    Optimize multiple prompts in batch.

    Args:
        prompt_keys: List of prompt keys to optimize
        dataset_path: Path to dataset file
        level: Extraction level
        max_metric_calls: Budget per prompt
        output_dir: Output directory

    Returns:
        Dictionary mapping prompt keys to (optimized_prompts, metrics)
    """
    # Load dataset
    with open(dataset_path, "r") as f:
        if dataset_path.suffix == ".json":
            dataset = json.load(f)
        else:
            # Assume line-delimited text
            dataset = [{"input": line.strip()} for line in f if line.strip()]

    results = {}

    for prompt_key in prompt_keys:
        logger.info(f"Optimizing {prompt_key}...")

        try:
            optimized, metrics = optimize_prompt(
                prompt_key=prompt_key,
                training_data=dataset,
                level=level,
                max_metric_calls=max_metric_calls,
                output_dir=output_dir,
            )

            results[prompt_key] = (optimized, metrics)

        except Exception as e:
            logger.error(f"Failed to optimize {prompt_key}: {e}")
            results[prompt_key] = (None, {"error": str(e)})

    return results


def compare_prompts(
    original_key: str,
    optimized_key: str,
    test_data: List[Dict[str, Any]],
    level: int = 0,
) -> Dict[str, Any]:
    """
    Compare original and optimized prompts on test data.

    Args:
        original_key: Original prompt key
        optimized_key: Optimized prompt key
        test_data: Test dataset
        level: Extraction level

    Returns:
        Comparison metrics
    """
    adapter = ConceptExtractionAdapter(level=level)

    # Load prompts
    original_system = get_prompt(f"{original_key}_system")
    original_user = get_prompt(f"{original_key}_user_template")

    optimized_system = get_prompt(f"{optimized_key}_system")
    optimized_user = get_prompt(f"{optimized_key}_user_template")

    # Evaluate original
    original_candidate = {
        "system_prompt": original_system,
        "user_template": original_user,
    }
    original_eval = adapter.evaluate(
        test_data, original_candidate, capture_traces=False
    )
    original_score = sum(original_eval.scores) / len(original_eval.scores)

    # Evaluate optimized
    optimized_candidate = {
        "system_prompt": optimized_system,
        "user_template": optimized_user,
    }
    optimized_eval = adapter.evaluate(
        test_data, optimized_candidate, capture_traces=False
    )
    optimized_score = sum(optimized_eval.scores) / len(optimized_eval.scores)

    return {
        "original_score": original_score,
        "optimized_score": optimized_score,
        "improvement": optimized_score - original_score,
        "improvement_percent": (
            ((optimized_score - original_score) / original_score * 100)
            if original_score > 0
            else 0
        ),
        "test_size": len(test_data),
    }
