"""
Common helper functions for prompt optimizers.

Models are provided via CLI arguments and passed through environment variables internally.
"""

import json
import os
from typing import List, Dict, Any, Tuple, Optional
import dspy


def load_json_training(path: str) -> List[Dict[str, Any]]:
    """
    Load training examples from a JSON file.

    Args:
        path: Path to the JSON file

    Returns:
        List of training examples

    Raises:
        ValueError: If file doesn't exist or no examples found
    """
    if not os.path.exists(path):
        raise ValueError(f"Training data file not found: {path}")

    with open(path, "r") as f:
        examples = json.load(f)

    if len(examples) == 0:
        raise ValueError(f"No training examples found at {path}")

    return examples


def split_train_val(
    examples: List[Any], ratio: float = 0.8
) -> Tuple[List[Any], List[Any]]:
    """
    Split examples into training and validation sets with random shuffling.

    Args:
        examples: List of examples to split (can be dspy.Example or dict)
        ratio: Fraction for training set (default: 0.8)

    Returns:
        Tuple of (trainset, valset)
    """
    import random

    if len(examples) == 0:
        raise ValueError("Cannot split empty dataset")

    if len(examples) == 1:

        return examples, examples

    shuffled = examples.copy()
    random.seed(42)
    random.shuffle(shuffled)

    train_size = max(1, int(ratio * len(shuffled)))
    trainset = shuffled[:train_size]
    valset = shuffled[train_size:]

    return trainset, valset


def configure_openai_lms(gen_model: str, ref_model: str, api_key: Optional[str] = None) -> Tuple[Any, Any]:
    """
    Configure OpenAI language models for generation and reflection.

    Args:
        gen_model: Generation model name (e.g., "gpt-5-nano", "gpt-4o-mini")
        ref_model: Reflection model name (e.g., "gpt-5", "gpt-4o")
        api_key: OpenAI API key (if not provided, uses OPENAI_API_KEY env var)

    Returns:
        Tuple of (generation_lm, reflection_lm)
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not provided and OPENAI_API_KEY env var not set"
            )

    os.environ["OPENAI_API_KEY"] = api_key
    
    # Set up DSPy cache directory to avoid database errors
    import tempfile
    cache_dir = tempfile.mkdtemp(prefix="dspy_cache_")
    os.environ["DSP_CACHEDIR"] = cache_dir
    # Also try to disable cache
    os.environ["DSP_DISABLE_CACHE"] = "1"

    # Configure models with proper settings
    if "gpt-5" in gen_model:
        lm = dspy.LM(
            model=f"openai/{gen_model}", 
            temperature=1.0, 
            max_tokens=16000,
            cache=False  # Disable caching
        )
    else:
        lm = dspy.LM(
            model=f"openai/{gen_model}", 
            max_tokens=2000,
            cache=False  # Disable caching
        )

    if "gpt-5" in ref_model:
        reflection_lm = dspy.LM(
            model=f"openai/{ref_model}",
            temperature=1.0,
            max_tokens=32000,  # Best practice: large context for reflection
            cache=False  # Disable caching
        )
    else:
        # For non-gpt-5 models, use best practice settings
        reflection_lm = dspy.LM(
            model=f"openai/{ref_model}",
            temperature=1.0,
            max_tokens=32000,  # Best practice for reflection
            cache=False  # Disable caching
        )

    return lm, reflection_lm


def get_gepa_params(trainset_size: int) -> Dict[str, int]:
    """
    Get GEPA optimizer parameters based on dataset size.

    Args:
        trainset_size: Number of training examples

    Returns:
        Dict with max_labeled_demos and max_bootstrapped_demos
    """
    return {
        "max_labeled_demos": min(5, trainset_size),
        "max_bootstrapped_demos": min(3, trainset_size),
    }


def extract_optimized_instruction(optimized: Any) -> str:
    """
    Extract the optimized instruction from a GEPA-optimized object.

    Args:
        optimized: The optimized object from GEPA

    Returns:
        The extracted instruction

    Raises:
        ValueError: If no optimized instruction can be extracted
    """
    # Use named_predictors() to access optimized instructions
    if hasattr(optimized, "named_predictors"):
        for name, predictor in optimized.named_predictors():
            if hasattr(predictor, "signature") and hasattr(
                predictor.signature, "instructions"
            ):
                instructions = predictor.signature.instructions
                if len(instructions) > 20 and instructions.strip():  # Relaxed guard with fallback
                    return instructions
                elif instructions.strip():  # Accept any non-empty instruction as fallback
                    return instructions

    # If no substantial optimized instruction found, this is an error
    raise ValueError(
        f"No optimized instruction found in {type(optimized)}. This indicates GEPA optimization failed or was not applied correctly."
    )


def evaluate_initial_performance(
    valset: List[dspy.Example], metric_func: Any, module: Any
) -> Dict[str, Any]:
    """
    Evaluate performance with default prompts on validation set.

    Args:
        valset: Validation examples
        metric_func: Metric function to evaluate predictions
        module: DSPy module to evaluate

    Returns:
        Dictionary with average score and individual scores
    """
    import logging

    logger = logging.getLogger(__name__)

    print("Evaluating baseline performance with default prompts...")
    scores = []

    for i, example in enumerate(valset):
        try:
            # Extract inputs from example
            inputs = {k: getattr(example, k) for k in example.inputs()}
            pred = module(**inputs)

            # Evaluate prediction
            result = metric_func(example, pred)
            score = result.score if hasattr(result, "score") else 0.0
            scores.append(score)

            if (i + 1) % 10 == 0:
                print(f"  Evaluated {i + 1}/{len(valset)} examples...")

        except Exception as e:
            logger.warning(f"Failed to evaluate example {i}: {e}")
            scores.append(0.0)

    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "avg_score": avg_score,
        "individual_scores": scores,
        "total_examples": len(valset),
    }


def get_optimizer_config(auto_level: str = "light") -> Dict[str, Any]:
    """
    Get standard GEPA optimizer configuration.

    Args:
        auto_level: Optimization level ("light", "medium", "heavy")

    Returns:
        Dictionary of optimizer kwargs
    """
    optimizer_kwargs = {
        "num_threads": 16,  # Balanced between performance and stability
        "reflection_minibatch_size": 3,  # Best practice efficiency setting
        "candidate_selection_strategy": "pareto",  # Best practice for diverse solutions
        "skip_perfect_score": True,  # Don't waste time on perfect examples
        "use_merge": False,  # Disable merge to simplify debugging
        "seed": 42,  # Reproducibility
        "track_stats": True,  # CRITICAL - enables detailed_results for reporting
        "track_best_outputs": True,  # Best practice - helpful for debugging
    }

    # Check for environment overrides
    max_metric_calls = os.getenv("GEPA_MAX_METRIC_CALLS")
    max_full_evals = os.getenv("GEPA_MAX_FULL_EVALS")

    if max_metric_calls:
        optimizer_kwargs["max_metric_calls"] = int(max_metric_calls)
        print(f"Using max_metric_calls: {max_metric_calls}")
    elif max_full_evals:
        optimizer_kwargs["max_full_evals"] = int(max_full_evals)
        print(f"Using max_full_evals: {max_full_evals}")
    else:
        optimizer_kwargs["auto"] = auto_level
        print(f"Using optimization level: {auto_level}")

    return optimizer_kwargs


def extract_signature_metadata(optimized: Any, default_system_prompt: str = "", default_user_prompt: str = "") -> Optional[Dict[str, Any]]:
    """
    Extract DSPy signature metadata from GEPA optimization results.

    Args:
        optimized: The optimized object from GEPA
        default_system_prompt: Default system prompt to combine with optimized instructions
        default_user_prompt: Default user prompt to combine with optimized instructions

    Returns:
        Dictionary containing signature metadata or None if extraction fails
    """
    try:
        if not hasattr(optimized, "named_predictors"):
            return None

        for name, predictor in optimized.named_predictors():
            if hasattr(predictor, "signature"):
                signature = predictor.signature
                
                # Extract field information
                input_fields = {}
                output_fields = {}
                
                if hasattr(signature, "input_fields"):
                    for field_name, field in signature.input_fields.items():
                        input_fields[field_name] = getattr(field, "desc", "")
                        
                if hasattr(signature, "output_fields"):
                    for field_name, field in signature.output_fields.items():
                        output_fields[field_name] = getattr(field, "desc", "")

                # Extract and combine instructions with system prompt context
                optimized_instructions = getattr(signature, "instructions", "")
                
                # Build combined instructions by merging default system prompt and optimized user prompt
                if default_system_prompt and optimized_instructions:
                    instructions = f"{default_system_prompt}\n\n{optimized_instructions}"
                elif default_system_prompt and default_user_prompt:
                    instructions = f"{default_system_prompt}\n\n{default_user_prompt}"
                elif optimized_instructions:
                    instructions = optimized_instructions
                else:
                    instructions = default_user_prompt or ""
                
                # Build signature string
                signature_str = str(signature) if hasattr(signature, "__str__") else ""
                
                # Determine predictor type by inspecting predictor classes
                predictor_type = "Predict"  # Default
                
                # Check predictor class type directly
                if isinstance(predictor, dspy.ChainOfThought):
                    predictor_type = "ChainOfThought"
                elif hasattr(dspy, 'TypedPredictor') and isinstance(predictor, dspy.TypedPredictor):
                    predictor_type = "TypedPredictor"
                else:
                    # Check if predictor has ChainOfThought characteristics by looking at its module
                    predictor_module = getattr(predictor, '__module__', '')
                    predictor_class_name = type(predictor).__name__
                    
                    if 'ChainOfThought' in predictor_class_name or 'chainofthought' in predictor_module.lower():
                        predictor_type = "ChainOfThought"
                    elif 'TypedPredictor' in predictor_class_name or 'typed' in predictor_module.lower():
                        predictor_type = "TypedPredictor"
                    # Fallback: check signature characteristics for reasoning patterns
                    elif hasattr(signature, "output_fields"):
                        output_field_names = list(signature.output_fields.keys())
                        if any("reasoning" in field.lower() or "rationale" in field.lower() for field in output_field_names):
                            predictor_type = "ChainOfThought"
                
                metadata = {
                    "input_fields": input_fields,
                    "output_fields": output_fields,
                    "instructions": instructions,
                    "signature_str": signature_str,
                    "predictor_type": predictor_type
                }
                
                return metadata
                
        return None
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to extract signature metadata: {e}")
        return None


def run_optimization(
    program_name: str,
    training_data_path: str,
    dspy_module: Any,
    metric_func: Any,
    prepare_examples_func: Any,
    default_system_prompt: str,
    default_user_prompt: str,
    create_training_data_func: Any,
) -> Tuple[str, str]:
    """
    Run GEPA optimization for a specific prompt optimization task.

    Args:
        program_name: Name of the program (e.g., "lv0_s1_concept_extraction")
        training_data_path: Path to training data JSON file
        dspy_module: DSPy module class to optimize
        metric_func: Metric function for GEPA
        prepare_examples_func: Function to prepare DSPy examples
        default_system_prompt: Default system prompt
        default_user_prompt: Default user prompt template
        create_training_data_func: Function to create training data

    Returns:
        Tuple of (system_prompt, user_prompt_template)
    """
    import time
    import warnings
    import asyncio
    from dspy.teleprompt import GEPA
    from prompt_optimization.core import save_prompt
    from prompt_optimization.reporter import create_optimization_report
    
    # Ensure proper asyncio event loop handling
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    print(f"Starting {program_name} prompt optimization...")
    start_time = time.time()

    # Configure LMs  
    gen_model = os.getenv("GEPA_GEN_MODEL")
    ref_model = os.getenv("GEPA_REFLECTION_MODEL")
    if not gen_model or not ref_model:
        raise ValueError("Generation and reflection models must be provided via CLI arguments")
    
    lm, reflection_lm = configure_openai_lms(gen_model, ref_model)
    dspy.settings.configure(
        lm=lm,
        trace=None,  # Disable tracing to avoid database issues
        cache=False  # Explicitly disable caching
    )

    # Load and prepare data
    print("Loading training data...")
    inputs, outputs = create_training_data_func()
    examples = prepare_examples_func(inputs, outputs)
    print(f"Loaded {len(examples)} training examples")

    # Split data
    trainset, valset = split_train_val(examples, 0.8)
    print(f"Split into {len(trainset)} train and {len(valset)} validation examples")

    # Create module instance
    module = dspy_module()

    # Evaluate baseline
    initial_scores = evaluate_initial_performance(valset, metric_func, module)
    print(f"Baseline average score: {initial_scores['avg_score']:.3f}")

    # Configure optimizer
    print("Configuring GEPA optimizer...")
    auto_level = os.getenv("GEPA_AUTO", "light")
    optimizer_kwargs = get_optimizer_config(auto_level)
    optimizer_kwargs["metric"] = metric_func
    optimizer_kwargs["reflection_lm"] = reflection_lm

    optimizer = GEPA(**optimizer_kwargs)

    # Run optimization
    print("Running GEPA optimization (this may take a while)...")
    optimized = optimizer.compile(module, trainset=trainset, valset=valset)

    print("\nExtracting optimized prompts...")
    user_prompt_template = extract_optimized_instruction(optimized)

    # Extract signature metadata from optimized object
    print("Extracting signature metadata...")
    signature_metadata = None
    try:
        signature_metadata = extract_signature_metadata(optimized, default_system_prompt, default_user_prompt)
        if signature_metadata:
            print("✓ Successfully extracted signature metadata")
        else:
            print("⚠ No signature metadata available, falling back to text-only")
    except Exception as e:
        print(f"⚠ Failed to extract signature metadata: {e}")
        # Continue with text-only approach

    # Save prompts
    print("Saving optimized prompts...")
    prompt_key_prefix = program_name.replace("_", "_").split("_")[:2]  # e.g., "lv0_s1"
    prompt_key_prefix = "_".join(prompt_key_prefix)

    system_path = save_prompt(f"{prompt_key_prefix}_system", default_system_prompt, signature_metadata=signature_metadata)
    user_path = save_prompt(f"{prompt_key_prefix}_user", user_prompt_template, signature_metadata=signature_metadata)

    print(f"✓ Saved system prompt to: {system_path}")
    print(f"✓ Saved user prompt to: {user_path}")

    # Calculate duration
    duration = time.time() - start_time
    duration_str = f"{int(duration // 60)}m {int(duration % 60)}s"

    # Generate report
    print("\nGenerating optimization report...")
    try:
        # Clean optimizer_kwargs for JSON serialization - remove functions
        serializable_config = {
            k: v
            for k, v in optimizer_kwargs.items()
            if not callable(v) and k not in ["metric", "reflection_lm"]
        }
        serializable_config["metric_name"] = getattr(
            metric_func, "__name__", "metric_with_feedback"
        )
        serializable_config["reflection_model"] = getattr(
            reflection_lm, "model", "unknown"
        )

        # Collect prompt information for the report
        prompts_info = {
            "initial_system": default_system_prompt,
            "initial_user": default_user_prompt,
            "optimized_system": default_system_prompt,  # System prompt typically doesn't change
            "optimized_user": user_prompt_template,
        }

        report_paths = create_optimization_report(
            initial_scores=initial_scores,
            optimized_program=optimized,
            detailed_results=getattr(optimized, "detailed_results", None),
            prompts=prompts_info,
            metadata={
                "program_name": program_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration": duration_str,
                "training_data": {
                    "source": training_data_path,
                    "examples": len(examples),
                    "train": len(trainset),
                    "validation": len(valset),
                },
                "optimizer_config": serializable_config,
                "generated_files": [system_path, user_path],
                "signature_metadata_extracted": signature_metadata is not None,
            },
        )

        print("\nOptimization report generated:")
        print(f"  Summary: {report_paths['txt_path']}")
        print(f"  Details: {report_paths['json_path']}")
        print(f"  Latest: {report_paths['txt_latest_path']}")

    except Exception as e:
        print(f"Warning: Failed to generate optimization report: {e}")

    # Test optimized program
    if len(valset) + len(trainset) > 0:
        print("\nTesting optimized program on validation example...")
        test_example = valset[0] if valset else trainset[0]
        inputs = {k: getattr(test_example, k) for k in test_example.inputs()}
        test_result = optimized(**inputs)

        # Print test results (customize per task)
        print(f"Test input: {str(inputs)[:200]}...")
        print(f"Test output: {str(test_result)[:200]}...")
    else:
        print("\nWarning: No examples available for testing")

    print(f"\n✅ Optimization complete! (took {duration_str})")
    
    # Clean up event loop to prevent warnings
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.stop()
    except Exception:
        pass  # Ignore cleanup errors
    
    # Clean up temp cache directory if it exists
    try:
        import shutil
        cache_dir = os.environ.get("DSP_CACHEDIR", "")
        if cache_dir and cache_dir.startswith("/tmp/dspy_cache_") and os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
    except Exception:
        pass  # Ignore cleanup errors
    
    return default_system_prompt, user_prompt_template
