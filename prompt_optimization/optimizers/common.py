"""
Common helper functions for prompt optimizers.

Environment Variables:
    GEPA_GEN_MODEL: Generation model (default: gpt-5-nano)
    GEPA_REFLECTION_MODEL: Reflection model (default: gpt-5)
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
    Split examples into training and validation sets.

    Args:
        examples: List of examples to split (can be dspy.Example or dict)
        ratio: Fraction for training set (default: 0.8)

    Returns:
        Tuple of (trainset, valset)
    """
    if len(examples) == 0:
        raise ValueError("Cannot split empty dataset")

    if len(examples) == 1:
        # With only one example, use it for both train and val
        return examples, examples

    train_size = max(1, int(ratio * len(examples)))
    trainset = examples[:train_size]
    valset = examples[train_size:]

    return trainset, valset


def configure_openai_lms(api_key: Optional[str] = None) -> Tuple[Any, Any]:
    """
    Configure OpenAI language models for generation and reflection.

    Args:
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

    gen_model = os.getenv(
        "GEPA_GEN_MODEL", "gpt-5-nano"
    )  # Default: efficient task model
    ref_model = os.getenv(
        "GEPA_REFLECTION_MODEL", "gpt-5"
    )  # Default: strong reflection model (best practice)

    os.environ["OPENAI_API_KEY"] = api_key

    if "gpt-5" in gen_model:
        lm = dspy.LM(model=f"openai/{gen_model}", temperature=1.0, max_tokens=16000)
    else:
        lm = dspy.LM(model=f"openai/{gen_model}", max_tokens=2000)

    if "gpt-5" in ref_model:
        reflection_lm = dspy.LM(
            model=f"openai/{ref_model}",
            temperature=1.0,
            max_tokens=32000,  # Best practice: large context for reflection
        )
    else:
        # For non-gpt-5 models, use best practice settings
        reflection_lm = dspy.LM(
            model=f"openai/{ref_model}",
            temperature=1.0,
            max_tokens=32000,  # Best practice for reflection
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


def extract_optimized_instruction(optimized: Any, default_fallback: str) -> str:
    """
    Extract the optimized instruction from a GEPA-optimized object.

    Args:
        optimized: The optimized object from GEPA
        default_fallback: Default template to use if extraction fails

    Returns:
        The extracted instruction or default fallback
    """
    raw_instruction = getattr(getattr(optimized, "prog", None), "signature", None)
    raw_instruction = getattr(raw_instruction, "instruction", None)

    extracted = raw_instruction if isinstance(raw_instruction, str) else None

    if not extracted or "InputField(" in str(raw_instruction):
        return default_fallback

    return f"{extracted}\n\n{default_fallback}"
