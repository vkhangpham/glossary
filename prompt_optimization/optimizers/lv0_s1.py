"""Simplified standalone optimizer for lv0_s1 concept extraction using DSPy GEPA.

Environment Variables:
    GEPA_GEN_MODEL: Generation model (default: gpt-4o-mini)
    GEPA_REFLECTION_MODEL: Reflection model (default: same as GEPA_GEN_MODEL)
"""

import json
import sys
from typing import Any, Dict, List

# Load environment variables from .env file if it exists
from dotenv import load_dotenv

load_dotenv()

import dspy  # noqa: E402
from dspy.teleprompt import GEPA  # noqa: E402

from prompt_optimization.core import save_prompt  # noqa: E402
from prompt_optimization.optimizers.common import (  # noqa: E402
    configure_openai_lms,
    extract_optimized_instruction,
    load_json_training,
    split_train_val,
)

# Training data path constant
TRAINING_DATA_PATH = "data/prompts_training_data/lv0_s1.json"


class ExtractConceptsSignature(dspy.Signature):
    """DSPy signature for concept extraction from academic texts."""

    instruction = dspy.InputField(
        desc="System instructions for extraction", prefix="Instructions:"
    )
    text = dspy.InputField(desc="Text to extract concepts from", prefix="Input Text:")
    extraction = dspy.OutputField(
        desc="JSON array of extraction objects, each with 'source' and 'concepts' fields",
        prefix="Extraction (JSON):",
    )


class ConceptExtractor(dspy.Module):
    """DSPy module for extracting academic concepts."""

    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict(ExtractConceptsSignature)

    def forward(self, instruction: str, text: str) -> dspy.Prediction:
        """Extract concepts from text."""
        return self.prog(instruction=instruction, text=text)


def load_training_data(
    file_path: str = TRAINING_DATA_PATH,
) -> List[Dict[str, Any]]:
    """Load training data from JSON file."""
    return load_json_training(file_path)


def create_training_data() -> tuple[List[str], List[List[Dict[str, Any]]]]:
    """Create training data from JSON file."""
    training_data = load_training_data()

    inputs = []
    outputs = []

    for item in training_data:
        inputs.append(item["input"])

        # Transform expected concepts list into the format expected by the metric
        # The metric expects a list of dicts with "source" and "concepts" fields
        expected_extraction = [
            {
                "source": item["input"],
                "concepts": item["expected"],  # This is the list of concepts
            }
        ]
        outputs.append(expected_extraction)

    return inputs, outputs


def prepare_dspy_examples(
    inputs: List[str], outputs: List[List[Dict[str, Any]]]
) -> List[dspy.Example]:
    """Convert training data to DSPy examples."""
    examples = []

    for text, expected_extractions in zip(inputs, outputs):
        # Format expected output as JSON string
        formatted_output = json.dumps(expected_extractions, indent=2)

        example = dspy.Example(
            instruction="Extract academic concepts from the given text.",
            text=text,
            extraction=formatted_output,
        ).with_inputs("instruction", "text")

        examples.append(example)

    return examples


def metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Metric function for GEPA that evaluates concept extraction quality.
    Returns dspy.Prediction with score and feedback.
    """
    try:
        # Parse predicted extraction
        pred_extraction = json.loads(pred.extraction)

        # Parse gold extraction
        if isinstance(gold.extraction, str):
            gold_extractions = json.loads(gold.extraction)
        else:
            gold_extractions = gold.extraction

        # Calculate scores

        # Check if prediction is a list of extractions
        if isinstance(pred_extraction, list):
            pred_concepts = set()
            for ext in pred_extraction:
                pred_concepts.update(ext.get("concepts", []))
        else:
            pred_concepts = set(pred_extraction.get("concepts", []))

        # Get gold concepts
        gold_concepts = set()
        for ext in gold_extractions:
            gold_concepts.update(ext.get("concepts", []))

        # Calculate metrics
        if not gold_concepts:
            return dspy.Prediction(
                score=0.0, feedback="No gold concepts to compare against"
            )

        matches = pred_concepts & gold_concepts
        precision = len(matches) / len(pred_concepts) if pred_concepts else 0
        recall = len(matches) / len(gold_concepts) if gold_concepts else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Generate feedback
        feedback = []
        if f1 < 0.5:
            feedback.append("Low F1 score indicates poor concept extraction.")
            if precision < recall:
                feedback.append(
                    "Missing many expected concepts. Be more comprehensive."
                )
            else:
                feedback.append(
                    "Extracting too many irrelevant concepts. Be more selective."
                )

        if not pred_concepts:
            feedback.append(
                "No concepts extracted. Ensure you're identifying academic terms."
            )

        # Check for structure
        if not isinstance(pred_extraction, (list, dict)):
            feedback.append("Output should be a valid JSON structure.")

        feedback_str = " ".join(feedback) if feedback else "Good extraction quality."

        return dspy.Prediction(score=f1, feedback=feedback_str)

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return dspy.Prediction(
            score=0.0,
            feedback=f"Failed to parse extraction: {str(e)}. Ensure valid JSON output.",
        )


def optimize_prompts():
    """Run GEPA optimization for concept extraction prompts."""
    print("Starting lv0_s1 concept extraction prompt optimization...")

    lm, reflection_lm = configure_openai_lms()
    dspy.settings.configure(lm=lm)

    print("Loading training data...")
    inputs, outputs = create_training_data()
    examples = prepare_dspy_examples(inputs, outputs)
    print(f"Loaded {len(examples)} training examples")

    # Split into train and validation using common helper
    trainset, valset = split_train_val(examples, 0.8)
    print(f"Split into {len(trainset)} train and {len(valset)} validation examples")

    student = ConceptExtractor()

    print("Configuring GEPA optimizer...")
    import os

    max_metric_calls = os.getenv("GEPA_MAX_METRIC_CALLS")
    max_full_evals = os.getenv("GEPA_MAX_FULL_EVALS")
    auto_level = os.getenv("GEPA_AUTO", "light")

    optimizer_kwargs = {
        "metric": metric_with_feedback,
        "reflection_lm": reflection_lm,
        "num_threads": 2,  # Reduced to avoid concurrent futures errors
        "reflection_minibatch_size": 2,  # Smaller batch for stability
        "candidate_selection_strategy": "pareto",  # Best strategy for diverse solutions
        "skip_perfect_score": True,  # Don't waste time on perfect examples
        "use_merge": False,  # Disable merge to simplify debugging
        "seed": 42,  # Reproducibility
    }

    # Set budget parameter (only one should be set)
    if max_metric_calls:
        optimizer_kwargs["max_metric_calls"] = int(max_metric_calls)
        print(f"Using max_metric_calls: {max_metric_calls}")
    elif max_full_evals:
        optimizer_kwargs["max_full_evals"] = int(max_full_evals)
        print(f"Using max_full_evals: {max_full_evals}")
    else:
        optimizer_kwargs["auto"] = auto_level
        print(f"Using optimization level: {auto_level}")

    optimizer = GEPA(**optimizer_kwargs)

    print("Running GEPA optimization (this may take a while)...")
    optimized = optimizer.compile(student, trainset=trainset, valset=valset)

    print("\nExtracting optimized prompts...")

    # Define default templates with required placeholders
    DEFAULT_S1_USER = """Extract academic concepts from these institutions:\n\n{sources}\n\n"
        "For each institution, identify the core academic fields and disciplines.\n"
        "Return a JSON array where each object has 'source' (the institution name) and 'concepts' (list of concepts)."""

    # System prompt that clearly specifies the expected output schema
    system_prompt = """You are an expert at extracting academic concepts from text.\n"
        "Your task is to extract, for each provided source, a JSON object with:\n"
        "- source: the original text\n"
        "- concepts: list of extracted academic concepts (lowercase, deduplicated)\n"
        "Return a JSON array named 'extractions' containing these objects."""

    # Extract optimized instruction using common helper
    user_prompt_template = extract_optimized_instruction(optimized, DEFAULT_S1_USER)

    print("Saving optimized prompts...")

    system_path = save_prompt("lv0_s1_system", system_prompt)
    user_path = save_prompt("lv0_s1_user", user_prompt_template)

    print(f"✓ Saved system prompt to: {system_path}")
    print(f"✓ Saved user prompt to: {user_path}")

    # Test the optimized program on a validation example (only if we have examples)
    if len(valset) + len(trainset) > 0:
        print("\nTesting optimized program on validation example...")
        test_example = valset[0] if valset else trainset[0]
        test_result = optimized(
            instruction=test_example.instruction, text=test_example.text
        )
        print(f"Input text: {test_example.text[:100]}...")
        print(f"Extracted concepts: {test_result.extraction[:200]}...")
    else:
        print("\nWarning: No examples available for testing")

    print("\n✅ Optimization complete!")

    return system_prompt, user_prompt_template


if __name__ == "__main__":
    try:
        optimize_prompts()
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        sys.exit(1)
