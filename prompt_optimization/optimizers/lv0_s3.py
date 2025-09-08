"""Simplified standalone optimizer for lv0_s3 discipline verification using DSPy GEPA.

Environment Variables:
    GEPA_GEN_MODEL: Generation model (default: gpt-4o-mini)
    GEPA_REFLECTION_MODEL: Reflection model (default: same as GEPA_GEN_MODEL)
"""

import sys
from typing import Any, Dict, List

import dspy
from dotenv import load_dotenv
from dspy.teleprompt import GEPA

from prompt_optimization.core import save_prompt
from prompt_optimization.optimizers.common import (
    configure_openai_lms,
    extract_optimized_instruction,
    load_json_training,
    split_train_val,
)

load_dotenv()

# Training data path constant
TRAINING_DATA_PATH = "data/prompts_training_data/lv0_s3.json"


class AcademicDisciplineVerification(dspy.Signature):
    """DSPy signature for verifying if a term is an academic discipline."""

    instruction = dspy.InputField(desc="System instructions for verification")
    term = dspy.InputField(desc="Term to verify as academic discipline")
    is_discipline = dspy.OutputField(
        desc="Boolean indicating if term is an academic discipline"
    )
    reasoning = dspy.OutputField(desc="Explanation for the decision")


class DisciplineVerifier(dspy.Module):
    """DSPy module for verifying academic disciplines."""

    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict(AcademicDisciplineVerification)

    def forward(self, instruction: str, term: str) -> dspy.Prediction:
        """Verify if a term is an academic discipline."""
        return self.prog(instruction=instruction, term=term)


def load_training_data(
    file_path: str = TRAINING_DATA_PATH,
) -> List[Dict[str, Any]]:
    """Load training data from JSON file."""
    return load_json_training(file_path)


def create_training_data() -> tuple[List[str], List[Dict[str, Any]]]:
    """Create training data from JSON file."""
    training_data = load_training_data()

    inputs = []
    outputs = []

    for item in training_data:
        inputs.append(item["keyword"])

        # Output is the expected verification result with reasoning
        output = {
            "is_discipline": item["expected"],
            "reasoning": f"Based on evidence from: {', '.join(item.get('evidence_colleges', []))}",
        }
        outputs.append(output)

    return inputs, outputs


def prepare_dspy_examples(
    inputs: List[str], outputs: List[Dict[str, Any]]
) -> List[dspy.Example]:
    """Convert training data to DSPy examples."""
    examples = []

    for term, expected_output in zip(inputs, outputs):
        example = dspy.Example(
            instruction="Determine if the term represents an academic discipline.",
            term=term,
            is_discipline=str(expected_output["is_discipline"]).lower(),
            reasoning=expected_output["reasoning"],
        ).with_inputs("instruction", "term")

        examples.append(example)

    return examples


def metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Metric function for GEPA that evaluates discipline verification accuracy.
    Returns dspy.Prediction with score and feedback.
    """
    try:
        pred_is_discipline = pred.is_discipline.lower() in ["true", "yes", "1"]
        gold_is_discipline = gold.is_discipline.lower() in ["true", "yes", "1"]

        correct = pred_is_discipline == gold_is_discipline

        score = 1.0 if correct else 0.0

        if hasattr(pred, "reasoning") and pred.reasoning:
            reasoning_len = len(pred.reasoning.strip())
            if reasoning_len < 20:
                score *= 0.8
                feedback = "Reasoning too brief. Provide more detailed explanation."
            elif reasoning_len > 500:
                score *= 0.9
                feedback = "Reasoning too verbose. Be more concise."
            else:
                feedback = "Good classification with appropriate reasoning."
        else:
            score *= 0.5
            feedback = "Missing reasoning. Always explain your decision."

        if not correct:
            if pred_is_discipline and not gold_is_discipline:
                feedback = f"False positive: '{gold.term}' is not an academic discipline. Be more selective."
            else:
                feedback = f"False negative: '{gold.term}' is an academic discipline. Consider broader academic fields."

        return dspy.Prediction(score=score, feedback=feedback)

    except Exception as e:
        return dspy.Prediction(
            score=0.0,
            feedback=f"Error evaluating prediction: {str(e)}. Ensure proper output format.",
        )


def optimize_prompts():
    """Run GEPA optimization for discipline verification prompts."""
    print("Starting lv0_s3 discipline verification prompt optimization...")

    lm, reflection_lm = configure_openai_lms()
    dspy.settings.configure(lm=lm)

    print("Loading training data...")
    inputs, outputs = create_training_data()
    examples = prepare_dspy_examples(inputs, outputs)
    print(f"Loaded {len(examples)} training examples")

    # Split into train and validation using common helper
    trainset, valset = split_train_val(examples, 0.8)
    print(f"Split into {len(trainset)} train and {len(valset)} validation examples")

    student = DisciplineVerifier()

    print("Configuring GEPA optimizer...")
    import os

    max_metric_calls = os.getenv("GEPA_MAX_METRIC_CALLS")
    max_full_evals = os.getenv("GEPA_MAX_FULL_EVALS")
    auto_level = os.getenv("GEPA_AUTO", "light")

    # Configure GEPA following best practices from research paper
    optimizer_kwargs = {
        "metric": metric_with_feedback,
        "reflection_lm": reflection_lm,
        "num_threads": 8,  # Balanced between performance and stability (best practice: 16-32)
        "reflection_minibatch_size": 3,  # Best practice efficiency setting
        "candidate_selection_strategy": "pareto",  # Best practice for diverse solutions
        "skip_perfect_score": True,  # Don't waste time on perfect examples
        "use_merge": False,  # Disable merge to simplify debugging
        "seed": 42,  # Reproducibility
        "track_stats": True,  # CRITICAL - enables detailed_results for reporting
        "track_best_outputs": True,  # Best practice - helpful for debugging and analysis
    }

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

    DEFAULT_S3_USER = """Analyze whether "{keyword}" is a valid broad academic discipline.\n\n"
        "Evidence - Colleges/schools/divisions that mention this concept:\n{evidence_colleges}\n\n"
        "Consider:\n1. Is it a recognized major field of study or broad academic discipline?\n"
        "2. Does it represent a major division of knowledge in academia?\n"
        "3. Is it broad enough to encompass multiple research areas or subdisciplines?\n\n"
        "Answer with true if it meets these criteria, false otherwise."""

    # System prompt that clearly identifies academic disciplines
    system_prompt = """You are an expert at identifying academic disciplines and fields of study.\n"
        "Your task is to determine whether a given term represents a legitimate academic discipline.\n"
        "Consider established fields taught at universities, research areas, and recognized academic domains.\n"
        "Be precise and avoid classifying general terms, companies, or non-academic concepts as disciplines."""

    # Extract optimized instruction using common helper
    user_prompt_template = extract_optimized_instruction(optimized, DEFAULT_S3_USER)

    print("Saving optimized prompts...")

    system_path = save_prompt("lv0_s3_system", system_prompt)
    user_path = save_prompt("lv0_s3_user", user_prompt_template)

    print(f"✓ Saved system prompt to: {system_path}")
    print(f"✓ Saved user prompt to: {user_path}")

    # Test the optimized program on a validation example (only if we have examples)
    if len(valset) + len(trainset) > 0:
        print("\nTesting optimized program on validation example...")
        test_example = valset[0] if valset else trainset[0]
        test_result = optimized(
            instruction=test_example.instruction, term=test_example.term
        )
        print(f"Term: {test_example.term}")
        print(f"Is discipline: {test_result.is_discipline}")
        print(f"Reasoning: {test_result.reasoning[:200]}...")
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
