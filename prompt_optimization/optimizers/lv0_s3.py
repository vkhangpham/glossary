"""Simplified standalone optimizer for lv0_s3 discipline verification using DSPy GEPA.

Environment Variables:
    GEPA_GEN_MODEL: Generation model (default: gpt-4o-mini)
    GEPA_REFLECTION_MODEL: Reflection model (default: same as GEPA_GEN_MODEL)
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

import dspy
from dspy.teleprompt import GEPA

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prompt_optimization.save import save_prompt
from prompt_optimization.optimizers.common import (
    load_json_training, split_train_val, configure_openai_lms,
    get_gepa_params, extract_optimized_instruction
)


class AcademicDisciplineVerification(dspy.Signature):
    """DSPy signature for verifying if a term is an academic discipline."""
    instruction = dspy.InputField(desc="System instructions for verification")
    term = dspy.InputField(desc="Term to verify as academic discipline")
    is_discipline = dspy.OutputField(desc="Boolean indicating if term is an academic discipline")
    reasoning = dspy.OutputField(desc="Explanation for the decision")


class DisciplineVerifier(dspy.Module):
    """DSPy module for verifying academic disciplines."""
    
    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict(AcademicDisciplineVerification)
    
    def forward(self, instruction: str, term: str) -> dspy.Prediction:
        """Verify if a term is an academic discipline."""
        return self.prog(instruction=instruction, term=term)


def load_training_data(file_path: str = "data/training/lv0_s3.json") -> List[Dict[str, Any]]:
    """Load training data from JSON file."""
    return load_json_training(file_path)


def create_training_data() -> tuple[List[str], List[Dict[str, Any]]]:
    """Create training data from JSON file."""
    training_data = load_training_data()
    
    inputs = []
    outputs = []
    
    for item in training_data:
        # Input is the term to verify
        inputs.append(item["input"])
        
        # Output is the expected verification result
        outputs.append(item["expected_output"])
    
    return inputs, outputs


def prepare_dspy_examples(inputs: List[str], outputs: List[Dict[str, Any]]) -> List[dspy.Example]:
    """Convert training data to DSPy examples."""
    examples = []
    
    for term, expected_output in zip(inputs, outputs):
        example = dspy.Example(
            instruction="Determine if the term represents an academic discipline.",
            term=term,
            is_discipline=str(expected_output["is_discipline"]).lower(),
            reasoning=expected_output["reasoning"]
        ).with_inputs("instruction", "term")
        
        examples.append(example)
    
    return examples


def metric_with_feedback(gold, pred, trace=None):
    """
    Metric function for GEPA that evaluates discipline verification accuracy.
    Returns (score, feedback) tuple.
    """
    try:
        # Parse predictions
        pred_is_discipline = pred.is_discipline.lower() in ['true', 'yes', '1']
        gold_is_discipline = gold.is_discipline.lower() in ['true', 'yes', '1']
        
        # Check if classification is correct
        correct = pred_is_discipline == gold_is_discipline
        
        # Base score on correctness
        score = 1.0 if correct else 0.0
        
        # Evaluate reasoning quality if present
        if hasattr(pred, 'reasoning') and pred.reasoning:
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
        
        # Additional feedback for incorrect classifications
        if not correct:
            if pred_is_discipline and not gold_is_discipline:
                feedback = f"False positive: '{gold.term}' is not an academic discipline. Be more selective."
            else:
                feedback = f"False negative: '{gold.term}' is an academic discipline. Consider broader academic fields."
        
        return score, feedback
        
    except Exception as e:
        return 0.0, f"Error evaluating prediction: {str(e)}. Ensure proper output format."


def optimize_prompts():
    """Run GEPA optimization for discipline verification prompts."""
    print("Starting lv0_s3 discipline verification prompt optimization...")
    
    # Configure language models from common helper
    lm, reflection_lm = configure_openai_lms()
    dspy.settings.configure(lm=lm)
    
    # Load training data
    print("Loading training data...")
    inputs, outputs = create_training_data()
    examples = prepare_dspy_examples(inputs, outputs)
    print(f"Loaded {len(examples)} training examples")
    
    # Split into train and validation using common helper
    trainset, valset = split_train_val(examples, 0.8)
    print(f"Split into {len(trainset)} train and {len(valset)} validation examples")
    
    # Create student program
    student = DisciplineVerifier()
    
    # Configure GEPA optimizer with appropriate params based on dataset size
    print("Configuring GEPA optimizer...")
    gepa_params = get_gepa_params(len(trainset))
    optimizer = GEPA(
        metric=metric_with_feedback,
        max_bootstrapped_demos=gepa_params["max_bootstrapped_demos"],
        max_labeled_demos=gepa_params["max_labeled_demos"],
        num_threads=4,
        auto="light"  # Use light mode for faster optimization
    )
    
    # Set the reflection LM for GEPA
    optimizer.reflection_lm = reflection_lm
    
    # Run optimization
    print("Running GEPA optimization (this may take a while)...")
    optimized = optimizer.compile(
        student,
        trainset=trainset,
        valset=valset
    )
    
    # Extract optimized prompts
    print("\nExtracting optimized prompts...")
    
    # Define default templates with required placeholders
    DEFAULT_S3_USER = (
        """Analyze whether "{keyword}" is a valid broad academic discipline.\n\n"
        "Evidence - Colleges/schools/divisions that mention this concept:\n{evidence_colleges}\n\n"
        "Consider:\n1. Is it a recognized major field of study or broad academic discipline?\n"
        "2. Does it represent a major division of knowledge in academia?\n"
        "3. Is it broad enough to encompass multiple research areas or subdisciplines?\n\n"
        "Answer with true if it meets these criteria, false otherwise."""
    )
    
    # System prompt that clearly identifies academic disciplines
    system_prompt = (
        """You are an expert at identifying academic disciplines and fields of study.\n"
        "Your task is to determine whether a given term represents a legitimate academic discipline.\n"
        "Consider established fields taught at universities, research areas, and recognized academic domains.\n"
        "Be precise and avoid classifying general terms, companies, or non-academic concepts as disciplines."""
    )
    
    # Extract optimized instruction using common helper
    user_prompt_template = extract_optimized_instruction(optimized, DEFAULT_S3_USER)
    
    # Save optimized prompts
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
            instruction=test_example.instruction,
            term=test_example.term
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