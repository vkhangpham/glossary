"""Simplified standalone optimizer for lv0_s1 concept extraction using DSPy GEPA.

Environment Variables:
    GEPA_GEN_MODEL: Generation model (default: gpt-4o-mini)
    GEPA_REFLECTION_MODEL: Reflection model (default: same as GEPA_GEN_MODEL)
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

import dspy
from dspy.teleprompt import GEPA

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prompt_optimization.simple_save import save_simple_prompt
from prompt_optimization.optimizers.common import (
    load_json_training, split_train_val, configure_openai_lms,
    get_gepa_params, extract_optimized_instruction
)


class ConceptExtraction(BaseModel):
    """Model for a single concept extraction from source text."""
    source: str = Field(description="The original text this extraction came from")
    concepts: List[str] = Field(description="List of extracted academic concepts")
    reasoning: str = Field(description="Explanation of why these are valid concepts")


class ExtractConceptsSignature(dspy.Signature):
    """DSPy signature for concept extraction from academic texts."""
    instruction = dspy.InputField(desc="System instructions for extraction")
    text = dspy.InputField(desc="Text to extract concepts from")
    extraction = dspy.OutputField(desc="JSON object with source, concepts, and reasoning")


class ConceptExtractor(dspy.Module):
    """DSPy module for extracting academic concepts."""
    
    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict(ExtractConceptsSignature)
    
    def forward(self, instruction: str, text: str) -> dspy.Prediction:
        """Extract concepts from text."""
        return self.prog(instruction=instruction, text=text)


def load_training_data(file_path: str = "data/training/lv0_s1.json") -> List[Dict[str, Any]]:
    """Load training data from JSON file."""
    return load_json_training(file_path)


def create_training_data() -> tuple[List[str], List[List[Dict[str, Any]]]]:
    """Create training data from JSON file."""
    training_data = load_training_data()
    
    inputs = []
    outputs = []
    
    for item in training_data:
        # Input is the text to extract from
        inputs.append(item["input"])
        
        # Output is the expected extractions
        outputs.append(item["expected_output"])
    
    return inputs, outputs


def prepare_dspy_examples(inputs: List[str], outputs: List[List[Dict[str, Any]]]) -> List[dspy.Example]:
    """Convert training data to DSPy examples."""
    examples = []
    
    for text, expected_extractions in zip(inputs, outputs):
        # Format expected output as JSON string
        formatted_output = json.dumps(expected_extractions, indent=2)
        
        example = dspy.Example(
            instruction="Extract academic concepts from the given text.",
            text=text,
            extraction=formatted_output
        ).with_inputs("instruction", "text")
        
        examples.append(example)
    
    return examples


def metric_with_feedback(gold, pred, trace=None):
    """
    Metric function for GEPA that evaluates concept extraction quality.
    Returns (score, feedback) tuple.
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
        total_gold_concepts = sum(len(ext.get("concepts", [])) for ext in gold_extractions)
        
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
            return 0.0, "No gold concepts to compare against"
        
        matches = pred_concepts & gold_concepts
        precision = len(matches) / len(pred_concepts) if pred_concepts else 0
        recall = len(matches) / len(gold_concepts) if gold_concepts else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Generate feedback
        feedback = []
        if f1 < 0.5:
            feedback.append("Low F1 score indicates poor concept extraction.")
            if precision < recall:
                feedback.append("Missing many expected concepts. Be more comprehensive.")
            else:
                feedback.append("Extracting too many irrelevant concepts. Be more selective.")
        
        if not pred_concepts:
            feedback.append("No concepts extracted. Ensure you're identifying academic terms.")
        
        # Check for structure
        if not isinstance(pred_extraction, (list, dict)):
            feedback.append("Output should be a valid JSON structure.")
        
        feedback_str = " ".join(feedback) if feedback else "Good extraction quality."
        
        return f1, feedback_str
        
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return 0.0, f"Failed to parse extraction: {str(e)}. Ensure valid JSON output."


def optimize_prompts():
    """Run GEPA optimization for concept extraction prompts."""
    print("Starting lv0_s1 concept extraction prompt optimization...")
    
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
    student = ConceptExtractor()
    
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
    DEFAULT_S1_USER = (
        """Extract academic concepts from these institutions:\n\n{sources}\n\n"
        "For each institution, identify the core academic fields and disciplines."""
    )
    
    # System prompt that clearly specifies the expected output schema
    system_prompt = (
        """You are an expert at extracting academic concepts from text.\n"
        "Your task is to extract, for each provided source, a JSON object with:\n"
        "- source: the original text\n"
        "- concepts: list of extracted academic concepts (lowercase, deduplicated)\n"
        "Return a JSON array named 'extractions' containing these objects."""
    )
    
    # Extract optimized instruction using common helper
    user_prompt_template = extract_optimized_instruction(optimized, DEFAULT_S1_USER)
    
    # Save optimized prompts
    print("Saving optimized prompts...")
    
    system_path = save_simple_prompt("lv0_s1_system", system_prompt)
    user_path = save_simple_prompt("lv0_s1_user", user_prompt_template)
    
    print(f"✓ Saved system prompt to: {system_path}")
    print(f"✓ Saved user prompt to: {user_path}")
    
    # Test the optimized program on a validation example (only if we have examples)
    if len(valset) + len(trainset) > 0:
        print("\nTesting optimized program on validation example...")
        test_example = valset[0] if valset else trainset[0]
        test_result = optimized(
            instruction=test_example.instruction,
            text=test_example.text
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