"""Template for creating new prompt optimizers.

Copy this file and modify for your specific prompt optimization needs.
Naming convention: {level}_s{step}.py (e.g., lv0_s1.py, lv1_s2.py)

For examples of working optimizers, see:
- lv0_s1.py
- lv0_s3.py

Usage:
- Direct execution: python prompt_optimization/optimizers/{level}_s{step}.py
- Runner script: python run_{level}_s{step}_optimization.py
"""

from pathlib import Path
from typing import Dict, List, Literal, cast

import dspy
from dspy import Example
from pydantic import BaseModel, Field

from prompt_optimization.core import save_prompt


# TODO: Define your Pydantic models matching the task
# For actual examples, see lv0_s1.py (ConceptExtraction model)
# or lv0_s3.py (SingleTokenConcept model)
class YourOutputModel(BaseModel):
    """Define the expected output structure"""

    field1: str = Field(description="Description of field1")
    field2: List[str] = Field(description="Description of field2")


# TODO: Define the DSPy signature for your task
# For actual examples, see ExtractConceptsSignature in lv0_s1.py
# or VerifySingleTokenSignature in lv0_s3.py
class YourTaskSignature(dspy.Signature):
    """One-line description of what this does"""

    input_field: str = dspy.InputField(desc="Description of the input")
    output_field: YourOutputModel = dspy.OutputField(desc="Description of the output")


# DSPy program to optimize
class YourProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        # Choose appropriate DSPy module: ChainOfThought, Predict, etc.
        self.process = dspy.ChainOfThought(YourTaskSignature)

    def forward(self, input_field):
        result = self.process(input_field=input_field)
        return result


def create_training_data():
    """Create training data for your task

    These are manually crafted examples for GEPA optimization.
    TODO: Replace with actual training examples for your task.
    """

    # TODO: Create training examples specific to your task
    training_examples = [
        {
            "input": {"input_field": "Example input 1"},
            "output": {
                "output_field": {
                    "field1": "Expected output 1",
                    "field2": ["item1", "item2"],
                }
            },
        },
    ]

    return training_examples


def prepare_dspy_examples(training_data: List[Dict]) -> List[Example]:
    """Convert training data to DSPy Examples

    TODO: Update this to match your data structure
    """
    examples = []

    for item in training_data:
        example = Example(
            input_field=item["input"]["input_field"],
            output_field=YourOutputModel(**item["output"]["output_field"]),
        ).with_inputs("input_field")

        examples.append(example)

    return examples


def metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Metric function for GEPA that evaluates quality with feedback

    GEPA requires this specific signature with 5 arguments, but DSPy's evaluate
    might call it with just 2, so we make the last 3 optional.
    TODO: Implement your evaluation metric
    """

    try:
        # TODO: Extract predicted and gold values
        # gold_output = gold.output_field
        # pred_output = pred.output_field

        # TODO: Calculate your metric (e.g., accuracy, F1, BLEU, etc.)
        # Example: score = calculate_score(gold.output_field, pred.output_field)
        score = 0.0  # Calculate actual score using the gold and pred fields

        # TODO: Generate helpful feedback based on the score
        if score < 0.5:
            feedback = "Low score. [Specific feedback about what went wrong]"
        elif score < 0.8:
            feedback = "Good but can improve. [Specific suggestions]"
        else:
            feedback = "Excellent! [What was done well]"

        return dspy.Prediction(score=score, feedback=feedback)

    except Exception as e:
        return dspy.Prediction(score=0.0, feedback=f"Error during evaluation: {e}")


def get_program():
    """Get the DSPy program to optimize"""
    return YourProgram()


def save_optimized_prompts(
    optimized_program, trainset, valset, task_model, reflection_model, optimization_mode
):
    """Save the optimized prompts.

    The save function handles metadata internally, so we just need
    to pass the prompt key and content. See lv0_s1.py and lv0_s3.py
    for working examples.

    TODO: Update this based on your program structure
    """
    # Extract the optimized prompt
    # TODO: Update based on your program structure
    optimized_prompt = optimized_program.process.signature.instructions

    saved_paths = []

    # TODO: Update the prompt key for your task (e.g., "lv0_s1_system", "lv0_s3_user")
    saved_path = save_prompt(
        prompt_key="YOUR_TASK",  # Change to your actual key
        prompt_content=optimized_prompt,
    )
    saved_paths.append(saved_path)

    # Also save the full optimized program if it has a save method
    # TODO: Update the filename
    program_path = Path("data/prompts") / "YOUR_TASK_program.json"
    if hasattr(optimized_program, "save") and callable(optimized_program.save):
        optimized_program.save(str(program_path))
        saved_paths.append(str(program_path))
    else:
        # If no save method, you may want to save the prompt string or other metadata
        # Comment out or implement alternative saving logic as needed
        pass

    return saved_paths


# Entry point for direct execution
if __name__ == "__main__":
    import os
    import dspy
    from dspy.teleprompt import GEPA

    # Load environment variables from .env file if it exists
    from dotenv import load_dotenv

    load_dotenv()

    # Configure OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    # gpt-5 models require temperature=1.0 and max_tokens >= 16000
    lm = dspy.LM(model="openai/gpt-5-nano", temperature=1.0, max_tokens=16000)
    dspy.settings.configure(lm=lm)

    # Create training data
    print("Creating training data...")
    training_data = create_training_data()

    # Prepare DSPy examples
    print("Preparing DSPy examples...")
    examples = prepare_dspy_examples(training_data)

    # Split into train/val
    split_idx = int(len(examples) * 0.8)
    trainset = examples[:split_idx]
    valset = examples[split_idx:]

    print(f"Training set size: {len(trainset)}")
    print(f"Validation set size: {len(valset)}")

    # Run optimization using GEPA directly
    print("Running GEPA optimization...")
    import os

    auto_level = os.getenv("GEPA_AUTO", "light")
    print(f"Using optimization level: {auto_level}")
    
    # Ensure auto_level is one of the valid options
    if auto_level not in ["light", "medium", "heavy"]:
        auto_level = "light"
    
    # Cast to proper Literal type for type checking
    auto_typed = cast(Literal["light", "medium", "heavy"], auto_level)

    # Configure with best practices
    optimizer = GEPA(
        metric=metric_with_feedback,  # type: ignore[arg-type]
        auto=auto_typed,
        num_threads=4,  # Parallel evaluation for speed
        reflection_minibatch_size=3,  # Good balance of reflection quality
        candidate_selection_strategy="pareto",  # Best strategy for diverse solutions
        skip_perfect_score=True,  # Don't waste time on perfect examples
        use_merge=True,  # Merge successful variants for better results
        max_merge_invocations=5,  # Reasonable merge attempts
        seed=42,  # Reproducibility
    )

    student = YourProgram()
    optimized_program = optimizer.compile(student, trainset=trainset, valset=valset)

    # Save optimized prompts
    print("Saving optimized prompts...")
    saved_paths = save_optimized_prompts(
        optimized_program,
        trainset=trainset,
        valset=valset,
        task_model="gpt-4o-mini",
        reflection_model="gpt-4",
        optimization_mode="gepa",
    )

    print(f"Optimization complete! Saved to: {saved_paths}")
