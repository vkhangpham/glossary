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

import json
from pathlib import Path
from typing import List, Dict

import dspy
from dspy import Example
from pydantic import BaseModel, Field

from prompt_optimization.save import save_prompt


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
    
    input_field: str = dspy.InputField(
        desc="Description of the input"
    )
    output_field: YourOutputModel = dspy.OutputField(
        desc="Description of the output"
    )


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
            "input": {
                "input_field": "Example input 1"
            },
            "output": {
                "output_field": {
                    "field1": "Expected output 1",
                    "field2": ["item1", "item2"]
                }
            }
        },
        # Add more examples...
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
            output_field=YourOutputModel(**item["output"]["output_field"])
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
        gold_output = gold.output_field
        pred_output = pred.output_field
        
        # TODO: Calculate your metric (e.g., accuracy, F1, BLEU, etc.)
        score = 0.0  # Calculate actual score
        
        # TODO: Generate helpful feedback based on the score
        if score < 0.5:
            feedback = "Low score. [Specific feedback about what went wrong]"
        elif score < 0.8:
            feedback = "Good but can improve. [Specific suggestions]"
        else:
            feedback = "Excellent! [What was done well]"
        
        return dspy.Prediction(score=score, feedback=feedback)
        
    except Exception as e:
        return dspy.Prediction(
            score=0.0,
            feedback=f"Error during evaluation: {e}"
        )


def get_program():
    """Get the DSPy program to optimize"""
    return YourProgram()


def save_optimized_prompts(optimized_program, trainset, valset, 
                          task_model, reflection_model, optimization_mode):
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
        prompt_content=optimized_prompt
    )
    saved_paths.append(saved_path)
    
    # Also save the full optimized program if it has a save method
    # TODO: Update the filename
    program_path = Path("data/prompts") / "YOUR_TASK_program.json"
    if hasattr(optimized_program, 'save') and callable(optimized_program.save):
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
    from prompt_optimization.core import run_optimization
    
    # Configure OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    # Setup DSPy with OpenAI
    lm = dspy.OpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=2000
    )
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
    
    # Run optimization
    print("Running GEPA optimization...")
    optimized_program = run_optimization(
        program=YourProgram(),
        trainset=trainset,
        valset=valset,
        metric=metric_with_feedback,
        task_model="gpt-4o-mini",
        optimize_kwargs={
            "verbose": True,
            "requires_permission_to_run": False,
            "max_iterations": 10
        }
    )
    
    # Save optimized prompts
    print("Saving optimized prompts...")
    saved_paths = save_optimized_prompts(
        optimized_program,
        trainset=trainset,
        valset=valset,
        task_model="gpt-4o-mini",
        reflection_model="gpt-4",
        optimization_mode="gepa"
    )
    
    print(f"Optimization complete! Saved to: {saved_paths}")