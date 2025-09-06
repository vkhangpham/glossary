"""Template for creating new prompt optimizers

Copy this file and modify for your specific prompt optimization needs.
Naming convention: {level}_s{step}.py (e.g., lv0_s1.py, lv1_s2.py)

Usage: uv run optimize-prompt --prompt {level}_s{step}
"""

import json
from pathlib import Path
from typing import List, Dict

import dspy
from dspy import Example
from pydantic import BaseModel, Field

from prompt_optimization.core import save_prompt


# TODO: Define your Pydantic models matching the task
class YourOutputModel(BaseModel):
    """Define the expected output structure"""
    field1: str = Field(description="Description of field1")
    field2: List[str] = Field(description="Description of field2")


# TODO: Define the DSPy signature for your task
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
    """Get the DSPy program to optimize (called by CLI)"""
    return YourProgram()


def save_optimized_prompts(optimized_program, trainset, valset, 
                          task_model, reflection_model, optimization_mode):
    """Save the optimized prompts (called by CLI)
    
    TODO: Update this based on your program structure
    """
    # Extract the optimized prompt
    # TODO: Update based on your program structure
    optimized_prompt = optimized_program.process.signature.instructions
    
    # Save the optimized prompt
    metadata = {
        "train_size": len(trainset),
        "val_size": len(valset),
        "task_model": task_model,
        "reflection_model": reflection_model,
        "optimization_mode": optimization_mode,
        "final_score": getattr(optimized_program, "score", None)
    }
    
    saved_paths = []
    
    # TODO: Update the prompt key for your task
    saved_path = save_prompt(
        prompt_key="YOUR_TASK",
        prompt_content=optimized_prompt,
        metadata=metadata
    )
    saved_paths.append(saved_path)
    
    # Also save the full optimized program
    # TODO: Update the filename
    program_path = Path("data/prompts") / "YOUR_TASK_program.json"
    optimized_program.save(str(program_path))
    saved_paths.append(program_path)
    
    return saved_paths