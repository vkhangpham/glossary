"""Template for creating new prompt optimizers with DSPy 2024-2025 signature metadata support.

This template demonstrates modern prompt optimization with signature metadata extraction,
enabling declarative DSPy programming and improved prompt integration.

Key Features:
- ✅ Signature metadata extraction from GEPA optimization results
- ✅ DSPy 2024-2025 best practices compliance validation
- ✅ Enhanced prompt saving with declarative programming support
- ✅ Modular composition patterns with ChainOfThought/TypedPredictor
- ✅ Metric-driven development with comprehensive feedback

Copy this file and modify for your specific prompt optimization needs.
Naming convention: {level}_s{step}.py (e.g., lv0_s1.py, lv1_s2.py)

For examples of working optimizers with signature metadata, see:
- lv0_s1.py: Concept extraction with signature metadata
- lv0_s3.py: Token verification with signature metadata

Usage:
- Direct execution: python prompt_optimization/optimizers/{level}_s{step}.py
- Runner script: python run_{level}_s{step}_optimization.py

Best Practices for DSPy 2024-2025:
1. Extract signature metadata from optimized programs
2. Use run_optimization() for comprehensive automation
3. Validate DSPy compliance after optimization
4. Prefer declarative programming over text-based approaches
5. Enable modular composition with proper predictor types
"""

from pathlib import Path
from typing import Dict, List, Literal, cast

import dspy
from dspy import Example
from pydantic import BaseModel, Field

from prompt_optimization.core import save_prompt
from prompt_optimization.optimizers.common import run_optimization

# Default task identifier for template examples
DEFAULT_TASK_ID = "example_optimization"

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


def extract_signature_metadata_example(optimized_program):
    """
    Example of how to extract signature metadata from GEPA optimization results.
    
    This function demonstrates the signature metadata extraction that should be
    implemented in your specific optimizer. The extracted metadata enables
    declarative DSPy programming with precise signature information.
    
    TODO: Adapt this example to your specific DSPy program structure.
    """
    try:
        # Access the optimized signature from your program
        # Common patterns: optimized_program.process.signature, optimized_program.predictor.signature
        if hasattr(optimized_program, 'process') and hasattr(optimized_program.process, 'signature'):
            signature = optimized_program.process.signature
        else:
            # Adapt based on your program structure
            return None
        
        # Extract field information
        input_fields = {}
        output_fields = {}
        
        # Input fields
        if hasattr(signature, 'input_fields'):
            for field_name, field in signature.input_fields.items():
                input_fields[field_name] = getattr(field, 'desc', f"Input field: {field_name}")
        
        # Output fields  
        if hasattr(signature, 'output_fields'):
            for field_name, field in signature.output_fields.items():
                output_fields[field_name] = getattr(field, 'desc', f"Output field: {field_name}")
        
        # Extract instructions
        instructions = getattr(signature, 'instructions', "")
        
        # Build signature string
        signature_str = str(signature) if hasattr(signature, '__str__') else ""
        
        # Determine predictor type based on program structure
        predictor_type = "Predict"  # Default
        if hasattr(optimized_program, 'process'):
            if "ChainOfThought" in str(type(optimized_program.process)):
                predictor_type = "ChainOfThought"
            elif "TypedPredictor" in str(type(optimized_program.process)):
                predictor_type = "TypedPredictor"
        
        # Build signature metadata dictionary (matching common.py format)
        signature_metadata = {
            "input_fields": input_fields,
            "output_fields": output_fields,
            "instructions": instructions,
            "signature_str": signature_str,
            "predictor_type": predictor_type
        }
        
        return signature_metadata
        
    except Exception as e:
        print(f"Warning: Failed to extract signature metadata: {e}")
        return None


def save_optimized_prompts_with_metadata(optimized_program, program_name: str, task_id: str = None):
    """
    Enhanced prompt saving with signature metadata extraction.
    
    This demonstrates the modern approach using signature metadata for
    DSPy 2024-2025 compliance. The common.run_optimization function
    handles the metadata extraction and saving automatically.
    
    Args:
        optimized_program: The optimized DSPy program
        program_name: Name of the program
        task_id: Task identifier for prompt keys (defaults to program_name)
    """
    if not task_id:
        task_id = program_name
    
    # Validate task_id
    if not task_id or not task_id.strip():
        raise ValueError("task_id cannot be empty")
    
    # Ensure task_id contains only allowed characters (alphanumeric, underscore, hyphen)
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', task_id):
        raise ValueError("task_id must contain only alphanumeric characters, underscores, and hyphens")
    # Extract signature metadata (example - adapt to your program)
    signature_metadata = extract_signature_metadata_example(optimized_program)
    
    # Method 1: Use run_optimization for comprehensive handling (RECOMMENDED)
    # This is the modern approach that handles everything automatically
    print("Using run_optimization for comprehensive prompt handling...")
    # Note: This would typically be called from main() with all required parameters
    
    # Method 2: Manual saving with signature metadata (for custom scenarios)
    print("Demonstrating manual signature metadata saving...")
    
    try:
        # Extract optimized prompt content
        optimized_prompt = ""
        if hasattr(optimized_program, 'process') and hasattr(optimized_program.process, 'signature'):
            optimized_prompt = getattr(optimized_program.process.signature, 'instructions', "")
        
        if not optimized_prompt:
            print("Warning: Could not extract optimized prompt content")
            return []
        
        saved_paths = []
        
        # Save system prompt with signature metadata
        system_path = save_prompt(
            prompt_key=f"{task_id}_system",
            prompt_content="Process the input according to the optimized instructions.",
            signature_metadata=signature_metadata
        )
        saved_paths.append(system_path)
        
        # Save user prompt with signature metadata  
        user_path = save_prompt(
            prompt_key=f"{task_id}_user",
            prompt_content=optimized_prompt,
            signature_metadata=signature_metadata
        )
        saved_paths.append(user_path)
        
        return saved_paths
        
    except Exception as e:
        print(f"Error in manual prompt saving: {e}")
        return []


def validate_dspy_compliance_example(program_name: str = "example_task"):
    """
    Example of validating DSPy 2024-2025 best practices compliance.
    
    This demonstrates how to check if your optimization results align
    with modern DSPy principles for declarative programming.
    
    Args:
        program_name: The program/use case name to validate (defaults to "example_task")
    """
    try:
        # Import validation function
        from generate_glossary.llm.signatures import validate_dspy_compliance
        
        # Validate the optimized prompts for DSPy compliance
        use_case = program_name or "example_task"
        
        compliance_result = validate_dspy_compliance(use_case)
        
        print(f"DSPy Compliance Score: {compliance_result['overall_score']:.2f}")
        print("\nCompliance Checks:")
        for check, passed in compliance_result['compliance_checks'].items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
        
        print("\nRecommendations:")
        for rec in compliance_result['recommendations']:
            print(f"  • {rec}")
        
        if compliance_result['improvement_priority']:
            print("\nPriority Improvements:")
            for priority in compliance_result['improvement_priority']:
                print(f"  🔧 {priority}")
        
        return compliance_result
        
    except ImportError:
        print("DSPy compliance validation not available - install enhanced LLM utils")
        return None
    except Exception as e:
        print(f"Compliance validation failed: {e}")
        return None


if __name__ == "__main__":
    import os
    import dspy
    from dspy.teleprompt import GEPA

    from dotenv import load_dotenv

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    # gpt-5 models require temperature=1.0 and max_tokens >= 16000
    lm = dspy.LM(model="openai/gpt-5-nano", temperature=1.0, max_tokens=16000)
    dspy.settings.configure(lm=lm)

    print("Creating training data...")
    training_data = create_training_data()

    print("Preparing DSPy examples...")
    examples = prepare_dspy_examples(training_data)

    split_idx = int(len(examples) * 0.8)
    trainset = examples[:split_idx]
    valset = examples[split_idx:]

    print(f"Training set size: {len(trainset)}")
    print(f"Validation set size: {len(valset)}")

    print("Running GEPA optimization...")
    import os

    auto_level = os.getenv("GEPA_AUTO", "light")
    print(f"Using optimization level: {auto_level}")

    if auto_level not in ["light", "medium", "heavy"]:
        auto_level = "light"

    auto_typed = cast(Literal["light", "medium", "heavy"], auto_level)

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

    print("Saving optimized prompts with signature metadata...")
    
    # Method 1: Use enhanced common.run_optimization (RECOMMENDED)
    # This handles signature metadata extraction automatically
    try:
        from prompt_optimization.optimizers.common import run_optimization
        
        # Example of using run_optimization for comprehensive handling
        # TODO: Implement this by calling run_optimization with your parameters
        print("For production use, call run_optimization() instead of manual GEPA")
        print("See lv0_s1.py and lv0_s3.py for working examples")
        
    except ImportError:
        print("Enhanced optimization not available - using manual approach")
    
    # Method 2: Manual approach with signature metadata (demonstration)
    saved_paths = save_optimized_prompts_with_metadata(optimized_program, DEFAULT_TASK_ID)
    
    print(f"Prompts saved to: {saved_paths}")
    
    # Demonstrate DSPy compliance validation
    print("\nValidating DSPy 2024-2025 compliance...")
    compliance_result = validate_dspy_compliance_example(DEFAULT_TASK_ID)
    
    if compliance_result:
        if compliance_result['overall_score'] >= 0.8:
            print("🎉 Excellent DSPy compliance!")
        elif compliance_result['overall_score'] >= 0.6:
            print("👍 Good DSPy compliance with improvement opportunities")
        else:
            print("🔧 Consider implementing the recommended improvements")
    
    print(f"Optimization complete!")
