"""Optimizer for Level 0 Step 3: Single-word academic discipline verification

This module is called by the CLI: uv run optimize-prompt --prompt lv0_s3
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

import dspy
from dspy import Example
from pydantic import BaseModel, Field

from prompt_optimization.core import save_prompt


class VerificationResult(BaseModel):
    """Result of academic discipline verification"""
    is_valid: bool = Field(
        description="Whether the term is a valid broad academic discipline"
    )


class AcademicDisciplineVerification(dspy.Signature):
    """Verify if a term represents a valid broad academic discipline.
    
    Analyze whether the keyword is a legitimate broad academic discipline by considering:
    1. Academic relevance - Is it a recognized field of study or broad academic discipline?
    2. Disciplinary context - Does it represent a major division of knowledge in academia?
    3. Scope - Is it broad enough to encompass multiple research areas or subdisciplines?
    
    Accept broad disciplines like humanities, sciences, engineering, arts, medicine, law.
    Reject narrow specializations, technical methodologies, specific research topics."""
    
    keyword: str = dspy.InputField(
        desc="Single word term to verify as academic discipline"
    )
    evidence_colleges: str = dspy.InputField(
        desc="List of colleges/schools/departments that mention this term"
    )
    reasoning: str = dspy.OutputField(
        desc="Step-by-step analysis considering academic relevance, disciplinary context, and scope"
    )
    is_valid: VerificationResult = dspy.OutputField(
        desc="Boolean decision: true if valid broad academic discipline, false otherwise"
    )


class DisciplineVerifier(dspy.Module):
    """Module for verifying academic disciplines using ChainOfThought reasoning"""
    
    def __init__(self):
        super().__init__()
        self.verify = dspy.ChainOfThought(AcademicDisciplineVerification)
    
    def forward(self, keyword, evidence_colleges):
        return self.verify(keyword=keyword, evidence_colleges=evidence_colleges)


def create_training_data() -> List[Dict]:
    """Load and prepare training data for discipline verification.
    
    Returns training data in the format expected by the optimization CLI.
    """
    training_file = Path("data/prompts_training_data/lv0_s3.json")
    if not training_file.exists():
        raise FileNotFoundError(f"Training data not found at {training_file}")
    
    with open(training_file, "r") as f:
        raw_data = json.load(f)
    
    # Convert to the expected format for DSPy optimization
    training_data = []
    for item in raw_data:
        # Format evidence colleges as a string
        evidence_str = "\n".join(f"- {college}" for college in item["evidence_colleges"])
        
        training_data.append({
            "input": {
                "keyword": item["keyword"],
                "evidence_colleges": evidence_str
            },
            "output": {
                "is_valid": item["expected"]
            }
        })
    
    # Shuffle for better training
    random.shuffle(training_data)
    return training_data


def prepare_dspy_examples(training_data: List[Dict]) -> List[Example]:
    """Convert training data to DSPy Examples with proper formatting"""
    examples = []
    
    for item in training_data:
        # Extract input and output from the standardized format
        input_data = item["input"]
        output_data = item["output"]
        
        # Create the example with proper structure
        example = Example(
            keyword=input_data["keyword"],
            evidence_colleges=input_data["evidence_colleges"],
            is_valid=VerificationResult(is_valid=output_data["is_valid"])
        ).with_inputs("keyword", "evidence_colleges")
        
        examples.append(example)
    
    return examples


def calculate_metrics(gold_labels: List[bool], pred_labels: List[bool]) -> Dict[str, float]:
    """Calculate binary classification metrics"""
    if not gold_labels or not pred_labels:
        return {"mcc": 0.0, "balanced_accuracy": 0.0}
    
    # Matthews Correlation Coefficient - best for binary classification
    mcc = matthews_corrcoef(gold_labels, pred_labels)
    
    # Balanced accuracy - handles class imbalance
    balanced_acc = balanced_accuracy_score(gold_labels, pred_labels)
    
    return {
        "mcc": mcc,
        "balanced_accuracy": balanced_acc
    }


def metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Evaluate verification quality with feedback for GEPA optimization.
    
    Uses Matthews Correlation Coefficient (MCC) as primary metric for binary classification.
    MCC ranges from -1 to 1: 1 = perfect, 0 = random, -1 = perfectly wrong.
    """
    try:
        # Extract the boolean values
        gold_value = gold.is_valid.is_valid if hasattr(gold.is_valid, 'is_valid') else gold.is_valid
        
        # Handle prediction extraction
        if hasattr(pred, 'is_valid'):
            if isinstance(pred.is_valid, VerificationResult):
                pred_value = pred.is_valid.is_valid
            elif isinstance(pred.is_valid, bool):
                pred_value = pred.is_valid
            else:
                pred_value = False
        else:
            pred_value = False
        
        # Calculate score: 1.0 for correct, 0.0 for incorrect
        score = 1.0 if gold_value == pred_value else 0.0
        
        # Generate feedback
        if score == 1.0:
            feedback = f"Correct: '{gold.keyword}' -> {pred_value}"
        else:
            feedback = f"Wrong: '{gold.keyword}' expected {gold_value}, got {pred_value}"
            
            # Add specific feedback for common errors
            if gold_value and not pred_value:
                feedback += " (false negative - missed valid discipline)"
            elif not gold_value and pred_value:
                feedback += " (false positive - accepted invalid specialization)"
        
        return dspy.Prediction(score=score, feedback=feedback)
        
    except Exception as e:
        return dspy.Prediction(score=0.0, feedback=f"Error evaluating: {e}")


def get_program():
    """Get the DSPy program to optimize"""
    return DisciplineVerifier()


def save_optimized_prompts(
    optimized_program, trainset, valset, task_model, reflection_model, optimization_mode
):
    """Save optimized prompts and program state"""
    saved_paths = []
    
    # Extract the optimized prompt from the program
    system_prompt = ""
    if hasattr(optimized_program, "verify"):
        predictor = optimized_program.verify
        if hasattr(predictor, "extended_signature"):
            system_prompt = str(predictor.extended_signature)
        elif hasattr(predictor, "signature"):
            system_prompt = str(predictor.signature)
        else:
            system_prompt = str(predictor)
    
    # Calculate final metrics on validation set
    gold_labels = []
    pred_labels = []
    
    for example in valset:
        try:
            prediction = optimized_program(
                keyword=example.keyword,
                evidence_colleges=example.evidence_colleges
            )
            
            gold_value = example.is_valid.is_valid if hasattr(example.is_valid, 'is_valid') else example.is_valid
            pred_value = prediction.is_valid.is_valid if hasattr(prediction.is_valid, 'is_valid') else prediction.is_valid
            
            gold_labels.append(gold_value)
            pred_labels.append(pred_value)
        except:
            continue
    
    metrics = calculate_metrics(gold_labels, pred_labels) if gold_labels else {"mcc": 0.0, "balanced_accuracy": 0.0}
    
    # Save metadata
    metadata = {
        "train_size": len(trainset),
        "val_size": len(valset),
        "task_model": task_model,
        "reflection_model": reflection_model,
        "optimization_mode": optimization_mode,
        "final_score": getattr(optimized_program, "score", None),
        "validation_metrics": metrics
    }
    
    # Save system prompt
    saved_paths.append(
        save_prompt(
            prompt_key="lv0_s3_system",
            prompt_content=system_prompt,
            metadata=metadata,
        )
    )
    
    # Save user template for verification
    user_template = """Analyze whether "{keyword}" is a valid broad academic discipline.

Evidence - Colleges/schools/divisions that mention this concept:
{evidence_colleges}

Consider:
1. Is it a recognized major field of study or broad academic discipline?
2. Does it represent a major division of knowledge in academia?
3. Is it broad enough to encompass multiple research areas or subdisciplines?

Answer with true if it meets these criteria, false otherwise."""
    
    saved_paths.append(
        save_prompt(
            prompt_key="lv0_s3_user",
            prompt_content=user_template,
            metadata=metadata,
        )
    )
    
    # Save the program state
    program_path = Path("data/prompts") / "lv0_s3_program.json"
    program_path.parent.mkdir(parents=True, exist_ok=True)
    optimized_program.save(str(program_path))
    saved_paths.append(program_path)
    
    return saved_paths


def load_optimized_program(program_path: Path = None):
    """Load previously optimized program from disk"""
    if program_path is None:
        program_path = Path("data/prompts") / "lv0_s3_program.json"
    
    if not program_path.exists():
        raise FileNotFoundError(f"No saved program found at {program_path}")
    
    program = DisciplineVerifier()
    program.load(str(program_path))
    return program