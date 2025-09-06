"""Optimizer for Level 0 Step 1: Concept extraction from college names

This module is called by the CLI: uv run optimize-prompt --prompt lv0_s1
"""

import json
import random
from pathlib import Path
from typing import List, Dict

import dspy
from dspy import Example
from pydantic import BaseModel, Field

from prompt_optimization.core import save_prompt


class ConceptExtraction(BaseModel):
    source: str = Field(description="Source text being processed")
    concepts: List[str] = Field(description="List of extracted concepts")


class ConceptExtractionList(BaseModel):
    extractions: List[ConceptExtraction] = Field(
        description="List of concept extractions", default_factory=list
    )


class ExtractConceptsSignature(dspy.Signature):
    """Extract academic concepts from college/school names.

    Extract the core academic disciplines and fields of study from the given college/school names.
    Focus on the academic subjects, not the institution type or location."""

    sources: str = dspy.InputField(
        desc="List of college/school names to extract concepts from"
    )
    extractions: ConceptExtractionList = dspy.OutputField(
        desc="List of extracted academic concepts for each source"
    )


class ConceptExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(ExtractConceptsSignature)

    def forward(self, sources):
        return self.extract(sources=sources)


def create_training_data(batch_size=5, val_batch_size=None, seed=None, return_split=False):
    """Load and batch training data to match production usage.

    Production processes 20 institutions per request.
    Default batch_size=5 for faster training.
    
    Args:
        batch_size: Size of training batches (default: 5)
        val_batch_size: Size of validation batches (default: same as batch_size)
        seed: Random seed for shuffling (default: None for non-deterministic)
        return_split: If True, return (train_examples, val_examples) tuple (default: False)
    
    Returns:
        If return_split=False: Combined list of all examples (shuffled)
        If return_split=True: Tuple of (train_examples, val_examples)
    """
    training_file = Path("data/prompts_training_data/lv0_s1.json")
    if not training_file.exists():
        raise FileNotFoundError(f"Training data not found at {training_file}")

    try:
        with open(training_file, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {training_file}: {e}")

    # Validate data structure
    if not data:
        raise ValueError(f"Empty training data in {training_file}")
    
    for item in data:
        if not isinstance(item.get("expected", []), list):
            raise ValueError(f"Invalid 'expected' field in training data: {item}")
    
    # Calculate split boundaries
    n_total = len(data)
    split_idx = int(n_total * 0.8)
    
    # Default validation batch size to training batch size
    if val_batch_size is None:
        val_batch_size = batch_size
    
    # Create training examples with proper boundary capping
    train_examples = []
    for i in range(0, split_idx, batch_size):
        end = min(i + batch_size, split_idx)  # Cap at split boundary
        batch = data[i:end]
        sources = "\n".join(f"- {item['input']}" for item in batch)
        extractions = [
            {"source": item["input"], "concepts": item["expected"]} for item in batch
        ]
        train_examples.append(
            {"input": {"sources": sources}, "output": {"extractions": extractions}}
        )
    
    # Create validation examples with batching support
    val_examples = []
    for i in range(split_idx, n_total, val_batch_size):
        end = min(i + val_batch_size, n_total)
        batch = data[i:end]
        
        if val_batch_size == 1:
            # Single-item validation for backward compatibility
            for item in batch:
                val_examples.append(
                    {
                        "input": {"sources": f"- {item['input']}"},
                        "output": {
                            "extractions": [
                                {"source": item["input"], "concepts": item["expected"]}
                            ]
                        },
                    }
                )
        else:
            # Batched validation
            sources = "\n".join(f"- {item['input']}" for item in batch)
            extractions = [
                {"source": item["input"], "concepts": item["expected"]} for item in batch
            ]
            val_examples.append(
                {"input": {"sources": sources}, "output": {"extractions": extractions}}
            )
    
    # Apply shuffling with seed if provided
    if seed is not None:
        rnd = random.Random(seed)
        rnd.shuffle(train_examples)
        rnd.shuffle(val_examples)
    else:
        random.shuffle(train_examples)
        random.shuffle(val_examples)
    
    if return_split:
        return train_examples, val_examples
    else:
        # Combine and shuffle for backward compatibility
        all_examples = train_examples + val_examples
        if seed is not None:
            rnd = random.Random(seed)
            rnd.shuffle(all_examples)
        else:
            random.shuffle(all_examples)
        return all_examples


def prepare_dspy_examples(training_data: List[Dict]) -> List[Example]:
    """Convert training data to DSPy Examples"""
    examples = []
    for item in training_data:
        example = Example(
            sources=item["input"]["sources"],
            extractions=ConceptExtractionList(
                extractions=[
                    ConceptExtraction(**ex) for ex in item["output"]["extractions"]
                ]
            ),
        ).with_inputs("sources")
        examples.append(example)
    return examples


def calculate_f1(gold_concepts, pred_concepts):
    """Calculate F1 score between two sets of concepts"""
    if not gold_concepts and not pred_concepts:
        return 1.0
    if not pred_concepts or not gold_concepts:
        return 0.0

    common = gold_concepts.intersection(pred_concepts)
    precision = len(common) / len(pred_concepts)
    recall = len(common) / len(gold_concepts)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Evaluate extraction quality with feedback for GEPA optimization."""
    
    def _norm(s: str) -> str:
        """Normalize concept strings for fair comparison"""
        return s.strip().lower()
    
    try:
        if hasattr(pred, "extractions"):
            if isinstance(pred.extractions, ConceptExtractionList):
                pred_extractions = pred.extractions
            else:
                pred_extractions = ConceptExtractionList(extractions=[])
        else:
            pred_extractions = ConceptExtractionList(extractions=[])

        gold_extractions = gold.extractions

        # Normalize concepts in prediction dictionary
        pred_dict = {
            ex.source: {_norm(c) for c in ex.concepts} 
            for ex in pred_extractions.extractions
        }

        f1_scores = []
        issues = []

        for gold_ex in gold_extractions.extractions:
            source = gold_ex.source
            # Normalize gold concepts
            gold_concepts = {_norm(c) for c in gold_ex.concepts}
            pred_concepts = pred_dict.get(source, set())

            f1 = calculate_f1(gold_concepts, pred_concepts)
            f1_scores.append(f1)

            if f1 < 1.0:
                if not pred_concepts and gold_concepts:
                    issues.append(f"Missed: {source}")
                elif pred_concepts and not gold_concepts:
                    issues.append(f"Over-extracted: {source}")
                else:
                    missing = gold_concepts - pred_concepts
                    extra = pred_concepts - gold_concepts
                    if missing:
                        # Show original case for readability in feedback
                        missing_original = [c for c in gold_ex.concepts if _norm(c) in missing]
                        issues.append(f"{source} missing: {','.join(missing_original)}")
                    if extra:
                        # Find original case for extra concepts
                        pred_ex = next((ex for ex in pred_extractions.extractions if ex.source == source), None)
                        if pred_ex:
                            extra_original = [c for c in pred_ex.concepts if _norm(c) in extra]
                            issues.append(f"{source} extra: {','.join(extra_original)}")

        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        if avg_f1 == 1.0:
            feedback = "Perfect extraction!"
        else:
            feedback = f"F1: {avg_f1:.2f}. "
            if issues:
                feedback += " ".join(issues[:3])
                if len(issues) > 3:
                    feedback += f" (+{len(issues)-3} more)"

        return dspy.Prediction(score=avg_f1, feedback=feedback)

    except Exception as e:
        return dspy.Prediction(score=0.0, feedback=f"Error: {e}")


def get_program():
    """Get the DSPy program to optimize"""
    return ConceptExtractor()


def save_optimized_prompts(
    optimized_program, trainset, valset, task_model, reflection_model, optimization_mode
):
    """Save optimized prompts and program state"""
    saved_paths = []

    system_prompt = ""
    if hasattr(optimized_program, "extract"):
        predictor = optimized_program.extract
        if hasattr(predictor, "extended_signature"):
            system_prompt = str(predictor.extended_signature)
        elif hasattr(predictor, "signature"):
            system_prompt = str(predictor.signature)
        else:
            system_prompt = str(predictor)

    metadata = {
        "train_size": len(trainset),
        "val_size": len(valset),
        "task_model": task_model,
        "reflection_model": reflection_model,
        "optimization_mode": optimization_mode,
        "final_score": getattr(optimized_program, "score", None),
    }

    saved_paths.append(
        save_prompt(
            prompt_key="lv0_s1_system",
            prompt_content=system_prompt,
            metadata=metadata,
        )
    )
    user_template = """Extract academic concepts from these institutions:

{sources}

For each institution, identify the core academic fields and disciplines."""

    saved_paths.append(
        save_prompt(
            prompt_key="lv0_s1_user",
            prompt_content=user_template,
            metadata=metadata,
        )
    )
    program_path = Path("data/prompts") / "lv0_s1_program.json"
    program_path.parent.mkdir(parents=True, exist_ok=True)
    optimized_program.save(str(program_path))
    saved_paths.append(program_path)

    return saved_paths


def load_optimized_program(program_path: Path = None):
    """Load previously optimized program from disk"""
    if program_path is None:
        program_path = Path("data/prompts") / "lv0_s1_program.json"

    if not program_path.exists():
        raise FileNotFoundError(f"No saved program found at {program_path}")

    program = ConceptExtractor()
    program.load(str(program_path))
    return program

