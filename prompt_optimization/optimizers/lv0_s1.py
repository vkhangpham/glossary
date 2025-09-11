"""Enhanced optimizer for lv0_s1 concept extraction with DSPy 2024-2025 signature metadata support.

This optimizer demonstrates modern prompt optimization with automatic signature metadata
extraction, enabling declarative DSPy programming and improved runtime integration.

Features:
- ✅ GEPA optimization with signature metadata extraction
- ✅ DSPy ChainOfThought predictor for enhanced reasoning
- ✅ Comprehensive metric-driven development with detailed feedback
- ✅ Automatic compliance with DSPy 2024-2025 best practices
- ✅ Enhanced prompt artifacts with declarative programming support

Environment Variables:
    GEPA_GEN_MODEL: Generation model (default: gpt-4o-mini)
    GEPA_REFLECTION_MODEL: Reflection model (default: same as GEPA_GEN_MODEL)
"""

import json
import logging
import sys
from typing import Any, Dict, List

import dspy
from dotenv import load_dotenv
from prompt_optimization.optimizers.common import (
    load_json_training,
    run_optimization,
)

load_dotenv()

logger = logging.getLogger(__name__)


TRAINING_DATA_PATH = "data/prompts_training_data/lv0_s1.json"


class ExtractConceptsSignature(dspy.Signature):
    """You are an expert at extracting academic concepts from institutional names.

Extract the core academic disciplines and fields of study from college/school names following these rules:
1. Extract only academic subjects (e.g., 'engineering', 'medicine'), not institution types ('college', 'school')
2. Split conjunctions: 'Arts and Sciences' → ['arts', 'sciences']
3. Remove person names: 'Gerald R. Ford School of Public Policy' → ['public policy']
4. Keep multi-word disciplines together: 'computer science' (not 'computer', 'science')
5. Return empty list [] for generic units like 'Graduate School', 'Honors College'
6. Always normalize to lowercase"""

    text = dspy.InputField(desc="Text to extract concepts from", prefix="Input Text:")
    extraction = dspy.OutputField(
        desc="JSON array of extraction objects, each with 'source' and 'concepts' fields",
        prefix="Extraction (JSON):",
    )


class ConceptExtractor(dspy.Module):
    """DSPy module for extracting academic concepts."""

    def __init__(self):
        super().__init__()
        # Use ChainOfThought for better reasoning and optimization
        self.prog = dspy.ChainOfThought(ExtractConceptsSignature)

    def forward(self, text: str) -> dspy.Prediction:
        """Extract concepts from text."""
        return self.prog(text=text)


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

        formatted_output = json.dumps(expected_extractions, indent=2)

        example = dspy.Example(
            text=text,
            extraction=formatted_output,
        ).with_inputs("text")

        examples.append(example)

    return examples


def metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Metric function for GEPA that evaluates concept extraction quality.
    Returns dspy.Prediction with score and feedback.
    """
    try:

        pred_extraction = json.loads(pred.extraction)


        if isinstance(gold.extraction, str):
            gold_extractions = json.loads(gold.extraction)
        else:
            gold_extractions = gold.extraction



        if isinstance(pred_extraction, list):
            pred_concepts = set()
            for ext in pred_extraction:
                pred_concepts.update(ext.get("concepts", []))
        else:
            pred_concepts = set(pred_extraction.get("concepts", []))

        gold_concepts = set()
        for ext in gold_extractions:
            gold_concepts.update(ext.get("concepts", []))


        if not gold_concepts:
            # Empty gold concepts means NO concepts should be extracted
            if not pred_concepts:
                # Both empty - perfect match!
                return dspy.Prediction(
                    score=1.0, feedback="Correctly identified no extractable concepts"
                )
            else:
                # Gold is empty but pred extracted something - wrong!
                return dspy.Prediction(
                    score=0.0,
                    feedback=f"Incorrectly extracted {len(pred_concepts)} concepts when none were expected",
                )

        matches = pred_concepts & gold_concepts
        precision = len(matches) / len(pred_concepts) if pred_concepts else 0
        recall = len(matches) / len(gold_concepts) if gold_concepts else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Generate detailed, pattern-based feedback
        feedback = []
        
        # Analyze specific extraction patterns
        missing = gold_concepts - pred_concepts if gold_concepts else set()
        extra = pred_concepts - gold_concepts if pred_concepts else set()
        correct = gold_concepts & pred_concepts if gold_concepts and pred_concepts else set()
        
        # Score-based feedback tiers
        if f1 == 1.0:
            feedback.append("Perfect extraction! All academic concepts correctly identified.")
        elif f1 >= 0.8:
            feedback.append(f"Good extraction (F1: {f1:.2f}).")
            if missing:
                # Check for common patterns in what was missed
                for concept in list(missing)[:2]:
                    if " and " in gold.text:
                        feedback.append(f"Missed '{concept}' - remember to split 'and' conjunctions.")
                    elif len(concept.split()) > 1:
                        feedback.append(f"Missed multi-word concept '{concept}'.")
                    else:
                        feedback.append(f"Missed concept: '{concept}'.")
        elif f1 >= 0.5:
            feedback.append(f"Moderate extraction quality (F1: {f1:.2f}).")
            
            # Provide pattern-specific guidance
            if "and" in gold.text.lower() and recall < 0.8:
                feedback.append("Remember to split 'and' conjunctions (e.g., 'Arts and Sciences' → ['arts', 'sciences']).")
            
            if extra:
                # Check for common incorrect patterns
                for concept in list(extra)[:2]:
                    if concept in ["college", "school", "department", "institute"]:
                        feedback.append(f"Don't extract institution types like '{concept}' - focus on academic disciplines only.")
                    elif concept[0].isupper():
                        feedback.append(f"'{concept}' should be lowercase - normalize all concepts.")
                    else:
                        feedback.append(f"Incorrectly extracted: '{concept}'.")
        else:
            feedback.append(f"Poor extraction quality (F1: {f1:.2f}). Let's analyze the patterns:")
            
            # Provide specific examples for poor performance
            if gold.text:
                # Check for specific patterns in the input
                text_lower = gold.text.lower()
                
                if "graduate school" in text_lower or "honors college" in text_lower:
                    feedback.append("Generic administrative units like 'Graduate School' should return empty list [].")
                elif " and " in text_lower:
                    feedback.append("Split conjunctions: 'X and Y' should extract both 'x' and 'y' as separate concepts.")
                elif any(name in gold.text for name in ["Gerald R. Ford", "Stephen M. Ross", "David Geffen"]):
                    feedback.append("Remove person names from extractions - only keep the academic discipline.")
                
                # Show what was expected vs what was extracted
                if gold_concepts:
                    feedback.append(f"Expected: {sorted(list(gold_concepts)[:3])}.")
                    if pred_concepts:
                        feedback.append(f"Got: {sorted(list(pred_concepts)[:3])}.")

        if not isinstance(pred_extraction, (list, dict)):
            feedback.append("Output must be valid JSON array with 'source' and 'concepts' fields.")

        feedback_str = " ".join(feedback)

        return dspy.Prediction(score=f1, feedback=feedback_str)

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return dspy.Prediction(
            score=0.0,
            feedback=f"Failed to parse extraction: {str(e)}. Ensure valid JSON output.",
        )




# Default prompts - aligned with actual implementation in lv0_s1_extract_concepts.py
DEFAULT_S1_SYSTEM = """You are an expert at extracting academic concepts from institutional names.
Extract the core academic disciplines and fields of study from the given college/school names.
Focus on the academic subjects, not the institution type or location."""

DEFAULT_S1_USER = """Extract academic concepts from these institutions:

{sources}

For each institution, identify the core academic fields and disciplines."""


def optimize_prompts():
    """
    Run GEPA optimization for concept extraction prompts with signature metadata extraction.
    
    This function uses the enhanced run_optimization() which automatically:
    1. Extracts signature metadata from the GEPA optimization results
    2. Saves prompts with DSPy 2024-2025 compliant signature metadata
    3. Enables declarative programming for runtime integration
    4. Supports ChainOfThought predictor type recognition
    5. Provides metric-driven development with comprehensive feedback
    
    The ExtractConceptsSignature will be analyzed to extract:
    - Input fields: text (description: "Text to extract concepts from")
    - Output fields: extraction (description: "JSON array of extraction objects...")  
    - Instructions: Combined system + user prompt instructions
    - Predictor type: ChainOfThought (detected from ConceptExtractor.prog)
    """
    return run_optimization(
        program_name="lv0_s1_concept_extraction",
        training_data_path=TRAINING_DATA_PATH,
        dspy_module=ConceptExtractor,
        metric_func=metric_with_feedback,
        prepare_examples_func=prepare_dspy_examples,
        default_system_prompt=DEFAULT_S1_SYSTEM,
        default_user_prompt=DEFAULT_S1_USER,
        create_training_data_func=create_training_data,
    )


if __name__ == "__main__":
    """
    Enhanced lv0_s1 concept extraction optimization with DSPy 2024-2025 features.
    
    This will generate optimized prompts with signature metadata at:
    - data/prompts/lv0_s1_system_latest.json (with signature_metadata field)
    - data/prompts/lv0_s1_user_latest.json (with signature_metadata field)
    
    The signature metadata enables:
    - Declarative DSPy programming at runtime
    - Automatic predictor type selection (ChainOfThought)
    - Precise field mapping without text inference
    - Enhanced modular composition capabilities
    
    To validate DSPy compliance after optimization:
    > from generate_glossary.llm.signatures import validate_dspy_compliance
    > result = validate_dspy_compliance("lv0_s1")
    > print(f"DSPy compliance score: {result['overall_score']:.2f}")
    """
    try:
        print("🚀 Starting enhanced lv0_s1 optimization with signature metadata extraction...")
        optimize_prompts()
        print("✅ Optimization complete! Prompts saved with DSPy 2024-2025 signature metadata.")
        print("📋 Generated files include declarative programming metadata for runtime integration.")
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        sys.exit(1)
