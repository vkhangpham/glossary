"""Simplified standalone optimizer for lv0_s3 discipline verification using DSPy GEPA.

Environment Variables:
    GEPA_GEN_MODEL: Generation model (default: gpt-4o-mini)
    GEPA_REFLECTION_MODEL: Reflection model (default: same as GEPA_GEN_MODEL)
"""

import logging
import sys
from typing import Any, Dict, List

import dspy
from dotenv import load_dotenv
from prompt_optimization.optimizers.common import (
    load_json_training,
    run_optimization,
)

logger = logging.getLogger(__name__)

load_dotenv()


TRAINING_DATA_PATH = "data/prompts_training_data/lv0_s3.json"


class AcademicDisciplineVerification(dspy.Signature):
    """You are an expert at verifying academic disciplines.
Determine if the given term represents a legitimate academic discipline or field of study.
Consider whether it's taught at universities, has research communities, or represents a recognized field of knowledge."""

    term = dspy.InputField(desc="Term to verify as academic discipline")
    is_discipline = dspy.OutputField(
        desc="Boolean indicating if term is an academic discipline"
    )
    reasoning = dspy.OutputField(desc="Explanation for the decision")


class DisciplineVerifier(dspy.Module):
    """DSPy module for verifying academic disciplines."""

    def __init__(self):
        super().__init__()
        # Use ChainOfThought for better reasoning and optimization
        self.prog = dspy.ChainOfThought(AcademicDisciplineVerification)

    def forward(self, term: str) -> dspy.Prediction:
        """Verify if a term is an academic discipline."""
        return self.prog(term=term)


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
            term=term,
            is_discipline=str(expected_output["is_discipline"]).lower(),
            reasoning=expected_output["reasoning"],
        ).with_inputs("term")

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
            # Provide specific feedback based on the term and expected answer
            term = gold.term.lower()
            
            if pred_is_discipline and not gold_is_discipline:
                # False positive - should have said False
                if term in ['surgery', 'pediatrics', 'neurology', 'radiology', 'ophthalmology', 'gynecology', 'obstetrics']:
                    feedback = f"'{term}' is a medical specialty within medicine, not a top-level discipline."
                elif term in ['microbiology', 'immunology', 'neurobiology', 'dermatology']:
                    feedback = f"'{term}' is a sub-field of biology/medicine, not a standalone academic discipline."
                elif term in ['literatures', 'computing']:
                    feedback = f"'{term}' is too vague - the actual disciplines would be 'literature' or 'computer science'."
                elif term == 'race':
                    feedback = f"'{term}' is a social descriptor, not an academic discipline. Related field would be 'ethnic studies'."
                else:
                    feedback = f"'{term}' is too specific/specialized. Ask: would this have its own college at most universities?"
            else:
                # False negative - should have said True
                if term in ['engineering', 'medicine', 'business', 'law', 'humanities', 'mathematics', 'physics']:
                    feedback = f"'{term}' is a fundamental academic discipline with its own colleges at most universities."
                elif 'science' in term or term in ['psychology', 'sociology', 'anthropology']:
                    feedback = f"'{term}' is a recognized academic field taught as a major discipline."
                else:
                    feedback = f"'{term}' is a legitimate academic discipline - check if it would have its own college/school."

        return dspy.Prediction(score=score, feedback=feedback)

    except Exception as e:
        return dspy.Prediction(
            score=0.0,
            feedback=f"Error evaluating prediction: {str(e)}. Ensure proper output format.",
        )




# Default prompts - aligned with actual implementation in lv0_s3_verify_single_token.py
DEFAULT_S3_SYSTEM = """You are an expert in academic research classification with a deep understanding of research domains, 
academic departments, scientific disciplines, and specialized fields of study.

Your task is to verify whether terms represent legitimate broad academic disciplines by considering:
1. Academic relevance - Is it a recognized field of study or broad academic discipline?
2. Disciplinary context - Does it represent a major division of knowledge in academia?
3. Scope - Is it broad enough to encompass multiple research areas or subdisciplines?

Accept:
- Broad academic disciplines (e.g., humanities, sciences, engineering)
- Major fields of study (e.g., arts, medicine, law)
- Traditional knowledge domains (e.g., social sciences, natural sciences)

DO NOT accept:
- Narrow specializations or subdisciplines (e.g., organic chemistry, medieval history)
- Technical methodologies (e.g., spectroscopy, chromatography)
- Specific research topics (e.g., climate change, artificial intelligence)
- Acronyms (e.g., STEM, AI) unless they are universally recognized as standalone concepts
- Proper nouns or names (e.g., Harvard, MIT)
- Informal or colloquial terms (e.g., stuff, thing)
- General English words without specific academic meaning"""

DEFAULT_S3_USER = """Analyze whether "{keyword}" is a valid broad academic discipline.

Evidence - Colleges/schools/divisions that mention this concept:
{evidence_colleges}

Consider:
1. Is it a recognized major field of study or broad academic discipline?
2. Does it represent a major division of knowledge in academia?
3. Is it broad enough to encompass multiple research areas or subdisciplines?

Answer with true if it meets these criteria, false otherwise."""


def optimize_prompts():
    """Run GEPA optimization for discipline verification prompts."""
    return run_optimization(
        program_name="lv0_s3_discipline_verification",
        training_data_path=TRAINING_DATA_PATH,
        dspy_module=DisciplineVerifier,
        metric_func=metric_with_feedback,
        prepare_examples_func=prepare_dspy_examples,
        default_system_prompt=DEFAULT_S3_SYSTEM,
        default_user_prompt=DEFAULT_S3_USER,
        create_training_data_func=create_training_data,
    )


if __name__ == "__main__":
    try:
        optimize_prompts()
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        sys.exit(1)
