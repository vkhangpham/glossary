"""
GEPA adapter for optimizing concept extraction prompts.
This adapter integrates with the existing extraction pipeline to optimize prompts.
"""

from typing import List, Dict, Any, Optional
import asyncio

from gepa import GEPAAdapter, EvaluationBatch
from pydantic import BaseModel, Field

from generate_glossary.utils.logger import setup_logger
from generate_glossary.utils.llm import structured_completion_consensus

logger = setup_logger("gepa_adapter")


class ConceptExtraction(BaseModel):
    """Model for concept extraction results"""

    source: str = Field(description="Source text being processed")
    concepts: List[str] = Field(description="List of extracted concepts")


class ConceptExtractionList(BaseModel):
    """Model for batch extraction results"""

    extractions: List[ConceptExtraction] = Field(
        description="List of concept extractions"
    )


class ConceptExtractionAdapter(GEPAAdapter):
    """
    GEPA adapter for optimizing concept extraction prompts.

    This adapter:
    1. Evaluates candidate prompts on batches of college/department names
    2. Scores based on extraction quality (precision, recall, consistency)
    3. Provides feedback for prompt improvement
    """

    def __init__(
        self,
        level: int = 0,
        ground_truth: Optional[Dict[str, List[str]]] = None,
        task_model: str = "openai/gpt-4o-mini",
        num_consensus: int = 3,
        temperature: float = 1.0,
    ):
        """
        Initialize the adapter.

        Args:
            level: Extraction level (0=colleges, 1=departments, etc.)
            ground_truth: Optional ground truth for scoring
            task_model: Model to optimize prompts for
            num_consensus: Number of LLM calls for consensus
            temperature: Temperature for generation
        """
        self.level = level
        self.ground_truth = ground_truth or {}
        self.task_model = task_model
        self.num_consensus = num_consensus
        self.temperature = temperature
        self.cache = {}  # Cache extraction results

    def evaluate(
        self,
        batch: List[Dict[str, Any]],
        candidate: Dict[str, str],
        capture_traces: bool = True,
    ) -> EvaluationBatch:
        """
        Evaluate a candidate prompt on a batch of inputs.

        Args:
            batch: List of input data (college/department names)
            candidate: Candidate prompts to evaluate
            capture_traces: Whether to capture execution traces

        Returns:
            EvaluationBatch with outputs, scores, and traces
        """
        outputs = []
        scores = []
        trajectories = [] if capture_traces else None

        # Extract prompt components
        system_prompt = candidate.get("system_prompt", "")
        user_template = candidate.get("user_template", "")

        # Process each batch item
        for item in batch:
            input_text = item.get("input", "")
            expected = item.get("expected", [])  # Ground truth if available

            # Run extraction with candidate prompt
            try:
                extraction = self._extract_concepts(
                    input_text, system_prompt, user_template
                )

                # Score the extraction
                score = self._compute_score(extraction, expected, input_text)

                outputs.append({"extraction": extraction})
                scores.append(score)

                if capture_traces:
                    trajectories.append(
                        {
                            "input": input_text,
                            "extraction": extraction,
                            "expected": expected,
                            "system_prompt": system_prompt,
                            "score": score,
                        }
                    )

            except Exception as e:
                logger.error(f"Extraction failed for '{input_text}': {e}")
                outputs.append({"extraction": []})
                scores.append(0.0)

                if capture_traces:
                    trajectories.append(
                        {
                            "input": input_text,
                            "error": str(e),
                            "system_prompt": system_prompt,
                            "score": 0.0,
                        }
                    )

        return EvaluationBatch(
            outputs=outputs, scores=scores, trajectories=trajectories
        )

    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: List[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create reflection data for GEPA to improve prompts.

        Args:
            candidate: Current candidate prompts
            eval_batch: Evaluation results
            components_to_update: Which prompts to update

        Returns:
            Reflection dataset for each component
        """
        reflections = {}

        for component in components_to_update:
            items = []

            # Analyze failures and successes
            for traj, score, output in zip(
                eval_batch.trajectories or [], eval_batch.scores, eval_batch.outputs
            ):
                # Focus on low-scoring extractions for improvement
                if score < 0.8:
                    feedback = self._generate_feedback(traj, score)

                    items.append(
                        {
                            "Input": traj["input"],
                            "Extraction": traj.get("extraction", []),
                            "Expected": traj.get("expected", []),
                            "Score": score,
                            "Feedback": feedback,
                            "Current_Prompt": candidate.get(component, ""),
                        }
                    )

            reflections[component] = items

        return reflections

    def _extract_concepts(
        self, input_text: str, system_prompt: str, user_template: str
    ) -> List[str]:
        """
        Extract concepts using the provided prompts.

        Args:
            input_text: Text to extract from
            system_prompt: System prompt
            user_template: User prompt template

        Returns:
            List of extracted concepts
        """
        # Format user prompt
        user_prompt = user_template.replace("{sources}", f"- {input_text}")

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # Check cache
        cache_key = f"{system_prompt}:{user_prompt}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            # Run consensus extraction
            result = asyncio.run(
                structured_completion_consensus(
                    messages=messages,
                    response_model=ConceptExtractionList,
                    tier="budget",
                    num_responses=self.num_consensus,
                    return_all=False,
                    temperature=self.temperature,
                    cache_ttl=3600,
                )
            )

            # Extract concepts from first result
            if result and result.extractions:
                concepts = result.extractions[0].concepts
                self.cache[cache_key] = concepts
                return concepts

        except Exception as e:
            logger.error(f"Extraction failed: {e}")

        return []

    def _compute_score(
        self, extracted: List[str], expected: List[str], input_text: str
    ) -> float:
        """
        Compute quality score for extraction.

        Args:
            extracted: Extracted concepts
            expected: Expected concepts (ground truth)
            input_text: Original input

        Returns:
            Score between 0 and 1
        """
        # If we have ground truth, use it
        if expected:
            if not extracted:
                return 0.0

            # Compute precision and recall
            extracted_set = set(c.lower() for c in extracted)
            expected_set = set(c.lower() for c in expected)

            if not expected_set:
                return 1.0 if not extracted_set else 0.0

            intersection = extracted_set & expected_set
            precision = len(intersection) / len(extracted_set) if extracted_set else 0
            recall = len(intersection) / len(expected_set) if expected_set else 0

            # F1 score
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            return f1

        # Heuristic scoring without ground truth
        score = 1.0

        # Penalize empty extractions for non-empty input
        if not extracted and input_text.strip():
            return 0.0

        # Penalize too many extractions
        if len(extracted) > 5:
            score *= 0.8

        # Check for reasonable concepts
        for concept in extracted:
            # Penalize very short or very long concepts
            if len(concept) < 3 or len(concept) > 50:
                score *= 0.9

            # Penalize concepts that are just the input
            if concept.lower() == input_text.lower():
                score *= 0.7

        return max(0.0, min(1.0, score))

    def _generate_feedback(self, trajectory: Dict[str, Any], score: float) -> str:
        """
        Generate feedback for prompt improvement.

        Args:
            trajectory: Execution trace
            score: Quality score

        Returns:
            Feedback string
        """
        input_text = trajectory.get("input", "")
        extracted = trajectory.get("extraction", [])
        expected = trajectory.get("expected", [])

        feedback_parts = []

        if score < 0.5:
            feedback_parts.append("Low extraction quality.")

        if expected:
            extracted_set = set(c.lower() for c in extracted)
            expected_set = set(c.lower() for c in expected)

            missed = expected_set - extracted_set
            extra = extracted_set - expected_set

            if missed:
                feedback_parts.append(f"Missed concepts: {', '.join(missed)}")
            if extra:
                feedback_parts.append(f"Extra concepts: {', '.join(extra)}")

        if not extracted and input_text:
            feedback_parts.append(
                "Failed to extract any concepts from non-empty input."
            )

        if len(extracted) > 5:
            feedback_parts.append("Too many concepts extracted. Be more selective.")

        if not feedback_parts:
            feedback_parts.append("Extraction seems reasonable but could be improved.")

        return " ".join(feedback_parts)

