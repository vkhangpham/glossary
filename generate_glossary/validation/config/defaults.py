"""
Immutable configuration dataclasses for the validation system.

This module defines comprehensive configuration dataclasses for each validator type,
following the functional programming pattern with immutable @frozen dataclasses.
"""

from dataclasses import dataclass
from typing import FrozenSet, Mapping, Tuple
from types import MappingProxyType

# Default blacklist terms for rule validation
DEFAULT_BLACKLIST_TERMS = frozenset({
    'test', 'example', 'sample', 'demo', 'none', 'null',
    'undefined', 'unknown', 'other', 'misc', 'miscellaneous',
    'page', 'home', 'index', 'about', 'contact'
})

# Default validation prompt for LLM validation
DEFAULT_VALIDATION_PROMPT = """
Is '{term}' a valid academic discipline, field of study, or technical concept?

Please answer with 'yes' or 'no' followed by a brief explanation.
Consider:
1. Is it a recognized academic or technical term?
2. Is it specific enough to be meaningful?
3. Is it used in academic or professional contexts?

Format: yes/no - explanation
"""

# Default batch validation prompt for LLM validation
BATCH_VALIDATION_PROMPT = """
For each of the following terms, determine if it is a valid academic discipline, field of study, or technical concept.

Terms:
{terms_list}

For each term, provide:
- Valid: yes/no
- Confidence: 0.0-1.0
- Reason: brief explanation

Return as JSON array with format:
[{{"term": "...", "valid": true/false, "confidence": 0.X, "reason": "..."}}]
"""


@dataclass(frozen=True)
class RuleValidationConfig:
    """Immutable configuration for rule-based validation."""
    max_workers: int = 4
    blacklist_terms: FrozenSet[str] = DEFAULT_BLACKLIST_TERMS
    min_term_length: int = 2
    max_term_length: int = 100
    show_progress: bool = True

    def __post_init__(self):
        """Validate configuration values."""
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if self.min_term_length <= 0:
            raise ValueError("min_term_length must be positive")
        if self.max_term_length <= self.min_term_length:
            raise ValueError("max_term_length must be greater than min_term_length")


@dataclass(frozen=True)
class WebValidationConfig:
    """Immutable configuration for web-based validation."""
    min_score: float = 0.5
    min_relevance_score: float = 0.5
    min_relevant_sources: int = 1
    high_quality_content_threshold: float = 0.7
    high_quality_relevance_threshold: float = 0.7
    max_workers: int = 4
    show_progress: bool = True

    def __post_init__(self):
        """Validate configuration values."""
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be between 0.0 and 1.0")
        if not 0.0 <= self.min_relevance_score <= 1.0:
            raise ValueError("min_relevance_score must be between 0.0 and 1.0")
        if self.min_relevant_sources <= 0:
            raise ValueError("min_relevant_sources must be positive")
        if not 0.0 <= self.high_quality_content_threshold <= 1.0:
            raise ValueError("high_quality_content_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.high_quality_relevance_threshold <= 1.0:
            raise ValueError("high_quality_relevance_threshold must be between 0.0 and 1.0")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")


@dataclass(frozen=True)
class LLMValidationConfig:
    """Immutable configuration for LLM-based validation."""
    provider: str = "gemini"
    batch_size: int = 10
    max_workers: int = 4
    validation_prompt: str = DEFAULT_VALIDATION_PROMPT
    batch_prompt: str = BATCH_VALIDATION_PROMPT
    tier: str = "budget"
    max_tokens: int = 100
    batch_max_tokens: int = 500
    show_progress: bool = True

    def __post_init__(self):
        """Validate configuration values."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.batch_max_tokens <= 0:
            raise ValueError("batch_max_tokens must be positive")
        if not self.validation_prompt.strip():
            raise ValueError("validation_prompt cannot be empty")
        if not self.batch_prompt.strip():
            raise ValueError("batch_prompt cannot be empty")
        valid_providers = {"openai", "gemini", "claude", "anthropic"}
        if self.provider not in valid_providers:
            raise ValueError(f"provider must be one of {valid_providers}")
        valid_tiers = {"budget", "standard", "premium"}
        if self.tier not in valid_tiers:
            raise ValueError(f"tier must be one of {valid_tiers}")


@dataclass(frozen=True)
class ValidationConfig:
    """
    Enhanced immutable configuration that composes all validator configs.

    Threshold Configuration:
    The top-level min_score and min_relevance_score fields serve as fallbacks
    when nested validator configs don't specify their own thresholds.
    However, when adapting to core validation, nested config values take
    precedence over top-level values to maintain specificity.
    """
    modes: Tuple[str, ...] = ("rule",)
    confidence_weights: Mapping[str, float] = None
    min_confidence: float = 0.5
    min_score: float = 0.5
    min_relevance_score: float = 0.5
    parallel: bool = True
    use_cache: bool = True
    rule_config: RuleValidationConfig = None
    web_config: WebValidationConfig = None
    llm_config: LLMValidationConfig = None

    def __post_init__(self):
        """Set defaults and validate configuration."""
        # Set default confidence weights if not provided
        if self.confidence_weights is None:
            object.__setattr__(self, 'confidence_weights', MappingProxyType({
                "rule": 0.3,
                "web": 0.5,
                "llm": 0.2
            }))
        elif not isinstance(self.confidence_weights, (MappingProxyType, dict)):
            raise ValueError("confidence_weights must be a mapping")
        elif isinstance(self.confidence_weights, dict):
            # Convert to immutable mapping
            object.__setattr__(self, 'confidence_weights', MappingProxyType(self.confidence_weights))

        # Set default validator configs if not provided
        if self.rule_config is None:
            object.__setattr__(self, 'rule_config', RuleValidationConfig())
        if self.web_config is None:
            object.__setattr__(self, 'web_config', WebValidationConfig())
        if self.llm_config is None:
            object.__setattr__(self, 'llm_config', LLMValidationConfig())

        # Validate configuration values
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be between 0.0 and 1.0")
        if not 0.0 <= self.min_relevance_score <= 1.0:
            raise ValueError("min_relevance_score must be between 0.0 and 1.0")

        # Validate modes
        valid_modes = {"rule", "web", "llm"}
        if not all(mode in valid_modes for mode in self.modes):
            raise ValueError(f"All modes must be one of {valid_modes}")
        if not self.modes:
            raise ValueError("At least one validation mode must be specified")

        # Validate confidence weights match modes
        for mode in self.modes:
            if mode not in self.confidence_weights:
                raise ValueError(f"confidence_weights missing for mode: {mode}")

        # Validate confidence weights sum to reasonable value
        total_weight = sum(self.confidence_weights[mode] for mode in self.modes)
        if not 0.1 <= total_weight <= 2.0:
            raise ValueError(f"Total confidence weights ({total_weight}) should be between 0.1 and 2.0")