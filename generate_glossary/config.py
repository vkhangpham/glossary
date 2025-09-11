"""
Centralized configuration for the glossary generation system.

This module provides a unified configuration system that consolidates all 
configuration constants from across the codebase. Configuration can be
overridden via environment variables and is validated at startup.

## Configuration Hierarchy:
1. Base configuration (defined in this module)
2. Level-specific overrides
3. Environment variable overrides (highest priority)

## Environment Variables:
All configuration can be overridden via environment variables with the
prefix GLOSSARY_. For example:
- GLOSSARY_BATCH_SIZE=10
- GLOSSARY_TEMPERATURE=0.7
- GLOSSARY_LLM_MODEL_TIER=flagship
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Set, Union
from dotenv import load_dotenv

from generate_glossary.utils.logger import get_logger

# Load environment variables
load_dotenv()

# Logger for config module
logger = get_logger(__name__)

# Base directory - project root
BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = BASE_DIR / "data"

@dataclass
class StepConfig:
    """Configuration for generation steps (integrated from level_config.py)."""
    batch_size: int
    agreement_threshold: int
    consensus_attempts: int  # Number of consensus responses to request
    search_patterns: List[str]
    quality_keywords: List[str]
    frequency_threshold: Union[float, str]
    processing_description: str
    context_description: str


@dataclass
class LLMConfig:
    """Configuration for LLM operations."""
    # Model tiers and selection
    model_tiers: Dict[str, List[str]] = field(default_factory=dict)
    model_aliases: Dict[str, str] = field(default_factory=dict)
    default_tier: str = "budget"
    
    # Provider prefix mappings for model normalization
    model_provider_prefixes: Dict[str, str] = field(default_factory=dict)
    openai_model_prefixes: List[str] = field(default_factory=list)
    
    # Processing parameters
    temperature: float = 1.0
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_multiplier: float = 2.0
    cache_ttl: int = 3600  # Cache TTL in seconds
    
    # Performance settings
    enable_json_validation: bool = True
    verbose: bool = False
    experimental_http_handler: bool = True
    
    # Smart Consensus Configuration
    enable_smart_consensus: bool = True
    confidence_threshold: float = 0.85  # Confidence threshold for early stopping
    min_responses: int = 2  # Minimum responses before early stopping
    max_responses: int = 3  # Maximum responses
    agreement_threshold: float = 0.8  # Agreement threshold for early stopping
    
    # Enhanced Caching Configuration
    enable_enhanced_cache: bool = True
    semantic_similarity_threshold: float = 0.95  # Similarity threshold for cache hits
    enable_persistent_cache: bool = True
    cache_storage_path: str = "data/cache"  # Path for persistent cache storage
    enable_cache_analytics: bool = True
    
    # Flexible Model and Temperature Configuration
    per_use_case_models: Dict[str, str] = field(default_factory=dict)
    per_use_case_temperatures: Dict[str, float] = field(default_factory=dict)
    
    # Optimization Settings
    enable_optimization: bool = True
    optimization_fallback_strategy: str = "conservative"  # "conservative" or "aggressive"
    performance_tuning_mode: str = "balanced"  # "speed", "quality", or "balanced"
    
    # Deterministic model selection
    deterministic_tier_selection: bool = False  # Enable deterministic tier selection for reproducibility
    random_seed: Optional[int] = None  # Random seed for deterministic selection
    
    def __post_init__(self):
        if not self.model_tiers:
            self.model_tiers = {
                "budget": [
                    "openai/gpt-5-nano",
                    "openai/gpt-4o-mini",
                    "vertex_ai/gemini-2.5-flash",
                ],
                "balanced": [
                    "openai/gpt-5-mini",
                    "vertex_ai/gemini-2.5-flash",
                ],
                "flagship": [
                    "openai/gpt-5",
                    "anthropic/claude-4-sonnet",
                    "vertex_ai/gemini-2.5-pro",
                ],
            }
        
        if not self.model_aliases:
            self.model_aliases = {
                "gpt5nano": "openai/gpt-5-nano",
                "gpt5mini": "openai/gpt-5-mini",
                "gpt5": "openai/gpt-5",
                "gpt4omini": "openai/gpt-4o-mini",
                "claude4sonnet": "anthropic/claude-4-sonnet",
                "gemini2.5flash": "vertex_ai/gemini-2.5-flash",
                "gemini2.5pro": "vertex_ai/gemini-2.5-pro",
                # Special aliases moved from hardcoded logic
                "sonnet": "anthropic/claude-sonnet-4-20250514",
                "haiku": "anthropic/claude-3-5-haiku-20241022",
                "gemini-2.5-flash": "vertex_ai/gemini-2.5-flash",
                "gemini-2.5-pro": "vertex_ai/gemini-2.5-pro",
            }
        
        # Initialize provider prefix mappings with current hardcoded behavior
        if not self.model_provider_prefixes:
            self.model_provider_prefixes = {
                "claude-": "anthropic/",
                "gemini-": "vertex_ai/",
            }
        
        # Initialize OpenAI model prefixes with current hardcoded patterns
        if not self.openai_model_prefixes:
            self.openai_model_prefixes = [
                "gpt-", "text-", "davinci", "curie", "babbage", "ada",
                "embedding", "whisper", "tts", "dall-e", "o1-"
            ]
        
        # Validate that provider IDs end with '/' to prevent malformed strings
        for prefix, provider_id in self.model_provider_prefixes.items():
            if not provider_id.endswith("/"):
                self.model_provider_prefixes[prefix] = f"{provider_id}/"
        
        # Initialize per-use-case configurations if empty
        if not self.per_use_case_models:
            self.per_use_case_models = {
                "concept_extraction": "openai/gpt-5-nano",
                "frequency_filtering": "openai/gpt-4o-mini",
                "token_verification": "openai/gpt-5-mini",
                "web_extraction": "openai/gpt-5-nano",
            }
        
        if not self.per_use_case_temperatures:
            self.per_use_case_temperatures = {
                "concept_extraction": 1.0,
                "frequency_filtering": 0.7,
                "token_verification": 0.5,
                "web_extraction": 0.8,
            }


@dataclass
class LevelConfig:
    """Configuration for a specific level (0, 1, 2, or 3)"""
    level: int
    name: str  # Human-readable name
    
    # Step configuration (integrated from level_config.py)
    step_config: Optional[StepConfig] = None
    
    # File paths - use Path objects for better path handling
    @property
    def data_dir(self) -> Path:
        """Directory for this level's data"""
        return DATA_DIR / f"lv{self.level}"
    
    @property
    def raw_dir(self) -> Path:
        """Raw data directory"""
        return self.data_dir / "raw"
    
    @property
    def postprocessed_dir(self) -> Path:
        """Post-processed data directory"""
        return self.data_dir / "postprocessed"
    
    def get_step_input_file(self, step: int) -> Path:
        """Get input file path for a specific step"""
        if step == 0:
            if self.level == 0:
                raise NotImplementedError("Step 0 input for level 0 is handled by lv0_s0 script")
            # For levels 1-3, step 0 input comes from the previous level's final file
            return DATA_DIR / f"lv{self.level-1}" / f"lv{self.level-1}_final.txt"
        elif step == 1:
            return self.get_step_output_file(0)
        else:
            return self.raw_dir / f"lv{self.level}_s{step-1}_{'filtered' if step == 3 else 'extracted'}_concepts.txt"
    
    def get_step_output_file(self, step: int) -> Path:
        """Get output file path for a specific step"""
        step_names = {
            0: f"lv{self.level}_s0_" + ["college_names", "dept_names", "research_areas", "conference_topics"][self.level] + ".txt",
            1: f"lv{self.level}_s1_extracted_concepts.txt",
            2: f"lv{self.level}_s2_filtered_concepts.txt", 
            3: f"lv{self.level}_s3_verified_concepts.txt"
        }
        return self.raw_dir / step_names[step]
    
    def get_step_metadata_file(self, step: int) -> Path:
        """Get metadata file path for a specific step"""
        return self.raw_dir / f"lv{self.level}_s{step}_metadata.json"
    
    def get_final_file(self) -> Path:
        """Get final output file path"""
        return self.data_dir / f"lv{self.level}_final.txt"
    
    def get_validation_metadata_file(self, step: int) -> Path:
        """Get validation metadata file path for a specific step"""
        return self.raw_dir / f"lv{self.level}_s{step}_validation_metadata.json"
    
    # Pass-through properties for StepConfig fields
    @property
    def processing_description(self):
        return self.step_config.processing_description if self.step_config else ""
    
    @property
    def context_description(self):
        return self.step_config.context_description if self.step_config else ""
    
    @property
    def frequency_threshold(self):
        return self.step_config.frequency_threshold if self.step_config else 0.6
    
    @property
    def agreement_threshold(self):
        return self.step_config.agreement_threshold if self.step_config else 2
    
    @property
    def quality_keywords(self):
        return self.step_config.quality_keywords if self.step_config else []

@dataclass 
class ProcessingConfig:
    """Configuration for processing parameters.
    
    This consolidates all processing configuration from across the codebase,
    including constants that were hardcoded in individual modules.
    """
    
    # LLM settings (from lv0_s1_extract_concepts.py and others)
    llm_attempts: int = 3
    concept_agreement_threshold: int = 2
    temperature: float = 1.0  # DEPRECATED: Now an alias for LLMConfig.temperature (single source of truth)
    cache_ttl: int = 3600  # Cache consensus results for 1 hour
    
    # Batch processing
    batch_size: int = 20
    max_workers: int = 4
    chunk_size: int = 100
    
    # Filtering thresholds
    keyword_appearance_threshold: int = 2
    institution_frequency_threshold: float = 0.6  # 60% of institutions
    conference_frequency_threshold: int = 1
    
    # Performance settings
    max_concurrent_requests: int = 15
    max_concurrent_browsers: int = 3
    request_timeout: int = 30
    
    # Verification settings
    cooldown_period: int = 2  # seconds between API calls
    cooldown_frequency: int = 5  # calls before cooldown
    max_retries: int = 3
    max_examples: int = 5  # Max examples to show in prompts
    
    # Validation file paths
    validation_meta_suffix: str = "_validation_metadata.json"
    
    # Non-academic terms to filter out (from frequency_filtering.py)
    non_academic_terms: Set[str] = field(default_factory=set)
    
    # Concept validation (from frequency_filtering.py)
    min_concept_length: int = 2  # Minimum concept length
    max_concept_length: int = 100  # Maximum concept length
    min_dept_length: int = 3
    max_dept_length: int = 100
    min_consensus: float = 0.6  # For compound term splitting
    institution_freq_threshold_percent: float = 60  # 60% threshold
    min_college_freq_percent: float = 20  # Minimum college frequency percentage
    min_college_appearance: int = 2
    min_dept_appearance: int = 2
    min_dept_freq_percent: float = 20  # Minimum department frequency percentage
    max_concepts_per_plot: int = 20
    num_workers: int = 4  # For multiprocessing
    log_level: str = "INFO"
    
    def __post_init__(self):
        if not self.non_academic_terms:
            # Extended list from frequency_filtering.py
            self.non_academic_terms = {
                # Institutional terms
                "university", "college", "school", "department", "center", "institute", 
                "program", "studies", "research", "science", "sciences", "technology",
                "administration", "management", "development", "international", "public",
                "general", "applied", "theoretical", "advanced", "basic", "modern",
                "clinical", "experimental", "social", "human", "natural", "physical",
                # Web navigation terms (from frequency_filtering.py)
                "page", "home", "about", "contact", "staff", "faculty", "links",
                "click", "here", "website", "portal", "login", "apply", "apply now",
                "register", "registration", "more", "learn more", "read more",
                "back", "next", "previous", "link", "site", "menu", "navigation"
            }


@dataclass
class WebExtractionConfig:
    """Configuration for web extraction operations."""
    
    # Processing constants (from web_extraction_firecrawl.py)
    batch_size: int = 25  # Firecrawl handles batching efficiently
    num_llm_attempts: int = 3
    max_results_per_term: int = 10
    
    # Mining settings
    max_concurrent_mining: int = 30
    
    # Search parameters
    search_provider: str = "firecrawl"  # Default provider


@dataclass
class MiningConfig:
    """Configuration for mining operations."""
    
    # Processing constants (from mining/firecrawl.py)
    batch_size: int = 25
    max_concurrent_operations: int = 5
    max_urls_per_concept: int = 3
    
    # Performance settings
    request_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass 
class ValidationConfig:
    """Configuration for validation and post-processing"""
    
    # Rule validation constants (from validation/rule_validator.py)
    max_workers: int = field(default_factory=lambda: min(32, (os.cpu_count() or 1) * 2))
    
    # Web mining
    search_provider: str = "rapidapi"  # or "tavily"
    max_concurrent_mining: int = 30
    
    # Deduplication
    dedup_method: str = "graph"  # or "fuzzy" 
    similarity_threshold: float = 0.8
    
    # Validation modes
    validation_modes: list = None
    
    def __post_init__(self):
        if self.validation_modes is None:
            self.validation_modes = ["web", "rule"]

class GlossaryConfig:
    """Main configuration class for the entire glossary generation system.
    
    This class consolidates all configuration from across the codebase and
    provides a unified interface for accessing configuration values.
    """
    
    def __init__(self):
        # Initialize step configs (from level_config.py)
        self._init_step_configs()
        
        # Level configurations with integrated step configs
        self.levels = {
            0: LevelConfig(0, "Colleges/Schools", self.step_configs.get(0)),
            1: LevelConfig(1, "Departments", self.step_configs.get(1)),
            2: LevelConfig(2, "Research Areas", self.step_configs.get(2)),
            3: LevelConfig(3, "Conference Topics", self.step_configs.get(3))
        }
        
        # LLM configuration
        self.llm = LLMConfig()
        
        # Processing configuration - can be overridden per level
        self.processing = ProcessingConfig()
        
        # Web extraction configuration
        self.web_extraction = WebExtractionConfig()
        
        # Mining configuration
        self.mining = MiningConfig()
        
        # Validation configuration
        self.validation = ValidationConfig()
        
        # Level-specific processing overrides
        self._setup_level_overrides()
        
        # Apply environment overrides
        self._apply_environment_overrides()
    
    def _init_step_configs(self):
        """Initialize step configurations for each level (from level_config.py)."""
        self.step_configs = {
            # Level 0 configuration (new - was missing from level_config.py)
            0: StepConfig(
                batch_size=20,
                agreement_threshold=2,
                consensus_attempts=3,  # Number of responses for consensus
                search_patterns=[
                    "{term} college site:edu",
                    "{term} school site:edu",
                    "{term} academic programs site:edu",
                ],
                quality_keywords=["college", "school", "university", "academic", "education"],
                frequency_threshold=0.6,
                processing_description="College/School extraction from institutional data",
                context_description="academic colleges and schools"
            ),
            # Level 1 configuration (from level_config.py)
            1: StepConfig(
                batch_size=15,
                agreement_threshold=2,
                consensus_attempts=3,  # Number of responses for consensus
                search_patterns=[
                    "{term} departments site:edu",
                    "departments {term} university site:edu",
                    "{term} academic programs site:edu",
                    "{term} schools site:edu"
                ],
                quality_keywords=["department", "school", "program", "college", "faculty", "academic", "research"],
                frequency_threshold=0.6,
                processing_description="Department extraction from college contexts",
                context_description="academic departments and fields of study"
            ),
            # Level 2 configuration (from level_config.py)
            2: StepConfig(
                batch_size=5,
                agreement_threshold=3,
                consensus_attempts=5,  # Higher consensus attempts for more specific levels
                search_patterns=[
                    "{term} research areas site:edu",
                    "{term} research groups site:edu",
                    "{term} labs site:edu",
                    "{term} research centers site:edu",
                    "{term} specializations site:edu"
                ],
                quality_keywords=["research", "lab", "group", "center", "institute", "laboratory", "academic", "conference"],
                frequency_threshold=0.6,
                processing_description="Research area extraction from department contexts",
                context_description="research areas and academic specializations"
            ),
            # Level 3 configuration (from level_config.py)
            3: StepConfig(
                batch_size=5,
                agreement_threshold=3,
                consensus_attempts=5,  # Higher consensus attempts for most specific level
                search_patterns=[
                    "{term} conference topics",
                    "{term} call for papers",
                    "{term} conference tracks",
                    "{term} special issues",
                    "{term} workshop topics",
                    "{term} symposium topics"
                ],
                quality_keywords=["conference", "workshop", "symposium", "cfp", "call", "papers", "track", "academic"],
                frequency_threshold="venue_based",
                processing_description="Conference topic extraction from research area contexts",
                context_description="conference topics and academic themes"
            )
        }
        
    def _setup_level_overrides(self):
        """Setup level-specific processing parameter overrides.
        
        These override the base ProcessingConfig values for each level.
        Values here take precedence over base config but can still be
        overridden by environment variables.
        """
        self.level_overrides = {
            # Level 0 - Colleges (broadest, more conservative)
            0: {
                "batch_size": 20,
                "concept_agreement_threshold": 2,
                "keyword_appearance_threshold": 2,
                "chunk_size": 100,
                "max_workers": 4,
                "temperature": 1.0,  # GPT-5 requirement
            },
            # Level 1 - Departments (moderate)  
            1: {
                "batch_size": 15,
                "concept_agreement_threshold": 2,
                "keyword_appearance_threshold": 2,
                "chunk_size": 100,
            },
            # Level 2 - Research Areas (more specific)
            2: {
                "batch_size": 5,
                "concept_agreement_threshold": 3,  # Higher agreement needed
                "keyword_appearance_threshold": 1,
                "chunk_size": 50,
            },
            # Level 3 - Conference Topics (most specific)
            3: {
                "batch_size": 5,
                "concept_agreement_threshold": 3,
                "keyword_appearance_threshold": 1,
                "chunk_size": 50,
            }
        }
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides to configuration.
        
        Environment variables follow the pattern GLOSSARY_<SECTION>_<KEY>.
        For example:
        - GLOSSARY_PROCESSING_BATCH_SIZE=10
        - GLOSSARY_LLM_TEMPERATURE=0.7
        - GLOSSARY_LLM_DEFAULT_TIER=flagship
        """
        def _parse_env(env_value: str, current_value, env_key: str = ""):
            """Parse environment variable based on the type of current value."""
            # Check bool before int since bool is a subclass of int
            if isinstance(current_value, bool):
                return env_value.lower() in ('true', '1', 'yes', 'y', 'on')
            if isinstance(current_value, int):
                return int(env_value)
            if isinstance(current_value, float):
                return float(env_value)
            if isinstance(current_value, (dict, list)):
                try:
                    return json.loads(env_value)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse JSON for environment variable {env_key or 'unknown'}="
                        f"{env_value!r} (type: {type(current_value).__name__}): {e}. "
                        f"Using raw string value as fallback."
                    )
                    return env_value
            return env_value
        
        # Processing config overrides
        for key in asdict(self.processing).keys():
            env_key = f"GLOSSARY_PROCESSING_{key.upper()}"
            if env_value := os.getenv(env_key):
                try:
                    current_value = getattr(self.processing, key)
                    setattr(self.processing, key, _parse_env(env_value, current_value, env_key))
                except (ValueError, TypeError):
                    pass  # Keep default if conversion fails
        
        # LLM config overrides
        for key in asdict(self.llm).keys():
            env_key = f"GLOSSARY_LLM_{key.upper()}"
            if env_value := os.getenv(env_key):
                try:
                    current_value = getattr(self.llm, key)
                    setattr(self.llm, key, _parse_env(env_value, current_value, env_key))
                except (ValueError, TypeError):
                    pass
        
        # Web extraction config overrides
        for key in asdict(self.web_extraction).keys():
            env_key = f"GLOSSARY_WEB_EXTRACTION_{key.upper()}"
            if env_value := os.getenv(env_key):
                try:
                    current_value = getattr(self.web_extraction, key)
                    setattr(self.web_extraction, key, _parse_env(env_value, current_value, env_key))
                except (ValueError, TypeError):
                    pass
        
        # Mining config overrides
        for key in asdict(self.mining).keys():
            env_key = f"GLOSSARY_MINING_{key.upper()}"
            if env_value := os.getenv(env_key):
                try:
                    current_value = getattr(self.mining, key)
                    setattr(self.mining, key, _parse_env(env_value, current_value, env_key))
                except (ValueError, TypeError):
                    pass
        
        # Validation config overrides
        for key in asdict(self.validation).keys():
            env_key = f"GLOSSARY_VALIDATION_{key.upper()}"
            if env_value := os.getenv(env_key):
                try:
                    current_value = getattr(self.validation, key)
                    setattr(self.validation, key, _parse_env(env_value, current_value, env_key))
                except (ValueError, TypeError):
                    pass
    
    def get_processing_config(self, level: int) -> ProcessingConfig:
        """Get processing configuration for a specific level with overrides applied"""
        config = ProcessingConfig(
            **{
                # Start with base configuration
                **self.processing.__dict__,
                # Apply level-specific overrides
                **self.level_overrides.get(level, {})
            }
        )
        # Apply legacy environment overrides for backward compatibility
        # Precedence: base config -> level overrides -> new env vars (via _apply_environment_overrides) -> legacy env vars
        config = EnvConfig.apply_env_overrides(config)
        # Temperature is now sourced from LLMConfig as single source of truth
        # ProcessingConfig.temperature is deprecated and acts as an alias
        config.temperature = self.llm.temperature
        return config
    
    def get_level_config(self, level: int) -> LevelConfig:
        """Get level configuration."""
        if level not in self.levels:
            raise ValueError(f"Invalid level: {level}. Must be 0, 1, 2, or 3")
        return self.levels[level]
    
    def get_step_config(self, level: int) -> Optional[StepConfig]:
        """Get step configuration for a level.
        
        This provides backward compatibility with level_config.py.
        """
        return self.step_configs.get(level)
    
    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration."""
        return self.llm
    
    def get_web_extraction_config(self) -> WebExtractionConfig:
        """Get web extraction configuration."""
        return self.web_extraction
    
    def get_mining_config(self) -> MiningConfig:
        """Get mining configuration."""
        return self.mining
    
    def get_validation_config(self) -> ValidationConfig:
        """Get validation configuration."""
        return self.validation
    
    def validate_configuration(self) -> List[str]:
        """Validate the current configuration.
        
        Returns:
            List of validation errors (empty if configuration is valid)
        """
        errors = []
        
        # Validate processing config
        if self.processing.batch_size <= 0:
            errors.append("Processing batch_size must be positive")
        if self.processing.max_workers <= 0:
            errors.append("Processing max_workers must be positive")
        if not 0 <= self.processing.temperature <= 2:
            errors.append("Processing temperature must be between 0 and 2")
        
        # Validate LLM config
        if not self.llm.model_tiers:
            errors.append("LLM model_tiers cannot be empty")
        if self.llm.default_tier not in self.llm.model_tiers:
            errors.append(f"LLM default_tier '{self.llm.default_tier}' not in model_tiers")
        
        # Validate level configs
        for level, config in self.levels.items():
            if config.step_config:
                if config.step_config.batch_size <= 0:
                    errors.append(f"Level {level} batch_size must be positive")
                if config.step_config.agreement_threshold <= 0:
                    errors.append(f"Level {level} agreement_threshold must be positive")
        
        return errors
        
    def ensure_directories(self, level: int = None):
        """Ensure all required directories exist"""
        if level is not None:
            # Create directories for specific level
            level_config = self.get_level_config(level)
            level_config.data_dir.mkdir(parents=True, exist_ok=True)
            level_config.raw_dir.mkdir(parents=True, exist_ok=True)
            level_config.postprocessed_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Create directories for all levels
            for level in self.levels.keys():
                self.ensure_directories(level)

# Global configuration instance
config = GlossaryConfig()

# Convenience functions for backward compatibility
def get_level_config(level: int) -> LevelConfig:
    """Get configuration for a specific level."""
    return config.get_level_config(level)

def get_processing_config(level: int) -> ProcessingConfig:
    """Get processing configuration for a specific level."""
    return config.get_processing_config(level)

def get_step_config(level: int) -> Optional[StepConfig]:
    """Get step configuration for a specific level.
    
    Provides backward compatibility with level_config.py.
    """
    return config.get_step_config(level)

def get_llm_config() -> LLMConfig:
    """Get LLM configuration."""
    return config.get_llm_config()

def get_web_extraction_config() -> WebExtractionConfig:
    """Get web extraction configuration."""
    return config.get_web_extraction_config()

def get_mining_config() -> MiningConfig:
    """Get mining configuration."""
    return config.get_mining_config()

def get_validation_config() -> ValidationConfig:
    """Get validation configuration."""
    return config.get_validation_config()

def ensure_directories(level: int = None):
    """Ensure required directories exist."""
    return config.ensure_directories(level)

def validate_configuration() -> List[str]:
    """Validate the current configuration.
    
    Returns:
        List of validation errors (empty if valid)
    """
    return config.validate_configuration()

# Environment variable configuration
class EnvConfig:
    """Environment-based configuration for API keys and legacy overrides.
    
    This class handles API keys and provides backward compatibility for
    the old environment variable format. New code should use the
    GLOSSARY_<SECTION>_<KEY> format which is handled in GlossaryConfig.
    """
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") 
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
    
    # Legacy environment variable support (backward compatibility)
    BATCH_SIZE = os.getenv("GLOSSARY_BATCH_SIZE")
    MAX_WORKERS = os.getenv("GLOSSARY_MAX_WORKERS") 
    CONCEPT_AGREEMENT_THRESHOLD = os.getenv("GLOSSARY_CONCEPT_AGREEMENT_THRESHOLD")
    
    @classmethod
    def apply_env_overrides(cls, processing_config: ProcessingConfig) -> ProcessingConfig:
        """Apply legacy environment variable overrides.
        
        This method provides backward compatibility for the old environment
        variable format. New code should use GLOSSARY_PROCESSING_* format.
        """
        if cls.BATCH_SIZE:
            try:
                processing_config.batch_size = int(cls.BATCH_SIZE)
            except ValueError:
                pass
                
        if cls.MAX_WORKERS:
            try:
                processing_config.max_workers = int(cls.MAX_WORKERS)
            except ValueError:
                pass
                
        if cls.CONCEPT_AGREEMENT_THRESHOLD:
            try:
                processing_config.concept_agreement_threshold = int(cls.CONCEPT_AGREEMENT_THRESHOLD)
            except ValueError:
                pass
                
        return processing_config

