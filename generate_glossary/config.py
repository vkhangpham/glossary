"""
Centralized configuration for the glossary generation system.

This module provides a unified configuration system that eliminates 
hardcoded values scattered across 12+ generation scripts.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory - project root
BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = BASE_DIR / "data"

@dataclass
class LevelConfig:
    """Configuration for a specific level (0, 1, 2, or 3)"""
    level: int
    name: str  # Human-readable name
    
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
            # Step 0 files have special naming patterns
            step_0_names = {
                0: "lv0_s0_college_names.txt",
                1: "lv1_s0_dept_names.txt", 
                2: "lv2_s0_research_areas.txt",
                3: "lv3_s0_conference_topics.txt"
            }
            return self.raw_dir / step_0_names[self.level]
        elif step == 1:
            return self.raw_dir / f"lv{self.level}_s{step-1}_extracted_concepts.txt" if step > 1 else self.get_step_input_file(0)
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

@dataclass 
class ProcessingConfig:
    """Configuration for processing parameters"""
    
    # LLM settings
    llm_attempts: int = 3
    concept_agreement_threshold: int = 2
    temperature: float = 0.3
    
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
    
    # Non-academic terms to filter out
    non_academic_terms: set = None
    
    # Additional configuration constants
    min_concept_length: int = 3
    max_concept_length: int = 50
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
        if self.non_academic_terms is None:
            self.non_academic_terms = {
                "university", "college", "school", "department", "center", "institute", 
                "program", "studies", "research", "science", "sciences", "technology",
                "administration", "management", "development", "international", "public",
                "general", "applied", "theoretical", "advanced", "basic", "modern",
                "clinical", "experimental", "social", "human", "natural", "physical"
            }

@dataclass
class ValidationConfig:
    """Configuration for validation and post-processing"""
    
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
    """Main configuration class for the entire glossary generation system"""
    
    def __init__(self):
        # Level configurations
        self.levels = {
            0: LevelConfig(0, "Colleges/Schools"),
            1: LevelConfig(1, "Departments"),  
            2: LevelConfig(2, "Research Areas"),
            3: LevelConfig(3, "Conference Topics")
        }
        
        # Processing configuration - can be overridden per level
        self.processing = ProcessingConfig()
        
        # Validation configuration
        self.validation = ValidationConfig()
        
        # Level-specific processing overrides
        self._setup_level_overrides()
        
    def _setup_level_overrides(self):
        """Setup level-specific processing parameter overrides"""
        self.level_overrides = {
            # Level 0 - Colleges (broadest, more conservative)
            0: {
                "batch_size": 20,
                "concept_agreement_threshold": 2,
                "keyword_appearance_threshold": 2,
            },
            # Level 1 - Departments (moderate)  
            1: {
                "batch_size": 15,
                "concept_agreement_threshold": 2,
                "keyword_appearance_threshold": 2,
            },
            # Level 2 - Research Areas (more specific)
            2: {
                "batch_size": 5,
                "concept_agreement_threshold": 3,  # Higher agreement needed
                "keyword_appearance_threshold": 1,
            },
            # Level 3 - Conference Topics (most specific)
            3: {
                "batch_size": 5,
                "concept_agreement_threshold": 3,
                "keyword_appearance_threshold": 1,
            }
        }
    
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
        return config
    
    def get_level_config(self, level: int) -> LevelConfig:
        """Get level configuration"""
        if level not in self.levels:
            raise ValueError(f"Invalid level: {level}. Must be 0, 1, 2, or 3")
        return self.levels[level]
        
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
    """Get configuration for a specific level"""
    return config.get_level_config(level)

def get_processing_config(level: int) -> ProcessingConfig:
    """Get processing configuration for a specific level"""
    return config.get_processing_config(level)

def ensure_directories(level: int = None):
    """Ensure required directories exist"""
    return config.ensure_directories(level)

# Environment variable configuration
class EnvConfig:
    """Environment-based configuration"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") 
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
    
    # Optional overrides from environment
    BATCH_SIZE = os.getenv("GLOSSARY_BATCH_SIZE")
    MAX_WORKERS = os.getenv("GLOSSARY_MAX_WORKERS") 
    CONCEPT_AGREEMENT_THRESHOLD = os.getenv("GLOSSARY_CONCEPT_AGREEMENT_THRESHOLD")
    
    @classmethod
    def apply_env_overrides(cls, processing_config: ProcessingConfig) -> ProcessingConfig:
        """Apply environment variable overrides to processing configuration"""
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

# Apply environment overrides to global config
for level in config.levels.keys():
    level_config = config.get_processing_config(level)
    EnvConfig.apply_env_overrides(level_config)