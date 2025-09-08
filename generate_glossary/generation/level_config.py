"""
Level-specific configuration for generation pipeline.

This module centralizes all the parameter differences between levels 1-3,
allowing the shared functional utilities to be completely generic.
"""

from dataclasses import dataclass
from typing import List, Union


@dataclass
class StepConfig:
    """Configuration for a generation level."""
    batch_size: int
    agreement_threshold: int
    search_patterns: List[str]
    quality_keywords: List[str]
    frequency_threshold: Union[float, str]
    processing_description: str
    context_description: str


# Level-specific configurations
LEVEL_CONFIGS = {
    1: StepConfig(
        batch_size=15,
        agreement_threshold=2,
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
    
    2: StepConfig(
        batch_size=5,
        agreement_threshold=3,
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
    
    3: StepConfig(
        batch_size=5,
        agreement_threshold=3,
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


def get_level_config(level: int) -> StepConfig:
    """Get configuration for a specific level."""
    if level not in LEVEL_CONFIGS:
        raise ValueError(f"No configuration found for level {level}")
    return LEVEL_CONFIGS[level]


def get_step_file_paths(level: int, step: str) -> tuple[str, str, str]:
    """Get input, output, and metadata file paths for a step."""
    data_dir = f"data/lv{level}"
    
    if step == "s0":
        if level == 1:
            input_file = "data/lv0/lv0_final.txt"
        else:
            input_file = f"data/lv{level-1}/lv{level-1}_final.txt"
        
        # Step 0 output varies by level
        step_names = {1: "department_names", 2: "research_areas", 3: "conference_topics"}
        output_file = f"{data_dir}/raw/lv{level}_s0_{step_names[level]}.txt"
        
    elif step == "s1":
        step_names = {1: "department_names", 2: "research_areas", 3: "conference_topics"}  
        input_file = f"{data_dir}/raw/lv{level}_s0_{step_names[level]}.txt"
        output_file = f"{data_dir}/raw/lv{level}_s1_extracted_concepts.txt"
        
    elif step == "s2":
        input_file = f"{data_dir}/raw/lv{level}_s1_extracted_concepts.txt"  
        output_file = f"{data_dir}/raw/lv{level}_s2_filtered_concepts.txt"
        
    elif step == "s3":
        input_file = f"{data_dir}/raw/lv{level}_s2_filtered_concepts.txt"
        output_file = f"{data_dir}/raw/lv{level}_s3_verified_concepts.txt"
    
    else:
        raise ValueError(f"Unknown step: {step}")
    
    metadata_file = output_file.replace(".txt", "_metadata.json")
    return input_file, output_file, metadata_file