"""
Level 1 Generation Module

This module contains standalone scripts for generating Level 1 (Department) glossary entries.
Each script handles one step of the 4-step generation pipeline:

- Step 0: Extract department names from college websites (web extraction)
- Step 1: Extract academic concepts from department names (LLM extraction)
- Step 2: Filter concepts by institutional frequency (frequency filtering)
- Step 3: Verify single-token academic terms (token verification)

Each script can be run independently and includes both main() and test() functions.
"""

from pathlib import Path

# Module version
__version__ = "1.0.0"

# Level configuration
LEVEL = 1
LEVEL_NAME = "Department"

# Available steps
STEPS = {
    0: "lv1_s0_get_dept_names",
    1: "lv1_s1_extract_concepts",
    2: "lv1_s2_filter_by_freq",
    3: "lv1_s3_verify_tokens",
}

# Step descriptions
STEP_DESCRIPTIONS = {
    0: "Extract department names from college websites",
    1: "Extract academic concepts from department names",
    2: "Filter concepts by institutional frequency",
    3: "Verify single-token academic terms",
}

def get_step_module(step: int):
    """
    Get the module for a specific step.
    
    Args:
        step: Step number (0-3)
        
    Returns:
        Module name for the step
        
    Raises:
        ValueError: If step is invalid
    """
    if step not in STEPS:
        raise ValueError(f"Invalid step {step}. Must be 0-3.")
    return STEPS[step]

def get_step_description(step: int) -> str:
    """
    Get the description for a specific step.
    
    Args:
        step: Step number (0-3)
        
    Returns:
        Description of the step
        
    Raises:
        ValueError: If step is invalid
    """
    if step not in STEP_DESCRIPTIONS:
        raise ValueError(f"Invalid step {step}. Must be 0-3.")
    return STEP_DESCRIPTIONS[step]

# Export main functions from each step for convenience
from .lv1_s0_get_dept_names import main as s0_main, test as s0_test
from .lv1_s1_extract_concepts import main as s1_main, test as s1_test
from .lv1_s2_filter_by_freq import main as s2_main, test as s2_test
from .lv1_s3_verify_tokens import main as s3_main, test as s3_test

__all__ = [
    "LEVEL",
    "LEVEL_NAME",
    "STEPS",
    "STEP_DESCRIPTIONS",
    "get_step_module",
    "get_step_description",
    "s0_main",
    "s0_test",
    "s1_main",
    "s1_test",
    "s2_main",
    "s2_test",
    "s3_main",
    "s3_test",
]