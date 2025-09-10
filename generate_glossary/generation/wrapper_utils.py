"""Shared utilities for level-specific step wrappers to eliminate code duplication."""

import argparse
import sys
import importlib
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict, Any

from ..utils.logger import get_logger

logger = get_logger(__name__)


def to_test_path(file_path: str, level: int) -> str:
    """Convert production path to test path for development."""
    p = Path(file_path)
    needle = f"data/lv{level}"
    return str(Path(str(p).replace(needle, f"data/test/lv{level}", 1)))


def setup_step_wrapper(level: int, step: str) -> dict:
    """Setup common configuration for step wrappers."""
    return {
        "level": level,
        "step": step,
        "test_mode": "--test" in sys.argv or "-t" in sys.argv
    }


def create_cli_wrapper(
    level: int,
    step: str, 
    main_func: Callable,
    test_func: Optional[Callable] = None,
    extra_args: Optional[Callable] = None
) -> None:
    """Create a CLI wrapper with common argument parsing."""
    parser = argparse.ArgumentParser(description=f"Level {level} Step {step}")
    
    # Common arguments
    parser.add_argument("--test", "-t", action="store_true", help="Use test data")
    parser.add_argument("--provider", default="openai", help="LLM provider")
    
    # Add extra arguments if provided
    if extra_args:
        extra_args(parser)
    
    args = parser.parse_args()
    
    # Convert args to dict
    kwargs = vars(args)
    test_flag = kwargs.pop("test", None)
    
    # Handle test mode
    if test_flag and test_func:
        # Pass through other CLI args to test function
        test_func(**kwargs)
    else:
        # Call main function with remaining kwargs
        main_func(**kwargs)


def handle_dry_run(dry_run: bool, func: Callable, *args, **kwargs):
    """Handle dry-run mode for functions that support it."""
    if dry_run:
        logger.info("DRY RUN MODE - No actual processing will occur")
        return {"dry_run": True, "would_process": True}
    return func(*args, **kwargs)


def validate_input_file(file_path: str) -> bool:
    """
    Validate that input file exists.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        True if file exists, False otherwise (with logged error)
    """
    if not Path(file_path).exists():
        logger.error(f"Input file not found: {file_path}")
        return False
    return True


def get_step_file_paths(level: int, step: int) -> tuple[str, str, str]:
    """
    Get the input, output, and metadata file paths for a given level and step.
    
    Args:
        level: Level number (0-3)
        step: Step number (1-3)
        
    Returns:
        Tuple of (input_path, output_path, metadata_path)
    """
    from ..config import get_level_config
    
    level_config = get_level_config(level)
    input_path = str(level_config.get_step_input_file(step))
    output_path = str(level_config.get_step_output_file(step))
    metadata_path = str(level_config.get_step_metadata_file(step))
    
    return input_path, output_path, metadata_path


def create_s1_wrapper(level: int):
    """
    Create wrapper for step 1 (concept extraction).
    
    Returns:
        Tuple of (main_func, test_func) callables
    """
    from .concept_extraction import extract_concepts_llm
    
    def main(provider: str = "openai", **kwargs):
        """Main function for step 1."""
        input_path, output_path, metadata_path = get_step_file_paths(level, 1)
        
        if not validate_input_file(input_path):
            return 1
        
        result = extract_concepts_llm(
            input_file=input_path,
            output_file=output_path,
            level=level,
            metadata_file=metadata_path,
            provider=provider
        )
        return 0 if is_success(result) else 1
    
    def test(provider: str = "openai", **kwargs):
        """Test function for step 1."""
        input_path, output_path, metadata_path = get_step_file_paths(level, 1)
        input_path = to_test_path(input_path, level)
        output_path = to_test_path(output_path, level)
        metadata_path = to_test_path(metadata_path, level)
        
        if not validate_input_file(input_path):
            return 1
        
        result = extract_concepts_llm(
            input_file=input_path,
            output_file=output_path,
            level=level,
            metadata_file=metadata_path,
            provider=provider
        )
        return 0 if is_success(result) else 1
    
    return main, test


def create_s2_wrapper(level: int):
    """
    Create wrapper for step 2 (frequency filtering).
    
    Returns:
        Tuple of (main_func, test_func) callables
    """
    from .frequency_filtering import filter_by_frequency
    
    def main(**kwargs):
        """Main function for step 2."""
        input_path, output_path, metadata_path = get_step_file_paths(level, 2)
        
        if not validate_input_file(input_path):
            return 1
        
        result = filter_by_frequency(
            input_file=input_path,
            output_file=output_path,
            level=level,
            metadata_file=metadata_path
        )
        return 0 if is_success(result) else 1
    
    def test(**kwargs):
        """Test function for step 2."""
        input_path, output_path, metadata_path = get_step_file_paths(level, 2)
        input_path = to_test_path(input_path, level)
        output_path = to_test_path(output_path, level)
        metadata_path = to_test_path(metadata_path, level)
        
        if not validate_input_file(input_path):
            return 1
        
        result = filter_by_frequency(
            input_file=input_path,
            output_file=output_path,
            level=level,
            metadata_file=metadata_path
        )
        return 0 if is_success(result) else 1
    
    return main, test


def create_s3_wrapper(level: int):
    """
    Create wrapper for step 3 (token verification).
    
    Returns:
        Tuple of (main_func, test_func) callables
    """
    from .token_verification import verify_single_tokens
    
    def main(provider: str = None, **kwargs):
        """Main function for step 3."""
        input_path, output_path, metadata_path = get_step_file_paths(level, 3)
        
        if not validate_input_file(input_path):
            return 1
        
        result = verify_single_tokens(
            input_file=input_path,
            output_file=output_path,
            level=level,
            metadata_file=metadata_path,
            provider=provider
        )
        return 0 if is_success(result) else 1
    
    def test(provider: str = None, **kwargs):
        """Test function for step 3."""
        input_path, output_path, metadata_path = get_step_file_paths(level, 3)
        input_path = to_test_path(input_path, level)
        output_path = to_test_path(output_path, level)
        metadata_path = to_test_path(metadata_path, level)
        
        if not validate_input_file(input_path):
            return 1
        
        result = verify_single_tokens(
            input_file=input_path,
            output_file=output_path,
            level=level,
            metadata_file=metadata_path,
            provider=provider
        )
        return 0 if is_success(result) else 1
    
    return main, test


def is_success(result) -> bool:
    """Helper function to determine if a step execution was successful."""
    if result is None or result is True:
        return True
    if isinstance(result, dict):
        if result.get('success') is True:
            return True
        # For Step 0 only: check processed_terms_count
        if result.get('processed_terms_count', 0) > 0:
            return True
    return False


def create_s0_wrapper(level: int):
    """
    Create wrapper for step 0 (data extraction).
    
    For Level 0: Uses Excel parsing (not web extraction)
    For other levels: Uses web extraction
    
    Returns:
        Tuple of (main_func, test_func) callables
    """
    def main(input_file: Optional[str] = None, **kwargs):
        """Main function for step 0."""
        if level == 0:
            # Level 0 uses Excel parsing, not web extraction
            try:
                from .lv0.lv0_s0_get_college_names import main as lv0_main
                result = lv0_main()
                return 0 if is_success(result) else 1
            except ImportError:
                raise RuntimeError(
                    "Level 0 Step 0 requires Excel parsing functionality. "
                    "Please ensure lv0_s0_get_college_names module is available."
                )
        else:
            # Other levels use web extraction
            from .web_extraction_firecrawl import extract_web_content
            from ..config import get_level_config
            
            level_config = get_level_config(level)
            
            # Use input_file override if provided, otherwise get from config
            if input_file:
                input_path = input_file
            else:
                input_path = str(level_config.get_step_input_file(0))
            
            output_path = str(level_config.get_step_output_file(0))
            metadata_path = str(level_config.get_step_metadata_file(0))
            
            if not validate_input_file(input_path):
                return 1
            
            result = extract_web_content(
                input_file=input_path,
                level=level,
                output_file=output_path,
                metadata_file=metadata_path
            )
            return 0 if is_success(result) else 1
    
    def test(input_file: Optional[str] = None, **kwargs):
        """Test function for step 0."""
        if level == 0:
            # Level 0 uses Excel parsing in test mode
            try:
                from .lv0.lv0_s0_get_college_names import test as lv0_test
                result = lv0_test()
                return 0 if is_success(result) else 1
            except ImportError:
                raise RuntimeError(
                    "Level 0 Step 0 requires Excel parsing functionality. "
                    "Please ensure lv0_s0_get_college_names module is available."
                )
        else:
            # Other levels use web extraction in test mode
            from .web_extraction_firecrawl import extract_web_content
            from ..config import get_level_config
            
            level_config = get_level_config(level)
            
            # Use input_file override if provided, otherwise get from config
            if input_file:
                input_path = to_test_path(input_file, level)
            else:
                input_path = str(level_config.get_step_input_file(0))
                input_path = to_test_path(input_path, level)
            
            output_path = str(level_config.get_step_output_file(0))
            metadata_path = str(level_config.get_step_metadata_file(0))
            
            output_path = to_test_path(output_path, level)
            metadata_path = to_test_path(metadata_path, level)
            
            if not validate_input_file(input_path):
                return 1
            
            result = extract_web_content(
                input_file=input_path,
                level=level,
                output_file=output_path,
                metadata_file=metadata_path
            )
            return 0 if is_success(result) else 1
    
    return main, test


def discover_step_function(level: int, step: int) -> Tuple[Callable, Callable]:
    """
    Dynamically discover and create appropriate wrapper function for any level/step combination.
    
    Args:
        level: Generation level (0-3)
        step: Step number (0-3)
        
    Returns:
        Tuple of (main_func, test_func) callables
        
    Raises:
        ValueError: If level or step is invalid
    """
    if level not in range(4):
        raise ValueError(f"Invalid level: {level}. Must be 0, 1, 2, or 3")
    if step not in range(4):
        raise ValueError(f"Invalid step: {step}. Must be 0, 1, 2, or 3")
    
    wrapper_map = {
        0: create_s0_wrapper,
        1: create_s1_wrapper,
        2: create_s2_wrapper,
        3: create_s3_wrapper
    }
    
    wrapper_func = wrapper_map[step]
    return wrapper_func(level)


def validate_step_dependencies(level: int, step: int, input_file: Optional[str] = None, create_dirs: bool = False) -> Dict[str, Any]:
    """
    Validate that all dependencies for a step are met.
    
    Args:
        level: Generation level (0-3)
        step: Step number (0-3)
        input_file: Optional override for input file (used for Step 0)
        create_dirs: Whether to create output directories (default: False for read-only validation)
        
    Returns:
        Dictionary with validation results and details
    """
    from ..config import get_level_config
    
    try:
        level_config = get_level_config(level)
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'level': level,
            'step': step
        }
        
        # Check if input file exists
        try:
            if input_file:
                # Use override input file
                if not Path(input_file).exists():
                    validation_result['valid'] = False
                    validation_result['errors'].append(f"Input file override not found: {input_file}")
            else:
                # Use config-determined input file
                config_input_file = level_config.get_step_input_file(step)
                if not Path(config_input_file).exists():
                    validation_result['valid'] = False
                    validation_result['errors'].append(f"Input file not found: {config_input_file}")
        except NotImplementedError as e:
            # Level 0, Step 0 is handled specially
            if level == 0 and step == 0:
                if input_file is None:
                    # No input file override provided for Level 0 Step 0
                    validation_result['valid'] = False
                    validation_result['errors'].append(f"Level 0 Step 0 requires Excel input file or input file override: {str(e)}")
                validation_result['warnings'].append(f"Level 0 Step 0 uses Excel parsing, not web extraction: {str(e)}")
            else:
                validation_result['valid'] = False
                validation_result['errors'].append(str(e))
        
        # Check if previous step is completed (for steps > 0)
        if step > 0:
            prev_output_file = level_config.get_step_output_file(step - 1)
            if not Path(prev_output_file).exists():
                validation_result['valid'] = False
                validation_result['errors'].append(f"Previous step output not found: {prev_output_file}")
        
        # Check output directory exists and is writable
        output_file = level_config.get_step_output_file(step)
        output_dir = Path(output_file).parent
        if not output_dir.exists():
            if create_dirs:
                output_dir.mkdir(parents=True, exist_ok=True)
                validation_result['warnings'].append(f"Created output directory: {output_dir}")
            else:
                validation_result['warnings'].append(f"Output directory needs creation: {output_dir}")
        
        return validation_result
        
    except Exception as e:
        return {
            'valid': False,
            'errors': [f"Validation failed: {str(e)}"],
            'warnings': [],
            'level': level,
            'step': step
        }


def get_step_parameters(level: int, step: int) -> Dict[str, Any]:
    """
    Automatically determine what parameters a step needs.
    
    Args:
        level: Generation level (0-3)
        step: Step number (0-3)
        
    Returns:
        Dictionary of parameter requirements and defaults
    """
    base_params = {
        'level': level,
        'step': step,
        'requires_provider': False,
        'default_provider': 'openai',
        'supports_test_mode': True
    }
    
    # Step-specific parameter requirements
    if step in [1, 3]:  # Concept extraction and token verification need LLM provider
        base_params['requires_provider'] = True
    
    return base_params


def create_generic_step_runner(level: int, step: int) -> Callable:
    """
    Create a generic step runner that works for all levels and steps.
    
    Args:
        level: Generation level (0-3)
        step: Step number (0-3)
        
    Returns:
        Generic step runner function
    """
    def run_step(**kwargs):
        """Generic step runner that handles any level/step combination."""
        # Validate dependencies first (with directory creation enabled)
        input_file_override = kwargs.get('input_file')
        validation = validate_step_dependencies(level, step, input_file=input_file_override, create_dirs=True)
        if not validation['valid']:
            logger.error(f"Step validation failed for Level {level} Step {step}")
            for error in validation['errors']:
                logger.error(f"  - {error}")
            return 1
        
        # Log warnings
        for warning in validation['warnings']:
            logger.warning(f"  - {warning}")
        
        # Get step functions
        try:
            main_func, test_func = discover_step_function(level, step)
        except Exception as e:
            logger.error(f"Failed to discover step function for Level {level} Step {step}: {str(e)}")
            return 1
        
        # Determine test mode
        test_mode = kwargs.get('test', False) or '--test' in sys.argv or '-t' in sys.argv
        
        # Run appropriate function
        try:
            if test_mode:
                return test_func(**kwargs)
            else:
                return main_func(**kwargs)
        except Exception as e:
            logger.error(f"Step execution failed for Level {level} Step {step}: {str(e)}")
            return 1
    
    return run_step