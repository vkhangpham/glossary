"""
Step discovery and validation module for the dynamic CLI system.

This module provides comprehensive step discovery, dependency validation, and 
metadata functions to support the modernized CLI system that eliminates 
hardcoded step mappings.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from ..utils.logger import get_logger
from ..config import get_level_config

logger = get_logger(__name__)


def discover_available_steps(level: int) -> Dict[int, Dict[str, Any]]:
    """
    Discover all available steps for a given level.
    
    Args:
        level: Generation level (0-3)
        
    Returns:
        Dictionary mapping step numbers to their metadata
    """
    from .wrapper_utils import get_step_parameters
    
    try:
        level_config = get_level_config(level)
        available_steps = {}
        
        # Standard steps for all levels
        for step in range(4):  # Steps 0, 1, 2, 3
            try:
                step_params = get_step_parameters(level, step)
                step_metadata = {
                    'step': step,
                    'level': level,
                    'description': _get_step_description(level, step),
                    'requires_provider': step_params['requires_provider'],
                    'default_provider': step_params['default_provider'],
                    'supports_test_mode': step_params['supports_test_mode'],
                    'input_file': None,
                    'output_file': None,
                    'metadata_file': None
                }
                
                # Try to get file paths
                try:
                    step_metadata['input_file'] = str(level_config.get_step_input_file(step))
                except (NotImplementedError, Exception):
                    step_metadata['input_file'] = f"Special handling for Level {level} Step {step}"
                
                try:
                    step_metadata['output_file'] = str(level_config.get_step_output_file(step))
                    step_metadata['metadata_file'] = str(level_config.get_step_metadata_file(step))
                except Exception:
                    step_metadata['output_file'] = None
                    step_metadata['metadata_file'] = None
                
                available_steps[step] = step_metadata
                
            except Exception as e:
                logger.warning(f"Could not discover step {step} for level {level}: {str(e)}")
                continue
        
        return available_steps
        
    except Exception as e:
        logger.error(f"Failed to discover steps for level {level}: {str(e)}")
        return {}


def _get_step_description(level: int, step: int) -> str:
    """
    Get human-readable description for a step.
    
    Args:
        level: Generation level (0-3)
        step: Step number (0-3)
        
    Returns:
        Human-readable step description
    """
    level_names = {
        0: "College/School",
        1: "Department", 
        2: "Research Area",
        3: "Conference Topic"
    }
    
    step_descriptions = {
        0: "Extract {} names from Excel data" if level == 0 else "Extract {} names from web sources",
        1: "Extract concepts from {} data using LLM",
        2: "Filter concepts by frequency and quality",
        3: "Verify single tokens using LLM"
    }
    
    level_name = level_names.get(level, f"Level {level}")
    step_desc = step_descriptions.get(step, f"Step {step}")
    
    return step_desc.format(level_name)


def get_step_metadata(level: int, step: int) -> Dict[str, Any]:
    """
    Get comprehensive metadata for a specific step.
    
    Args:
        level: Generation level (0-3)
        step: Step number (0-3)
        
    Returns:
        Dictionary containing step metadata
    """
    from .wrapper_utils import get_step_parameters, validate_step_dependencies
    
    try:
        level_config = get_level_config(level)
        step_params = get_step_parameters(level, step)
        validation = validate_step_dependencies(level, step, create_dirs=False)
        
        metadata = {
            'level': level,
            'step': step,
            'description': _get_step_description(level, step),
            'parameters': step_params,
            'validation': validation,
            'file_paths': {},
            'dependencies': _get_step_dependencies(level, step)
        }
        
        # Get file paths
        try:
            metadata['file_paths']['input'] = str(level_config.get_step_input_file(step))
        except (NotImplementedError, Exception) as e:
            metadata['file_paths']['input'] = f"Special handling: {str(e)}"
            
        try:
            metadata['file_paths']['output'] = str(level_config.get_step_output_file(step))
            metadata['file_paths']['metadata'] = str(level_config.get_step_metadata_file(step))
        except Exception:
            metadata['file_paths']['output'] = None
            metadata['file_paths']['metadata'] = None
        
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to get metadata for Level {level} Step {step}: {str(e)}")
        return {
            'level': level,
            'step': step,
            'description': f"Error getting metadata: {str(e)}",
            'error': str(e)
        }


def _get_step_dependencies(level: int, step: int) -> List[str]:
    """
    Get list of dependencies for a step.
    
    Args:
        level: Generation level (0-3)
        step: Step number (0-3)
        
    Returns:
        List of dependency descriptions
    """
    dependencies = []
    
    # Step 0 dependencies
    if step == 0:
        if level == 0:
            dependencies.append("Excel file with college data (handled by specialized script)")
        else:
            dependencies.append(f"Level {level-1} final file (lv{level-1}_final.txt)")
    
    # Steps 1-3 depend on previous step
    elif step > 0:
        dependencies.append(f"Level {level} Step {step-1} output file")
    
    # LLM provider dependencies
    if step in [1, 3]:
        dependencies.append("LLM provider (OpenAI API key or alternative)")
    
    # Firecrawl dependencies for step 0
    if step == 0:
        dependencies.append("Firecrawl API key for web extraction")
    
    return dependencies


def validate_all_steps(level: int) -> Dict[str, Any]:
    """
    Validate all steps for a given level.
    
    Args:
        level: Generation level (0-3)
        
    Returns:
        Comprehensive validation report
    """
    from .wrapper_utils import validate_step_dependencies
    
    validation_report = {
        'level': level,
        'overall_valid': True,
        'steps': {},
        'summary': {
            'total_steps': 0,
            'valid_steps': 0,
            'invalid_steps': 0,
            'steps_with_warnings': 0
        },
        'recommendations': []
    }
    
    try:
        available_steps = discover_available_steps(level)
        validation_report['summary']['total_steps'] = len(available_steps)
        
        for step_num in available_steps:
            step_validation = validate_step_dependencies(level, step_num, create_dirs=False)
            validation_report['steps'][step_num] = step_validation
            
            if step_validation['valid']:
                validation_report['summary']['valid_steps'] += 1
            else:
                validation_report['summary']['invalid_steps'] += 1
                validation_report['overall_valid'] = False
            
            if step_validation.get('warnings'):
                validation_report['summary']['steps_with_warnings'] += 1
        
        # Generate recommendations
        validation_report['recommendations'] = _generate_validation_recommendations(
            level, validation_report
        )
        
        return validation_report
        
    except Exception as e:
        logger.error(f"Failed to validate steps for level {level}: {str(e)}")
        return {
            'level': level,
            'overall_valid': False,
            'error': str(e),
            'summary': {'total_steps': 0, 'valid_steps': 0, 'invalid_steps': 0}
        }


def _generate_validation_recommendations(level: int, report: Dict[str, Any]) -> List[str]:
    """
    Generate recommendations based on validation results.
    
    Args:
        level: Generation level (0-3)
        report: Validation report
        
    Returns:
        List of recommendations
    """
    recommendations = []
    
    # Check for missing input files
    for step_num, step_validation in report.get('steps', {}).items():
        if not step_validation['valid']:
            for error in step_validation.get('errors', []):
                if 'Input file not found' in error:
                    if step_num == 0 and level > 0:
                        recommendations.append(
                            f"Run Level {level-1} pipeline to completion to generate required input"
                        )
                    elif step_num > 0:
                        recommendations.append(
                            f"Complete Level {level} Step {step_num-1} before running Step {step_num}"
                        )
    
    # Check for API key requirements
    if any(step in report.get('steps', {}) for step in [0, 1, 3]):
        recommendations.append("Ensure FIRECRAWL_API_KEY is set for web extraction")
        recommendations.append("Ensure OpenAI API key is configured for LLM operations")
    
    # Check for directory structure
    if not Path(f"data/lv{level}").exists():
        recommendations.append(f"Create data directory structure: mkdir -p data/lv{level}/raw")
    
    return recommendations


def get_validation_summary(level: int, step: int) -> str:
    """
    Get a human-readable validation summary for a step.
    
    Args:
        level: Generation level (0-3)
        step: Step number (0-3)
        
    Returns:
        Formatted validation summary string
    """
    from .wrapper_utils import validate_step_dependencies
    
    validation = validate_step_dependencies(level, step, create_dirs=False)
    
    if validation['valid']:
        status = "✅ READY"
        details = []
        if validation.get('warnings'):
            details.extend([f"⚠️  {warning}" for warning in validation['warnings']])
    else:
        status = "❌ NOT READY"
        details = []
        if validation.get('errors'):
            details.extend([f"❌ {error}" for error in validation['errors']])
        if validation.get('warnings'):
            details.extend([f"⚠️  {warning}" for warning in validation['warnings']])
    
    summary = f"Level {level} Step {step}: {status}\n"
    summary += f"Description: {_get_step_description(level, step)}\n"
    
    if details:
        summary += "\nDetails:\n" + "\n".join(f"  {detail}" for detail in details)
    
    return summary


def list_all_steps() -> Dict[int, Dict[int, str]]:
    """
    List all available steps across all levels.
    
    Returns:
        Nested dictionary: {level: {step: description}}
    """
    all_steps = {}
    
    for level in range(4):  # Levels 0, 1, 2, 3
        try:
            available_steps = discover_available_steps(level)
            all_steps[level] = {
                step_num: step_data['description'] 
                for step_num, step_data in available_steps.items()
            }
        except Exception as e:
            logger.warning(f"Could not list steps for level {level}: {str(e)}")
            all_steps[level] = {}
    
    return all_steps