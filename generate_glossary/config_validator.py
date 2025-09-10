"""
Configuration validation and startup checks for the glossary generation system.

This module provides comprehensive validation for all configuration sections
and offers utilities for testing configuration in different environments.
"""

import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import asdict

from generate_glossary.config import (
    config,
    validate_configuration,
    get_processing_config,
    get_step_config,
    get_llm_config,
    EnvConfig,
    DATA_DIR
)
from generate_glossary.utils.logger import get_logger

logger = get_logger("config_validator")


def validate_api_keys() -> List[str]:
    """
    Validate that required API keys are present.
    
    Returns:
        List of error messages (empty if all keys are present)
    """
    errors = []
    
    # Check for at least one LLM API key
    has_llm_key = False
    if EnvConfig.OPENAI_API_KEY:
        has_llm_key = True
        logger.info("✓ OpenAI API key found")
    else:
        logger.debug("✗ OpenAI API key not found")
    
    if EnvConfig.GEMINI_API_KEY:
        has_llm_key = True
        logger.info("✓ Gemini API key found")
    else:
        logger.debug("✗ Gemini API key not found")
    
    if EnvConfig.GOOGLE_API_KEY:
        has_llm_key = True
        logger.info("✓ Google API key found")
    else:
        logger.debug("✗ Google API key not found")
    
    if not has_llm_key:
        errors.append(
            "No LLM API keys found. Set at least one of: OPENAI_API_KEY, GEMINI_API_KEY, GOOGLE_API_KEY"
        )
    
    # Check optional API keys and provide warnings
    if not EnvConfig.TAVILY_API_KEY:
        logger.warning("⚠ Tavily API key not found (optional for web validation)")
    
    if not EnvConfig.RAPIDAPI_KEY:
        logger.warning("⚠ RapidAPI key not found (optional for web validation)")
    
    return errors


def validate_directories() -> List[str]:
    """
    Validate that required directories exist or can be created.
    
    Returns:
        List of error messages (empty if all directories are valid)
    """
    errors = []
    
    # Check base data directory - using centralized DATA_DIR for consistency
    data_dir = DATA_DIR
    if not data_dir.exists():
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Created data directory: {data_dir}")
        except Exception as e:
            errors.append(f"Cannot create data directory: {e}")
            return errors
    else:
        logger.info(f"✓ Data directory exists: {data_dir}")
    
    # Check level directories
    for level in range(4):
        level_config = config.get_level_config(level)
        try:
            # Ensure level directories exist
            level_config.data_dir.mkdir(parents=True, exist_ok=True)
            level_config.raw_dir.mkdir(parents=True, exist_ok=True)
            level_config.postprocessed_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"✓ Level {level} directories ready")
        except Exception as e:
            errors.append(f"Cannot create directories for level {level}: {e}")
    
    return errors


def validate_level_configs() -> List[str]:
    """
    Validate level-specific configurations.
    
    Returns:
        List of error messages (empty if all level configs are valid)
    """
    errors = []
    
    for level in range(4):
        try:
            # Get and validate level config
            level_config = config.get_level_config(level)
            if not level_config.name:
                errors.append(f"Level {level} missing name")
            
            # Get and validate step config
            step_config = config.get_step_config(level)
            if step_config:
                if step_config.batch_size <= 0:
                    errors.append(f"Level {level} batch_size must be positive")
                if step_config.agreement_threshold <= 0:
                    errors.append(f"Level {level} agreement_threshold must be positive")
                if not step_config.search_patterns:
                    errors.append(f"Level {level} missing search_patterns")
                if not step_config.quality_keywords:
                    errors.append(f"Level {level} missing quality_keywords")
            else:
                logger.warning(f"Level {level} has no step configuration")
            
            # Get and validate processing config
            processing_config = get_processing_config(level)
            if processing_config.batch_size <= 0:
                errors.append(f"Level {level} processing batch_size must be positive")
            
        except Exception as e:
            errors.append(f"Error validating level {level}: {e}")
    
    return errors


def validate_llm_config() -> List[str]:
    """
    Validate LLM configuration.
    
    Returns:
        List of error messages (empty if LLM config is valid)
    """
    errors = []
    
    llm_config = get_llm_config()
    
    # Check model tiers
    if not llm_config.model_tiers:
        errors.append("LLM model_tiers cannot be empty")
    else:
        # Validate each tier has at least one model
        for tier, models in llm_config.model_tiers.items():
            if not models:
                errors.append(f"LLM tier '{tier}' has no models")
    
    # Check default tier
    if llm_config.default_tier not in llm_config.model_tiers:
        errors.append(
            f"LLM default_tier '{llm_config.default_tier}' not in model_tiers"
        )
    
    # Validate temperature
    if not 0 <= llm_config.temperature <= 2:
        errors.append("LLM temperature must be between 0 and 2")
    
    # Validate retry settings
    if llm_config.max_retries < 0:
        errors.append("LLM max_retries must be non-negative")
    if llm_config.retry_delay < 0:
        errors.append("LLM retry_delay must be non-negative")
    
    return errors


def check_environment_overrides() -> Dict[str, Any]:
    """
    Check which configuration values are being overridden by environment variables.
    
    Returns:
        Dictionary of overridden values
    """
    overrides = {}
    
    # Check processing config overrides
    for key in ['batch_size', 'max_workers', 'temperature', 'chunk_size']:
        env_key = f"GLOSSARY_PROCESSING_{key.upper()}"
        if os.getenv(env_key):
            overrides[env_key] = os.getenv(env_key)
    
    # Check LLM config overrides
    for key in ['default_tier', 'temperature', 'max_retries', 'cache_ttl']:
        env_key = f"GLOSSARY_LLM_{key.upper()}"
        if os.getenv(env_key):
            overrides[env_key] = os.getenv(env_key)
    
    # Check validation config overrides
    for key in ['search_provider', 'max_concurrent_mining', 'similarity_threshold']:
        env_key = f"GLOSSARY_VALIDATION_{key.upper()}"
        if os.getenv(env_key):
            overrides[env_key] = os.getenv(env_key)
    
    # Check legacy overrides
    if EnvConfig.BATCH_SIZE:
        overrides["GLOSSARY_BATCH_SIZE (legacy)"] = EnvConfig.BATCH_SIZE
    if EnvConfig.MAX_WORKERS:
        overrides["GLOSSARY_MAX_WORKERS (legacy)"] = EnvConfig.MAX_WORKERS
    if EnvConfig.CONCEPT_AGREEMENT_THRESHOLD:
        overrides["GLOSSARY_CONCEPT_AGREEMENT_THRESHOLD (legacy)"] = EnvConfig.CONCEPT_AGREEMENT_THRESHOLD
    
    return overrides


def print_configuration_summary():
    """Print a summary of the current configuration."""
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    
    # Processing config
    processing_config = get_processing_config(0)  # Use level 0 as default
    print("\nProcessing Configuration:")
    print(f"  Batch Size: {processing_config.batch_size}")
    print(f"  Max Workers: {processing_config.max_workers}")
    print(f"  Temperature: {processing_config.temperature}")
    print(f"  LLM Attempts: {processing_config.llm_attempts}")
    print(f"  Chunk Size: {processing_config.chunk_size}")
    
    # LLM config
    llm_config = get_llm_config()
    print("\nLLM Configuration:")
    print(f"  Default Tier: {llm_config.default_tier}")
    print(f"  Available Tiers: {list(llm_config.model_tiers.keys())}")
    print(f"  Temperature: {llm_config.temperature}")
    print(f"  Cache TTL: {llm_config.cache_ttl}s")
    print(f"  Max Retries: {llm_config.max_retries}")
    
    # Level configurations
    print("\nLevel Configurations:")
    for level in range(4):
        level_config = config.get_level_config(level)
        step_config = config.get_step_config(level)
        print(f"  Level {level} ({level_config.name}):")
        if step_config:
            print(f"    Batch Size: {step_config.batch_size}")
            print(f"    Agreement Threshold: {step_config.agreement_threshold}")
            print(f"    Frequency Threshold: {step_config.frequency_threshold}")
    
    # Environment overrides
    overrides = check_environment_overrides()
    if overrides:
        print("\nEnvironment Overrides:")
        for key, value in overrides.items():
            print(f"  {key}: {value}")
    
    print("\n" + "="*60)


def validate_all(verbose: bool = False) -> bool:
    """
    Run all validation checks.
    
    Args:
        verbose: If True, print detailed output
        
    Returns:
        True if all validation passes, False otherwise
    """
    all_errors = []
    
    if verbose:
        print("\n" + "="*60)
        print("CONFIGURATION VALIDATION")
        print("="*60)
    
    # Run validation checks
    if verbose:
        print("\n1. Validating API keys...")
    api_errors = validate_api_keys()
    all_errors.extend(api_errors)
    
    if verbose:
        print("\n2. Validating directories...")
    dir_errors = validate_directories()
    all_errors.extend(dir_errors)
    
    if verbose:
        print("\n3. Validating level configurations...")
    level_errors = validate_level_configs()
    all_errors.extend(level_errors)
    
    if verbose:
        print("\n4. Validating LLM configuration...")
    llm_errors = validate_llm_config()
    all_errors.extend(llm_errors)
    
    if verbose:
        print("\n5. Validating base configuration...")
    config_errors = validate_configuration()
    all_errors.extend(config_errors)
    
    # Report results
    if all_errors:
        if verbose:
            print("\n" + "="*60)
            print("VALIDATION ERRORS:")
            print("="*60)
            for error in all_errors:
                print(f"  ✗ {error}")
            print("\n" + "="*60)
        return False
    else:
        if verbose:
            print("\n" + "="*60)
            print("✓ All validation checks passed!")
            print("="*60)
        return True


def startup_validation():
    """
    Run validation at application startup.
    
    This function should be called at the beginning of any main script
    to ensure configuration is valid before processing begins.
    
    Exits with error code 1 if validation fails.
    """
    logger.info("Running startup configuration validation...")
    
    if not validate_all(verbose=False):
        # Get all errors for detailed reporting
        all_errors = []
        all_errors.extend(validate_api_keys())
        all_errors.extend(validate_directories())
        all_errors.extend(validate_level_configs())
        all_errors.extend(validate_llm_config())
        all_errors.extend(validate_configuration())
        
        logger.error("Configuration validation failed!")
        logger.error("Errors found:")
        for error in all_errors:
            logger.error(f"  - {error}")
        
        print("\n" + "="*60)
        print("CONFIGURATION ERROR")
        print("="*60)
        print("\nThe following configuration errors must be fixed:")
        for error in all_errors:
            print(f"  ✗ {error}")
        print("\nPlease fix these errors and try again.")
        print("="*60 + "\n")
        
        sys.exit(1)
    
    logger.info("✓ Configuration validation passed")


def test_configuration():
    """
    Test configuration in the current environment.
    
    This is a utility function for testing and debugging configuration.
    """
    print("\n" + "="*80)
    print("GLOSSARY GENERATION CONFIGURATION TEST")
    print("="*80)
    
    # Run validation
    valid = validate_all(verbose=True)
    
    # Print configuration summary
    print_configuration_summary()
    
    # Print final status
    print("\n" + "="*80)
    if valid:
        print("✓ CONFIGURATION TEST PASSED")
    else:
        print("✗ CONFIGURATION TEST FAILED")
    print("="*80 + "\n")
    
    return valid


def migrate_old_config(old_config_path: str, dry_run: bool = True) -> Dict[str, Any]:
    """
    Helper to migrate from old configuration patterns.
    
    Args:
        old_config_path: Path to old configuration file
        dry_run: If True, don't make changes, just show what would be done
        
    Returns:
        Dictionary of migration actions
    """
    migration = {
        "actions": [],
        "warnings": [],
        "environment_vars": {}
    }
    
    # Check if old config exists
    old_path = Path(old_config_path)
    if not old_path.exists():
        migration["warnings"].append(f"Old config file not found: {old_config_path}")
        return migration
    
    # Read old config (assuming it's a Python file with constants)
    try:
        with open(old_path, 'r') as f:
            content = f.read()
        
        # Look for common patterns
        if "BATCH_SIZE = " in content:
            migration["environment_vars"]["GLOSSARY_PROCESSING_BATCH_SIZE"] = "20"
            migration["actions"].append("Migrate BATCH_SIZE to GLOSSARY_PROCESSING_BATCH_SIZE")
        
        if "MAX_WORKERS = " in content:
            migration["environment_vars"]["GLOSSARY_PROCESSING_MAX_WORKERS"] = "4"
            migration["actions"].append("Migrate MAX_WORKERS to GLOSSARY_PROCESSING_MAX_WORKERS")
        
        if "TEMPERATURE = " in content:
            migration["environment_vars"]["GLOSSARY_PROCESSING_TEMPERATURE"] = "1.0"
            migration["actions"].append("Migrate TEMPERATURE to GLOSSARY_PROCESSING_TEMPERATURE")
        
        if "MODEL_TIER = " in content or "DEFAULT_TIER = " in content:
            migration["environment_vars"]["GLOSSARY_LLM_DEFAULT_TIER"] = "budget"
            migration["actions"].append("Migrate MODEL_TIER to GLOSSARY_LLM_DEFAULT_TIER")
        
    except Exception as e:
        migration["warnings"].append(f"Error reading old config: {e}")
    
    # Apply migration if not dry run
    if not dry_run:
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file, 'a') as f:
                f.write("\n# Migrated configuration\n")
                for key, value in migration["environment_vars"].items():
                    f.write(f"{key}={value}\n")
                    migration["actions"].append(f"Added {key}={value} to .env")
    
    return migration


if __name__ == "__main__":
    # Run configuration test when module is executed directly
    test_configuration()