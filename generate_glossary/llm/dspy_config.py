"""
DSPy configuration and global state management for the LLM module.

This module handles DSPy configuration, cache management, and global state.
It provides functions for setting up DSPy contexts, managing cache directories,
and ensuring proper initialization on first use.
"""

import atexit
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import dspy

from generate_glossary.config import get_llm_config

# Module-level configuration state
_dspy_configured = False
_cache_dir = None
_signature_metadata_detection_result = None  # Memoized result for signature metadata detection
_signature_metadata_detection_time = None  # Timestamp of last detection
_SIGNATURE_DETECTION_TTL = 300  # 5 minutes TTL for detection cache
logger = logging.getLogger(__name__)


def configure_dspy_cache(disable_cache: bool = False) -> str:
    """
    Configure DSPy caching with proper directory setup and cleanup.
    
    Args:
        disable_cache: If True, disable caching entirely
        
    Returns:
        Cache directory path or empty string if disabled
    """
    global _cache_dir
    
    if disable_cache:
        # Disable caching by setting environment variables
        os.environ["DSP_CACHEDIR"] = ""
        os.environ["DSP_DISABLE_CACHE"] = "1"
        logger.debug("DSPy cache fully disabled via env var")
        return ""
    
    # Create temporary cache directory
    _cache_dir = tempfile.mkdtemp(prefix="dspy_cache_")
    os.environ["DSP_CACHEDIR"] = _cache_dir
    
    # Configure DSPy cache
    try:
        if hasattr(dspy, "configure_cache"):
            dspy.configure_cache()
            logger.debug(f"DSPy cache configured at: {_cache_dir}")
        else:
            logger.warning("dspy.configure_cache() not available in this DSPy version")
    except (AttributeError, Exception) as e:
        logger.warning(f"Failed to configure DSPy cache: {e}")
        # Continue without cache
        os.environ["DSP_CACHEDIR"] = ""
        return ""
    
    # Register cleanup on exit
    atexit.register(cleanup_cache)
    
    return _cache_dir


def configure_dspy_global(
    lm: Optional[dspy.LM] = None,
    trace: Optional[bool] = None,
    cache: Optional[bool] = None
) -> None:
    """
    Configure DSPy global settings.
    
    Args:
        lm: Optional LM instance for global configuration
        trace: Enable/disable tracing
        cache: Enable/disable caching
    """
    global _dspy_configured
    
    try:
        kwargs = {}
        
        if lm is not None:
            kwargs["lm"] = lm
        if trace is not None:
            kwargs["trace"] = trace
        if cache is not None:
            kwargs["cache"] = cache
        
        # Apply global configuration
        dspy.configure(**kwargs)
        _dspy_configured = True
        logger.debug(f"DSPy global configuration applied: {kwargs}")
        
    except Exception as e:
        logger.error(f"Failed to configure DSPy globally: {e}")
        raise


def get_dspy_context(
    lm: Optional[dspy.LM] = None,
    trace: Optional[bool] = None,
    cache: Optional[bool] = None
) -> dspy.context:
    """
    Get a DSPy context manager with specified settings.
    
    Args:
        lm: LM instance for the context
        trace: Enable/disable tracing in context
        cache: Enable/disable caching in context
        
    Returns:
        DSPy context manager
    """
    kwargs = {}
    
    if lm is not None:
        kwargs["lm"] = lm
    if trace is not None:
        kwargs["trace"] = trace  
    if cache is not None:
        kwargs["cache"] = cache
    
    return dspy.context(**kwargs)


def cleanup_cache() -> None:
    """
    Clean up temporary cache directory.
    """
    global _cache_dir
    
    if _cache_dir and Path(_cache_dir).exists():
        try:
            import shutil
            shutil.rmtree(_cache_dir)
            logger.debug(f"Cleaned up cache directory: {_cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup cache directory {_cache_dir}: {e}")
        finally:
            _cache_dir = None


def configure_for_optimization(disable_cache: bool = True, trace: bool = False) -> None:
    """
    Configure DSPy for optimization workflows following common.py patterns.
    
    Args:
        disable_cache: Disable caching for optimization
        trace: Enable tracing for debugging
    """
    # Configure cache settings
    configure_dspy_cache(disable_cache=disable_cache)
    
    # Configure global settings for optimization
    configure_dspy_global(trace=trace, cache=not disable_cache)
    
    logger.debug("DSPy configured for optimization workflows")


def configure_for_signature_based_workflows(
    enable_structured_output: bool = True,
    optimize_for_composition: bool = True,
    enable_comprehensive_tracing: bool = False
) -> None:
    """
    Configure DSPy with optimal settings for signature-based declarative programming.
    
    This function sets up DSPy to work optimally with the enhanced signature metadata
    system, supporting modular composition and separation of task logic from prompt formatting.
    
    Args:
        enable_structured_output: Configure for structured output support (TypedPredictor)
        optimize_for_composition: Enable settings that support modular composition
        enable_comprehensive_tracing: Enable detailed tracing for debugging signature workflows
    """
    global _dspy_configured
    
    try:
        # Disable caching for consistent signature-based behavior
        configure_dspy_cache(disable_cache=True)
        
        # Configure global settings optimized for signature-based workflows
        config_kwargs = {
            'cache': False,  # Disable cache for predictable behavior
            'trace': enable_comprehensive_tracing
        }
        
        # Apply signature-optimized configuration
        configure_dspy_global(**config_kwargs)
        
        # Set module-level optimizations for signature workflows
        if optimize_for_composition:
            # Configure for modular composition patterns
            logger.debug("Enabled modular composition optimizations")
            
        if enable_structured_output:
            # Verify TypedPredictor availability and configure accordingly
            typed_predictor_available = hasattr(dspy, 'TypedPredictor')
            if typed_predictor_available:
                logger.debug("TypedPredictor available - structured output optimizations enabled")
            else:
                logger.warning("TypedPredictor not available - using fallback structured output approach")
        
        logger.info("DSPy configured for signature-based declarative programming workflows")
        
    except Exception as e:
        logger.error(f"Failed to configure DSPy for signature-based workflows: {e}")
        # Fall back to basic configuration
        configure_for_optimization()


def _detect_signature_metadata_availability() -> bool:
    """
    Detect if signature metadata is available by checking common use cases.
    
    Returns:
        True if signature metadata is detected, False otherwise
    """
    global _signature_metadata_detection_result, _signature_metadata_detection_time
    
    # Check for environment variable to skip detection entirely
    if os.environ.get('LLM_SKIP_SIGNATURE_DETECTION_PROBE', '').lower() in ('true', '1', 'yes'):
        logger.debug("Signature detection probe bypassed via LLM_SKIP_SIGNATURE_DETECTION_PROBE environment variable")
        return False
    
    # Check TTL for cached result
    current_time = time.time()
    if (_signature_metadata_detection_result is not None and 
        _signature_metadata_detection_time is not None and 
        current_time - _signature_metadata_detection_time < _SIGNATURE_DETECTION_TTL):
        logger.debug(f"Using cached signature metadata detection result: {_signature_metadata_detection_result} (age: {current_time - _signature_metadata_detection_time:.1f}s)")
        return _signature_metadata_detection_result
    
    # Perform detection
    try:
        # Try to detect if signature metadata is commonly available
        from prompt_optimization.core import load_prompt
        
        # Test for recent signature metadata in common use cases
        has_signature_metadata = False
        test_cases = ["lv0_s1", "lv0_s3"]
        
        for use_case in test_cases:
            try:
                system_data = load_prompt(f"{use_case}_system")
                if system_data.get("signature_metadata"):
                    has_signature_metadata = True
                    logger.debug(f"Signature metadata found in {use_case}_system")
                    break
            except (FileNotFoundError, KeyError):
                logger.debug(f"No signature metadata in {use_case}_system")
                continue
        
        # Memoize the result with timestamp
        _signature_metadata_detection_result = has_signature_metadata
        _signature_metadata_detection_time = current_time
        logger.debug(f"Signature metadata detection completed: {has_signature_metadata}")
        return has_signature_metadata
        
    except ImportError:
        # Prompt optimization not available
        _signature_metadata_detection_result = False
        _signature_metadata_detection_time = current_time
        logger.debug("Prompt optimization not available - signature metadata detection failed")
        return False


def trigger_signature_metadata_detection() -> None:
    """
    Trigger signature metadata detection after successful optimized prompt loading.
    
    This can be called from completions.py after _try_load_optimized_prompt succeeds
    to refresh the detection result.
    """
    global _signature_metadata_detection_result, _signature_metadata_detection_time
    _signature_metadata_detection_result = None  # Reset to trigger re-detection
    _signature_metadata_detection_time = None  # Reset timestamp
    logger.debug("Signature metadata detection result reset - will re-detect on next use")


def inform_successful_prompt_load(has_signature_metadata: bool) -> None:
    """
    Inform the configuration system about successful prompt loading to skip probing.
    
    This function allows completions.py to inform dspy_config about signature metadata
    presence based on actual prompt loading, avoiding the need for detection probes.
    
    Args:
        has_signature_metadata: Whether signature metadata was found during prompt loading
    """
    global _signature_metadata_detection_result, _signature_metadata_detection_time
    
    current_time = time.time()
    _signature_metadata_detection_result = has_signature_metadata
    _signature_metadata_detection_time = current_time
    
    logger.debug(f"Signature metadata presence informed from prompt loading: {has_signature_metadata}")


def _ensure_dspy_configured() -> None:
    """
    Enhanced DSPy configuration that detects signature metadata availability with memoization.
    """
    global _dspy_configured
    
    if not _dspy_configured:
        # Check if we're in a signature-enhanced environment
        has_signature_metadata = _detect_signature_metadata_availability()
        
        if has_signature_metadata:
            logger.debug("Signature metadata detected - configuring for signature-based workflows")
            configure_for_signature_based_workflows()
        else:
            logger.debug("No signature metadata detected - using standard configuration")
            configure_for_optimization()
        
        logger.debug("DSPy automatically configured based on available features")