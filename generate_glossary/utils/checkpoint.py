"""
COMPATIBILITY SHIM - This file maintains backward compatibility.
The actual implementation has been moved to generate_glossary.processing.checkpoint.

This file will be removed in a future version. Please update imports to use:
    from generate_glossary.processing import CheckpointManager, ...
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from generate_glossary.utils.checkpoint is deprecated. "
    "Please use 'from generate_glossary.processing import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from new location for backward compatibility
from ..processing.checkpoint import *