"""
COMPATIBILITY SHIM - This file maintains backward compatibility.
The actual implementation has been moved to generate_glossary.mining.runner.

This file will be removed in a future version. Please update imports to use:
    from generate_glossary.mining import run_web_mining, ...
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from generate_glossary.utils.web_miner_runner is deprecated. "
    "Please use 'from generate_glossary.mining import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from new location for backward compatibility
from ..mining.runner import *