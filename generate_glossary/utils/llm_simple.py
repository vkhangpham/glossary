"""
COMPATIBILITY SHIM - This file maintains backward compatibility.
The actual implementation is now in generate_glossary.utils.llm.

This file will be removed in a future version. Please update imports to use:
    from generate_glossary.utils.llm import ...
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from generate_glossary.utils.llm_simple is deprecated. "
    "Please use 'from generate_glossary.utils.llm import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from new location for backward compatibility
from .llm import *