"""
COMPATIBILITY SHIM - This file maintains backward compatibility.
The actual implementation has been moved to generate_glossary.security.api_keys.

This file will be removed in a future version. Please update imports to use:
    from generate_glossary.security import APIKeyManager, ...
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from generate_glossary.utils.secure_config is deprecated. "
    "Please use 'from generate_glossary.security import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from new location for backward compatibility
from ..security.api_keys import *