# Deprecated shim for backward compatibility. Use `generate_glossary.llm` instead.
import warnings
import importlib.util

# Verify the target module exists before importing
spec = importlib.util.find_spec("generate_glossary.llm")
if spec is None:
    raise ImportError(
        "generate_glossary.llm module not found. "
        "Please ensure the package is properly installed and the module structure is correct."
    )

warnings.warn(
    "generate_glossary.utils.llm is deprecated; use generate_glossary.llm",
    DeprecationWarning,
    stacklevel=2,
)
from generate_glossary.llm import *  # re-export public & private API for tests