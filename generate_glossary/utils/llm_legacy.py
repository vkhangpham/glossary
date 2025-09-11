"""
DEPRECATED: This legacy module has been removed and replaced.

Please update your imports to use the new structure:

OLD:
    from generate_glossary.utils.llm_legacy import completion, structured_completion

NEW:
    from generate_glossary.llm import completion, structured_completion

The llm_legacy module has been completely removed and replaced with the main
generate_glossary.llm module for better organization and maintainability.

This stub will be removed in the next minor release.
"""

raise ImportError(
    "llm_legacy module has been removed. "
    "Use 'from generate_glossary.llm import ...' instead. "
    "The module has been moved to generate_glossary.llm with the same API."
)