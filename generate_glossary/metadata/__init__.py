"""
Metadata collection and processing utilities for glossary data.
"""

# Import main functions from submodules
from .file_discovery import (
    find_step_file,
    find_step_metadata,
    find_final_file,
    find_dedup_files,
    ensure_final_dirs_exist
)

from .extractors import (
    extract_parent_from_college,
    clean_parent_term,
    is_department_or_college_source,
    extract_concept_from_source,
    extract_metadata_from_json,
    extract_variations_from_dedup
)

from .collector import (
    collect_metadata,
    collect_resources,
    load_parent_level_info,
    process_source_files,
    process_dedup_files
)

from .consolidator import (
    consolidate_metadata_for_term,
    merge_resource_data,
    consolidate_variations,
    promote_terms_based_on_parents,
    consolidate_hierarchy_relationships,
    create_summary_statistics,
    save_consolidated_metadata
)

__all__ = [
    # File discovery
    'find_step_file',
    'find_step_metadata',
    'find_final_file',
    'find_dedup_files',
    'ensure_final_dirs_exist',
    
    # Extractors
    'extract_parent_from_college',
    'clean_parent_term',
    'is_department_or_college_source',
    'extract_concept_from_source',
    'extract_metadata_from_json',
    'extract_variations_from_dedup',
    
    # Collector
    'collect_metadata',
    'collect_resources',
    'load_parent_level_info',
    'process_source_files',
    'process_dedup_files',
    
    # Consolidator
    'consolidate_metadata_for_term',
    'merge_resource_data',
    'consolidate_variations',
    'promote_terms_based_on_parents',
    'consolidate_hierarchy_relationships',
    'create_summary_statistics',
    'save_consolidated_metadata'
]