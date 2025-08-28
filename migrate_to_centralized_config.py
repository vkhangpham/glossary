#!/usr/bin/env python3
"""
Migration script to update generation files to use centralized configuration.

This script automatically updates all generation scripts to use the new 
centralized configuration system instead of hardcoded Config classes.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Configuration mappings for different steps
LEVEL_STEP_CONFIGS = {
    # Level 0
    ("lv0", "s0"): {
        "input_method": "None  # Step 0 doesn't read from previous step",
        "output_method": "level_config.get_step_output_file(0)",
        "level": 0,
        "step": 0
    },
    ("lv0", "s1"): {
        "input_method": "level_config.get_step_input_file(1)",
        "output_method": "level_config.get_step_output_file(1)",
        "level": 0,
        "step": 1
    },
    ("lv0", "s2"): {
        "input_method": "level_config.get_step_input_file(2)",
        "output_method": "level_config.get_step_output_file(2)",
        "level": 0,
        "step": 2
    },
    ("lv0", "s3"): {
        "input_method": "level_config.get_step_input_file(3)",
        "output_method": "level_config.get_step_output_file(3)",
        "level": 0,
        "step": 3
    },
    # Level 1 - same pattern
    ("lv1", "s0"): {
        "input_method": "None  # Uses Level 0 final file",
        "output_method": "level_config.get_step_output_file(0)",
        "level": 1,
        "step": 0
    },
    ("lv1", "s1"): {
        "input_method": "level_config.get_step_input_file(1)",
        "output_method": "level_config.get_step_output_file(1)",
        "level": 1,
        "step": 1
    },
    # Add more as needed...
}

# Configuration field mappings
CONFIG_MAPPINGS = {
    "Config.BATCH_SIZE": "processing_config.batch_size",
    "Config.NUM_LLM_ATTEMPTS": "processing_config.llm_attempts",
    "Config.CONCEPT_AGREEMENT_THRESH": "processing_config.concept_agreement_threshold",
    "Config.KW_APPEARANCE_THRESH": "processing_config.keyword_appearance_threshold",
    "Config.MAX_WORKERS": "processing_config.max_workers",
    "Config.CHUNK_SIZE": "processing_config.chunk_size",
    "Config.CONFERENCE_FREQ_THRESHOLD": "processing_config.conference_frequency_threshold",
    "Config.AGREEMENT_THRESHOLD": "processing_config.concept_agreement_threshold",
    "Config.INPUT_FILE": "level_config.get_step_input_file({step})",
    "Config.OUTPUT_FILE": "level_config.get_step_output_file({step})",
    "Config.META_FILE": "level_config.get_step_metadata_file({step})",
}

def extract_level_step(filename: str) -> Tuple[str, str]:
    """Extract level and step from filename like lv0_s1_extract_concepts.py"""
    match = re.match(r'(lv\d)_(s\d)_.*\.py', filename)
    if match:
        return match.groups()
    return None, None

def generate_config_header(level: int) -> str:
    """Generate the new configuration header for a file"""
    return f'''# Use centralized configuration
LEVEL = {level}
level_config = get_level_config(LEVEL)
processing_config = get_processing_config(LEVEL)

# Ensure directories exist
ensure_directories(LEVEL)
'''

def find_config_class_section(content: str) -> Tuple[int, int]:
    """Find the start and end of the Config class definition"""
    lines = content.split('\n')
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        if line.strip().startswith('class Config:'):
            start_idx = i
        elif start_idx is not None and line and not line.startswith((' ', '\t')) and 'Config' not in line:
            end_idx = i
            break
    
    if start_idx is not None and end_idx is None:
        # Config class extends to end of file or next class
        for i in range(start_idx + 1, len(lines)):
            if lines[i].strip().startswith('class ') and 'Config' not in lines[i]:
                end_idx = i
                break
        if end_idx is None:
            end_idx = len(lines)
    
    return start_idx, end_idx

def update_imports(content: str) -> str:
    """Add centralized config imports"""
    if "from generate_glossary.config import" in content:
        return content  # Already updated
    
    # Find the line with logger import to add our imports after
    lines = content.split('\n')
    insert_idx = None
    
    for i, line in enumerate(lines):
        if 'from generate_glossary.utils.logger import' in line:
            insert_idx = i + 1
            break
    
    if insert_idx is None:
        # Fallback: add after last import
        for i, line in enumerate(lines):
            if line.startswith('from ') or line.startswith('import '):
                insert_idx = i + 1
    
    if insert_idx is not None:
        import_line = "from generate_glossary.config import get_level_config, get_processing_config, ensure_directories"
        lines.insert(insert_idx, import_line)
    
    return '\n'.join(lines)

def replace_config_references(content: str, level: int, step: int) -> str:
    """Replace Config.* references with centralized config calls"""
    
    # Basic replacements
    for old_config, new_config in CONFIG_MAPPINGS.items():
        if "{step}" in new_config:
            new_config = new_config.format(step=step)
        content = content.replace(old_config, new_config)
    
    # Handle special file path cases
    content = re.sub(
        r'Config\.(INPUT_FILE|OUTPUT_FILE|META_FILE)',
        lambda m: f"level_config.get_step_{m.group(1).lower()}({step})",
        content
    )
    
    return content

def migrate_file(file_path: Path) -> bool:
    """Migrate a single generation file to use centralized configuration"""
    print(f"Migrating {file_path}")
    
    # Extract level and step info
    level_str, step_str = extract_level_step(file_path.name)
    if not level_str or not step_str:
        print(f"  Skipping {file_path.name} - cannot determine level/step")
        return False
    
    level = int(level_str[2:])  # lv0 -> 0
    step = int(step_str[1:])    # s1 -> 1
    
    # Read current content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return False
    
    # Skip if already migrated
    if "from generate_glossary.config import" in content:
        print(f"  Already migrated: {file_path.name}")
        return True
    
    # Skip if no Config class found
    if "class Config:" not in content:
        print(f"  No Config class found in: {file_path.name}")
        return True
    
    # Update imports
    content = update_imports(content)
    
    # Replace Config class with centralized config
    config_start, config_end = find_config_class_section(content)
    if config_start is not None and config_end is not None:
        lines = content.split('\n')
        
        # Replace Config class section with centralized config
        config_header = generate_config_header(level)
        lines = lines[:config_start] + config_header.split('\n') + lines[config_end:]
        content = '\n'.join(lines)
    
    # Replace Config.* references
    content = replace_config_references(content, level, step)
    
    # Write back to file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  Successfully migrated: {file_path.name}")
        return True
    except Exception as e:
        print(f"  Error writing {file_path}: {e}")
        return False

def main():
    """Main migration function"""
    print("🔄 Starting migration to centralized configuration system")
    
    # Find all generation files
    generation_dir = Path("generate_glossary/generation")
    if not generation_dir.exists():
        print(f"❌ Generation directory not found: {generation_dir}")
        return
    
    generation_files = []
    for level_dir in generation_dir.glob("lv*"):
        if level_dir.is_dir():
            for py_file in level_dir.glob("*.py"):
                if py_file.name.startswith("lv") and "_s" in py_file.name:
                    generation_files.append(py_file)
    
    print(f"📁 Found {len(generation_files)} generation files to migrate")
    
    # Migrate each file
    success_count = 0
    for file_path in generation_files:
        if migrate_file(file_path):
            success_count += 1
    
    print(f"\n✅ Migration complete: {success_count}/{len(generation_files)} files migrated successfully")
    
    if success_count == len(generation_files):
        print("\n🎉 All files migrated successfully!")
        print("\nNext steps:")
        print("1. Test the migrated files to ensure they work correctly")
        print("2. Update any remaining manual Config references")
        print("3. Consider removing old requirements.txt in favor of pyproject.toml")
    else:
        print(f"\n⚠️  {len(generation_files) - success_count} files need manual attention")

if __name__ == "__main__":
    main()