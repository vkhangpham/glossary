#!/usr/bin/env python3
"""
Script to create backups before fixing empty sources issue.
"""

import os
import shutil
import json
from datetime import datetime
from pathlib import Path

def create_backup_directory():
    """Create a timestamped backup directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backups/empty_sources_fix_{timestamp}"
    os.makedirs(backup_dir, exist_ok=True)
    return backup_dir

def backup_data_final(backup_dir):
    """Backup the entire data/final directory."""
    print("📁 Backing up data/final directory...")
    
    if os.path.exists("data/final"):
        shutil.copytree("data/final", f"{backup_dir}/data_final")
        print(f"  ✅ Backed up data/final to {backup_dir}/data_final")
    else:
        print("  ⚠️  data/final directory not found")

def backup_metadata_collector(backup_dir):
    """Backup the metadata collector file."""
    print("📄 Backing up metadata_collector.py...")
    
    source_file = "generate_glossary/utils/metadata_collector.py"
    if os.path.exists(source_file):
        shutil.copy2(source_file, f"{backup_dir}/metadata_collector.py")
        print(f"  ✅ Backed up {source_file} to {backup_dir}/metadata_collector.py")
    else:
        print(f"  ⚠️  {source_file} not found")

def backup_hierarchy_files(backup_dir):
    """Backup hierarchy.json and related files."""
    print("🌳 Backing up hierarchy files...")
    
    hierarchy_files = [
        "data/final/hierarchy.json",
        "data/hierarchy_with_splits.json"
    ]
    
    for file_path in hierarchy_files:
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            shutil.copy2(file_path, f"{backup_dir}/{filename}")
            print(f"  ✅ Backed up {file_path}")
        else:
            print(f"  ⚠️  {file_path} not found")

def create_backup_manifest(backup_dir):
    """Create a manifest of what was backed up."""
    print("📋 Creating backup manifest...")
    
    manifest = {
        "backup_timestamp": datetime.now().isoformat(),
        "backup_reason": "Fix empty sources issue - overly aggressive filtering",
        "files_backed_up": [],
        "directories_backed_up": [],
        "empty_sources_before_fix": {}
    }
    
    # List all backed up files
    backup_path = Path(backup_dir)
    for item in backup_path.rglob("*"):
        if item.is_file():
            manifest["files_backed_up"].append(str(item.relative_to(backup_path)))
        elif item.is_dir():
            manifest["directories_backed_up"].append(str(item.relative_to(backup_path)))
    
    # Record current empty sources statistics
    for level in [0, 1, 2, 3]:
        try:
            metadata_file = f"data/final/lv{level}/lv{level}_metadata.json"
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                empty_count = sum(1 for data in metadata.values() if 'sources' in data and not data['sources'])
                total_count = len(metadata)
                
                manifest["empty_sources_before_fix"][f"level_{level}"] = {
                    "empty_count": empty_count,
                    "total_count": total_count,
                    "percentage": round(empty_count / total_count * 100, 1) if total_count > 0 else 0
                }
        except Exception as e:
            manifest["empty_sources_before_fix"][f"level_{level}"] = {"error": str(e)}
    
    # Save manifest
    with open(f"{backup_dir}/backup_manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"  ✅ Created backup manifest at {backup_dir}/backup_manifest.json")

def show_backup_summary(backup_dir):
    """Show summary of what was backed up."""
    print("\n" + "=" * 60)
    print("📊 BACKUP SUMMARY")
    print("=" * 60)
    
    if os.path.exists(f"{backup_dir}/backup_manifest.json"):
        with open(f"{backup_dir}/backup_manifest.json", 'r') as f:
            manifest = json.load(f)
        
        print(f"Backup Location: {backup_dir}")
        print(f"Backup Time: {manifest['backup_timestamp']}")
        print(f"Reason: {manifest['backup_reason']}")
        print(f"Files Backed Up: {len(manifest['files_backed_up'])}")
        print(f"Directories Backed Up: {len(manifest['directories_backed_up'])}")
        
        print("\nEmpty Sources Statistics (BEFORE fix):")
        total_empty = 0
        total_terms = 0
        for level_key, stats in manifest["empty_sources_before_fix"].items():
            if "error" not in stats:
                level = level_key.split('_')[1]
                empty_count = stats["empty_count"]
                total_count = stats["total_count"]
                percentage = stats["percentage"]
                total_empty += empty_count
                total_terms += total_count
                print(f"  Level {level}: {empty_count:2d}/{total_count:4d} terms with empty sources ({percentage:.1f}%)")
        
        if total_terms > 0:
            print(f"  Total: {total_empty}/{total_terms} terms with empty sources ({total_empty/total_terms*100:.1f}%)")

def main():
    print("🔄 Creating Backups Before Fixing Empty Sources")
    print("=" * 60)
    
    # Create backup directory
    backup_dir = create_backup_directory()
    print(f"📁 Created backup directory: {backup_dir}")
    
    # Perform backups
    backup_data_final(backup_dir)
    backup_metadata_collector(backup_dir)
    backup_hierarchy_files(backup_dir)
    create_backup_manifest(backup_dir)
    
    # Show summary
    show_backup_summary(backup_dir)
    
    print("\n" + "=" * 60)
    print("✅ BACKUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("You can now safely proceed with the fix.")
    print(f"If anything goes wrong, restore from: {backup_dir}")

if __name__ == "__main__":
    main() 