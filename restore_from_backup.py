#!/usr/bin/env python3
"""
Script to restore from backups if something goes wrong during the fix.
"""

import os
import shutil
import json
import glob
from pathlib import Path

def find_backup_directories():
    """Find all available backup directories."""
    backup_pattern = "backups/empty_sources_fix_*"
    backup_dirs = glob.glob(backup_pattern)
    backup_dirs.sort(reverse=True)  # Most recent first
    return backup_dirs

def show_available_backups():
    """Show all available backups."""
    backup_dirs = find_backup_directories()
    
    if not backup_dirs:
        print("❌ No backups found!")
        return None
    
    print("📁 Available Backups:")
    print("=" * 50)
    
    for i, backup_dir in enumerate(backup_dirs, 1):
        print(f"{i}. {backup_dir}")
        
        # Try to read manifest for more details
        manifest_file = f"{backup_dir}/backup_manifest.json"
        if os.path.exists(manifest_file):
            try:
                with open(manifest_file, 'r') as f:
                    manifest = json.load(f)
                print(f"   Time: {manifest['backup_timestamp']}")
                print(f"   Reason: {manifest['backup_reason']}")
                print(f"   Files: {len(manifest['files_backed_up'])}")
            except Exception as e:
                print(f"   (Error reading manifest: {e})")
        print()
    
    return backup_dirs

def restore_from_backup(backup_dir):
    """Restore from a specific backup directory."""
    print(f"🔄 Restoring from backup: {backup_dir}")
    print("=" * 50)
    
    # Check if backup exists
    if not os.path.exists(backup_dir):
        print(f"❌ Backup directory not found: {backup_dir}")
        return False
    
    # Read manifest
    manifest_file = f"{backup_dir}/backup_manifest.json"
    if os.path.exists(manifest_file):
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        print(f"📋 Backup from: {manifest['backup_timestamp']}")
        print(f"📋 Reason: {manifest['backup_reason']}")
    
    # Restore data/final directory
    backup_data_final = f"{backup_dir}/data_final"
    if os.path.exists(backup_data_final):
        print("📁 Restoring data/final directory...")
        
        # Remove current data/final if it exists
        if os.path.exists("data/final"):
            shutil.rmtree("data/final")
        
        # Restore from backup
        shutil.copytree(backup_data_final, "data/final")
        print("  ✅ Restored data/final directory")
    else:
        print("  ⚠️  data/final backup not found")
    
    # Restore metadata_collector.py
    backup_metadata_collector = f"{backup_dir}/metadata_collector.py"
    if os.path.exists(backup_metadata_collector):
        print("📄 Restoring metadata_collector.py...")
        
        target_file = "generate_glossary/utils/metadata_collector.py"
        shutil.copy2(backup_metadata_collector, target_file)
        print("  ✅ Restored metadata_collector.py")
    else:
        print("  ⚠️  metadata_collector.py backup not found")
    
    # Restore hierarchy files
    hierarchy_files = ["hierarchy.json", "hierarchy_with_splits.json"]
    for filename in hierarchy_files:
        backup_file = f"{backup_dir}/{filename}"
        if os.path.exists(backup_file):
            print(f"🌳 Restoring {filename}...")
            
            # Determine target location
            if filename == "hierarchy.json":
                target_file = "data/final/hierarchy.json"
            else:
                target_file = f"data/{filename}"
            
            # Create target directory if needed
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            
            # Restore file
            shutil.copy2(backup_file, target_file)
            print(f"  ✅ Restored {filename}")
        else:
            print(f"  ⚠️  {filename} backup not found")
    
    print("\n✅ RESTORE COMPLETED!")
    return True

def verify_restore():
    """Verify that the restore was successful."""
    print("\n🔍 Verifying restore...")
    print("=" * 30)
    
    # Check key files exist
    key_files = [
        "data/final/lv0/lv0_metadata.json",
        "data/final/lv1/lv1_metadata.json", 
        "data/final/lv2/lv2_metadata.json",
        "data/final/lv3/lv3_metadata.json",
        "data/final/hierarchy.json",
        "generate_glossary/utils/metadata_collector.py"
    ]
    
    all_good = True
    for file_path in key_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING!")
            all_good = False
    
    if all_good:
        print("\n✅ All key files restored successfully!")
    else:
        print("\n❌ Some files are missing! Check the restore process.")
    
    return all_good

def main():
    print("🔄 Restore from Backup Tool")
    print("=" * 40)
    
    backup_dirs = show_available_backups()
    if not backup_dirs:
        return
    
    # Get user choice
    try:
        choice = input(f"\nEnter backup number to restore (1-{len(backup_dirs)}) or 'q' to quit: ").strip()
        
        if choice.lower() == 'q':
            print("Cancelled.")
            return
        
        choice_num = int(choice)
        if choice_num < 1 or choice_num > len(backup_dirs):
            print("❌ Invalid choice!")
            return
        
        selected_backup = backup_dirs[choice_num - 1]
        
        # Confirm restore
        print(f"\n⚠️  This will OVERWRITE current data with backup from:")
        print(f"   {selected_backup}")
        confirm = input("Are you sure? (y/N): ").strip().lower()
        
        if confirm != 'y':
            print("Cancelled.")
            return
        
        # Perform restore
        if restore_from_backup(selected_backup):
            verify_restore()
        
    except (ValueError, KeyboardInterrupt):
        print("\nCancelled.")
    except Exception as e:
        print(f"❌ Error during restore: {e}")

if __name__ == "__main__":
    main() 