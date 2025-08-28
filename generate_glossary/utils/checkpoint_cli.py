"""
Command-line interface for managing checkpoints.

This CLI tool allows users to:
- List all available checkpoints
- Resume interrupted processing from checkpoints  
- Clean up old/corrupted checkpoints
- Get status of processing operations
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from generate_glossary.config import get_level_config
from generate_glossary.utils.checkpoint import CheckpointManager
from generate_glossary.utils.logger import setup_logger

logger = setup_logger("checkpoint_cli")

def list_checkpoints(level: Optional[int] = None) -> None:
    """List all available checkpoints"""
    if level is not None:
        # List checkpoints for specific level
        level_config = get_level_config(level)
        checkpoint_dir = level_config.data_dir / "checkpoints"
        manager = CheckpointManager(checkpoint_dir)
        
        checkpoints = manager.list_checkpoints()
        print(f"\n=== Level {level} Checkpoints ===")
        
    else:
        # List checkpoints for all levels
        all_checkpoints = []
        for lvl in range(4):
            try:
                level_config = get_level_config(lvl)
                checkpoint_dir = level_config.data_dir / "checkpoints"
                if checkpoint_dir.exists():
                    manager = CheckpointManager(checkpoint_dir)
                    checkpoints = manager.list_checkpoints()
                    for checkpoint in checkpoints:
                        checkpoint['level'] = lvl
                    all_checkpoints.extend(checkpoints)
            except Exception as e:
                logger.warning(f"Could not access level {lvl} checkpoints: {e}")
                
        checkpoints = sorted(all_checkpoints, key=lambda x: x.get("created_at", ""))
        print(f"\n=== All Checkpoints ({len(checkpoints)} found) ===")
    
    if not checkpoints:
        print("No checkpoints found.")
        return
        
    # Display checkpoints in table format
    print(f"{'Level':<5} {'Step':<8} {'Operation':<15} {'Progress':<12} {'Size (MB)':<10} {'Created':<20}")
    print("-" * 80)
    
    for checkpoint in checkpoints:
        level_str = str(checkpoint.get('level', '?'))
        step = checkpoint.get('step', '?')
        operation = checkpoint.get('operation', '?')
        progress = checkpoint.get('progress', '?/?')
        size_mb = f"{checkpoint.get('size_mb', 0):.2f}"
        created = checkpoint.get('created_at', '?')[:19] if checkpoint.get('created_at') else '?'
        
        print(f"{level_str:<5} {step:<8} {operation:<15} {progress:<12} {size_mb:<10} {created:<20}")

def show_checkpoint_details(level: int, step: str, operation: str) -> None:
    """Show detailed information about a specific checkpoint"""
    level_config = get_level_config(level)
    checkpoint_dir = level_config.data_dir / "checkpoints"
    manager = CheckpointManager(checkpoint_dir)
    
    checkpoint_data = manager.load_checkpoint(step, operation)
    
    if not checkpoint_data:
        print(f"No checkpoint found for {step} {operation}")
        return
        
    print(f"\n=== Checkpoint Details: {step} {operation} ===")
    print(f"Created: {checkpoint_data.metadata.created_at}")
    print(f"Total Items: {checkpoint_data.metadata.total_items}")
    print(f"Processed Items: {checkpoint_data.metadata.processed_items}")
    print(f"Batch Size: {checkpoint_data.metadata.batch_size}")
    print(f"Completed Batches: {len(checkpoint_data.completed_batches)}")
    print(f"Failed Items: {len(checkpoint_data.failed_items or [])}")
    print(f"Config Hash: {checkpoint_data.metadata.config_hash}")
    
    if checkpoint_data.metadata.provider_info:
        print(f"Provider Info: {json.dumps(checkpoint_data.metadata.provider_info, indent=2)}")
        
    progress_pct = (checkpoint_data.metadata.processed_items / 
                   checkpoint_data.metadata.total_items * 100)
    print(f"Progress: {progress_pct:.1f}%")
    
    if checkpoint_data.failed_items:
        print(f"\nFailed Items ({len(checkpoint_data.failed_items)}):")
        for item in checkpoint_data.failed_items[:10]:  # Show first 10
            print(f"  - {item}")
        if len(checkpoint_data.failed_items) > 10:
            print(f"  ... and {len(checkpoint_data.failed_items) - 10} more")

def cleanup_checkpoints(level: Optional[int] = None, confirm: bool = False) -> None:
    """Clean up checkpoints"""
    if level is not None:
        levels = [level]
    else:
        levels = range(4)
        
    total_cleaned = 0
    
    for lvl in levels:
        try:
            level_config = get_level_config(lvl)
            checkpoint_dir = level_config.data_dir / "checkpoints"
            
            if not checkpoint_dir.exists():
                continue
                
            manager = CheckpointManager(checkpoint_dir)
            checkpoints = manager.list_checkpoints()
            
            if not checkpoints:
                continue
                
            print(f"\nLevel {lvl} checkpoints to clean:")
            for checkpoint in checkpoints:
                print(f"  - {checkpoint['step']} {checkpoint['operation']} ({checkpoint['progress']})")
                
            if not confirm:
                response = input(f"Delete {len(checkpoints)} checkpoint(s) for level {lvl}? (y/N): ")
                if response.lower() != 'y':
                    continue
                    
            # Clean up checkpoints
            for checkpoint in checkpoints:
                try:
                    manager.cleanup_checkpoint(checkpoint['step'], checkpoint['operation'])
                    total_cleaned += 1
                except Exception as e:
                    logger.error(f"Failed to cleanup {checkpoint['step']} {checkpoint['operation']}: {e}")
                    
        except Exception as e:
            logger.warning(f"Could not cleanup level {lvl} checkpoints: {e}")
            
    print(f"\nCleaned up {total_cleaned} checkpoint(s)")

def resume_processing(level: int, step: str, operation: str) -> None:
    """Show information for resuming processing"""
    level_config = get_level_config(level)
    checkpoint_dir = level_config.data_dir / "checkpoints"
    manager = CheckpointManager(checkpoint_dir)
    
    resume_info = manager.get_resume_info(step, operation)
    
    if not resume_info:
        print(f"No checkpoint found for {step} {operation}")
        return
        
    print(f"\n=== Resume Information: {step} {operation} ===")
    print(f"Progress: {resume_info['progress']['percentage']:.1f}%")
    print(f"Processed: {resume_info['progress']['processed']}/{resume_info['progress']['total']}")
    print(f"Resume from index: {resume_info['resume_index']}")
    print(f"Completed batches: {len(resume_info['completed_batches'])}")
    print(f"Failed items: {len(resume_info['failed_items'])}")
    
    print(f"\nTo resume processing, run the original command:")
    
    # Generate helpful resume command based on step and operation
    if "extract_concepts" in operation:
        script_name = f"lv{level}_s1_extract_concepts"
        print(f"python -m generate_glossary.generation.lv{level}.{script_name}")
    elif "verify" in operation:
        script_name = f"lv{level}_s3_verify_single_token"
        print(f"python -m generate_glossary.generation.lv{level}.{script_name}")
    else:
        print(f"python -m generate_glossary.generation.lv{level}.[appropriate_script]")
        
    print("\nThe script will automatically detect and resume from the checkpoint.")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Manage processing checkpoints for glossary generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list                          # List all checkpoints
  %(prog)s list --level 0               # List level 0 checkpoints
  %(prog)s show 0 lv0_s1 extract_concepts  # Show checkpoint details
  %(prog)s resume 0 lv0_s1 extract_concepts # Show resume information
  %(prog)s cleanup --level 0            # Clean level 0 checkpoints
  %(prog)s cleanup --confirm            # Clean all checkpoints without prompting
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List checkpoints')
    list_parser.add_argument('--level', type=int, choices=[0, 1, 2, 3],
                            help='Show checkpoints for specific level only')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show checkpoint details')
    show_parser.add_argument('level', type=int, choices=[0, 1, 2, 3], 
                            help='Level number')
    show_parser.add_argument('step', type=str, help='Step name (e.g., lv0_s1)')
    show_parser.add_argument('operation', type=str, help='Operation name (e.g., extract_concepts)')
    
    # Resume command
    resume_parser = subparsers.add_parser('resume', help='Show resume information')
    resume_parser.add_argument('level', type=int, choices=[0, 1, 2, 3], 
                              help='Level number')
    resume_parser.add_argument('step', type=str, help='Step name (e.g., lv0_s1)')
    resume_parser.add_argument('operation', type=str, help='Operation name (e.g., extract_concepts)')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up checkpoints')
    cleanup_parser.add_argument('--level', type=int, choices=[0, 1, 2, 3],
                               help='Clean checkpoints for specific level only')
    cleanup_parser.add_argument('--confirm', action='store_true',
                               help='Skip confirmation prompts')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    try:
        if args.command == 'list':
            list_checkpoints(args.level)
        elif args.command == 'show':
            show_checkpoint_details(args.level, args.step, args.operation)
        elif args.command == 'resume':
            resume_processing(args.level, args.step, args.operation)
        elif args.command == 'cleanup':
            cleanup_checkpoints(args.level, args.confirm)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())