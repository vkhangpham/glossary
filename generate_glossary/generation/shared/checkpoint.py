"""
Simple checkpoint utility for generation modules.

This is a minimal implementation that saves/loads progress for expensive
operations (like LLM calls). No over-engineering, just the essentials.

Usage:
    # Simple save/load
    save_checkpoint(checkpoint_file, batch_idx, results)
    results, completed = load_checkpoint(checkpoint_file)
    
    # Or use the convenience function
    results = process_with_checkpoint(
        items=data,
        batch_size=20,
        checkpoint_file=Path(".checkpoint.json"),
        process_batch_func=my_process_func
    )
"""

import json
import time
from pathlib import Path
from typing import Dict, Set, Tuple, Any, List, Callable, Optional

# Checkpoint version for backward compatibility
CHECKPOINT_VERSION = "1.1"


def cleanup_orphaned_temps(checkpoint_dir: Path) -> int:
    """
    Clean up orphaned .tmp files from interrupted operations.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        Number of orphaned temp files cleaned
    """
    if not checkpoint_dir.exists():
        return 0
    
    cleaned_count = 0
    for tmp_file in checkpoint_dir.glob("*.tmp"):
        # Check if corresponding .json file exists
        json_file = tmp_file.with_suffix('.json')
        if not json_file.exists():
            # This is an orphaned temp file
            print(f"Cleaning orphaned temp file: {tmp_file.name}")
            tmp_file.unlink()
            cleaned_count += 1
    
    return cleaned_count


def detect_interrupted_checkpoint(checkpoint_file: Path) -> bool:
    """
    Detect if there's an interrupted checkpoint (temp file without json file).
    
    Args:
        checkpoint_file: Path to checkpoint file
        
    Returns:
        True if interrupted checkpoint detected
    """
    temp_file = checkpoint_file.with_suffix('.tmp')
    return temp_file.exists() and not checkpoint_file.exists()


def save_checkpoint(checkpoint_file: Path, batch_idx: int, results: Dict[str, Any], max_retries: int = 3) -> None:
    """
    Save progress checkpoint after completing a batch with retry logic.
    
    Args:
        checkpoint_file: Path to checkpoint file
        batch_idx: Index of completed batch
        results: Accumulated results so far
        max_retries: Maximum number of retry attempts for atomic write
    """
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint_data = {
        "version": CHECKPOINT_VERSION,
        "last_batch": batch_idx,
        "completed_batches": list(range(batch_idx + 1)),
        "timestamp": time.time(),
        "results": results
    }
    
    # Write atomically by using a temp file with retry logic
    temp_file = checkpoint_file.with_suffix('.tmp')
    
    for attempt in range(max_retries):
        try:
            # Write to temp file
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_file.replace(checkpoint_file)
            return  # Success
            
        except (OSError, IOError) as e:
            if attempt < max_retries - 1:
                print(f"Warning: Failed to save checkpoint (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(0.5)  # Brief pause before retry
            else:
                print(f"Error: Failed to save checkpoint after {max_retries} attempts: {e}")
                raise


def load_checkpoint(checkpoint_file: Path) -> Tuple[Dict[str, Any], Set[int]]:
    """
    Load checkpoint if it exists, with validation.
    
    Args:
        checkpoint_file: Path to checkpoint file
        
    Returns:
        Tuple of (results dict, set of completed batch indices)
    """
    # Check for interrupted checkpoint first
    if detect_interrupted_checkpoint(checkpoint_file):
        temp_file = checkpoint_file.with_suffix('.tmp')
        print(f"Found interrupted checkpoint, attempting recovery from: {temp_file.name}")
        
        # Try to recover from temp file
        try:
            with open(temp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate structure
            if _validate_checkpoint_structure(data):
                print(f"Successfully recovered from interrupted checkpoint (batch {data.get('last_batch', 0)})")
                # Move temp to main file for next save
                temp_file.replace(checkpoint_file)
                return data.get("results", {}), set(data.get("completed_batches", []))
            else:
                print("Warning: Interrupted checkpoint has invalid structure, starting fresh")
                temp_file.unlink()  # Remove corrupted temp file
                return {}, set()
                
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Could not recover from interrupted checkpoint: {e}")
            temp_file.unlink()  # Remove corrupted temp file
            return {}, set()
    
    if not checkpoint_file.exists():
        return {}, set()
    
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate checkpoint structure
        if not _validate_checkpoint_structure(data):
            print("Warning: Checkpoint has invalid structure, starting fresh")
            return {}, set()
        
        # Check version compatibility (backward compatible)
        version = data.get("version", "1.0")
        if version.split(".")[0] != CHECKPOINT_VERSION.split(".")[0]:
            print(f"Warning: Checkpoint version mismatch ({version} vs {CHECKPOINT_VERSION}), may have compatibility issues")
        
        return data.get("results", {}), set(data.get("completed_batches", []))
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Corrupted checkpoint file, starting fresh: {e}")
        return {}, set()


def _validate_checkpoint_structure(data: Any) -> bool:
    """
    Validate checkpoint data structure.
    
    Args:
        data: Data loaded from checkpoint file
        
    Returns:
        True if structure is valid
    """
    if not isinstance(data, dict):
        return False
    
    # Check required fields
    required_fields = ["last_batch", "completed_batches", "results"]
    for field in required_fields:
        if field not in data:
            return False
    
    # Validate field types
    if not isinstance(data["last_batch"], int):
        return False
    if not isinstance(data["completed_batches"], list):
        return False
    if not isinstance(data["results"], dict):
        return False
    
    return True


def clear_checkpoint(checkpoint_file: Path) -> None:
    """Remove checkpoint file after successful completion."""
    checkpoint_file.unlink(missing_ok=True)
    # Also remove temp file if it exists
    checkpoint_file.with_suffix('.tmp').unlink(missing_ok=True)


def process_with_checkpoint(
    items: List[Any],
    batch_size: int,
    checkpoint_file: Path,
    process_batch_func: Callable[[List[Any]], Dict[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    """
    Process items in batches with automatic checkpointing and enhanced logging.
    
    This is a convenience function that handles the checkpoint save/load
    cycle automatically, with orphan cleanup and resume detection.
    
    Args:
        items: List of items to process
        batch_size: Number of items per batch
        checkpoint_file: Path to checkpoint file
        process_batch_func: Function that processes a batch and returns results dict
        **kwargs: Additional arguments passed to process_batch_func
        
    Returns:
        Dictionary with all accumulated results
        
    Example:
        def process_batch(batch_items, temperature=0.3):
            # Process and return results
            return {"item1": result1, "item2": result2}
        
        results = process_with_checkpoint(
            items=my_data,
            batch_size=20,
            checkpoint_file=Path(".checkpoint.json"),
            process_batch_func=process_batch,
            temperature=0.5  # passed to process_batch_func
        )
    """
    from pydash import chunk
    
    # Clean up orphaned temp files on startup
    checkpoint_dir = checkpoint_file.parent
    cleaned = cleanup_orphaned_temps(checkpoint_dir)
    if cleaned > 0:
        print(f"Cleaned {cleaned} orphaned temp file(s)")
    
    # Check if we're resuming from interruption
    was_interrupted = detect_interrupted_checkpoint(checkpoint_file)
    
    # Load any existing checkpoint
    results, completed_batches = load_checkpoint(checkpoint_file)
    
    # Split items into batches
    batches = list(chunk(items, batch_size))
    total_batches = len(batches)
    
    # Show resume status
    if completed_batches:
        completed_count = len(completed_batches)
        if was_interrupted:
            print(f"Resuming from interrupted checkpoint: {completed_count}/{total_batches} batches completed")
        else:
            print(f"Resuming from checkpoint: {completed_count}/{total_batches} batches completed")
    else:
        print(f"Starting fresh: {total_batches} batches to process")
    
    # Track timing for progress estimation
    start_time = time.time()
    processed_in_session = 0
    
    # Process remaining batches
    for batch_idx, batch in enumerate(batches):
        if batch_idx in completed_batches:
            continue  # Skip already completed batches
        
        batch_start = time.time()
        
        # Process this batch
        batch_results = process_batch_func(batch, **kwargs)
        
        # Check for key collisions before updating (only for dict results)
        if isinstance(batch_results, dict):
            # Warn about potential overwrites
            overlapping_keys = set(results.keys()) & set(batch_results.keys())
            if overlapping_keys:
                print(f"Warning: Batch {batch_idx} has {len(overlapping_keys)} overlapping keys with previous results")
            results.update(batch_results)
        else:
            # If batch_results is not a dict, store with batch index as key
            results[f"batch_{batch_idx}"] = batch_results
        
        # Save checkpoint
        save_checkpoint(checkpoint_file, batch_idx, results)
        
        # Progress reporting with time estimation
        batch_time = time.time() - batch_start
        processed_in_session += 1
        completed_total = batch_idx + 1
        remaining = total_batches - completed_total
        
        if processed_in_session > 0:
            avg_time = (time.time() - start_time) / processed_in_session
            estimated_remaining = remaining * avg_time
            print(f"Completed batch {completed_total}/{total_batches} ({batch_time:.1f}s) - "
                  f"Est. remaining: {estimated_remaining:.0f}s")
        else:
            print(f"Completed batch {completed_total}/{total_batches}")
    
    # Clear checkpoint after successful completion
    clear_checkpoint(checkpoint_file)
    print(f"Processing complete! Total time: {time.time() - start_time:.1f}s")
    
    return results


# Optional: Simple context manager for checkpoint cleanup
class CheckpointContext:
    """
    Context manager for automatic checkpoint cleanup.
    
    Usage:
        with CheckpointContext(checkpoint_file) as checkpoint:
            for batch in batches:
                results = process(batch)
                checkpoint.save(batch_idx, results)
    """
    
    def __init__(self, checkpoint_file: Path):
        self.checkpoint_file = checkpoint_file
        self.results = {}
        self.completed = set()
    
    def __enter__(self):
        self.results, self.completed = load_checkpoint(self.checkpoint_file)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # No exception, clear checkpoint
            clear_checkpoint(self.checkpoint_file)
        # If there was an exception, keep the checkpoint for resume
    
    def save(self, batch_idx: int, batch_results: Dict[str, Any]):
        """Save progress for a batch."""
        self.results.update(batch_results)
        save_checkpoint(self.checkpoint_file, batch_idx, self.results)
    
    def is_completed(self, batch_idx: int) -> bool:
        """Check if a batch was already completed."""
        return batch_idx in self.completed