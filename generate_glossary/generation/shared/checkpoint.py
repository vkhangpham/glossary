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
from pathlib import Path
from typing import Dict, Set, Tuple, Any, List, Callable, Optional


def save_checkpoint(checkpoint_file: Path, batch_idx: int, results: Dict[str, Any]) -> None:
    """
    Save progress checkpoint after completing a batch.
    
    Args:
        checkpoint_file: Path to checkpoint file
        batch_idx: Index of completed batch
        results: Accumulated results so far
    """
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint_data = {
        "last_batch": batch_idx,
        "completed_batches": list(range(batch_idx + 1)),
        "results": results
    }
    
    # Write atomically by using a temp file
    temp_file = checkpoint_file.with_suffix('.tmp')
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    temp_file.replace(checkpoint_file)


def load_checkpoint(checkpoint_file: Path) -> Tuple[Dict[str, Any], Set[int]]:
    """
    Load checkpoint if it exists.
    
    Args:
        checkpoint_file: Path to checkpoint file
        
    Returns:
        Tuple of (results dict, set of completed batch indices)
    """
    if not checkpoint_file.exists():
        return {}, set()
    
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data.get("results", {}), set(data.get("completed_batches", []))
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Corrupted checkpoint file, starting fresh: {e}")
        return {}, set()


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
    Process items in batches with automatic checkpointing.
    
    This is a convenience function that handles the checkpoint save/load
    cycle automatically.
    
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
    
    # Load any existing checkpoint
    results, completed_batches = load_checkpoint(checkpoint_file)
    
    # Split items into batches
    batches = list(chunk(items, batch_size))
    
    # Process remaining batches
    for batch_idx, batch in enumerate(batches):
        if batch_idx in completed_batches:
            continue  # Skip already completed batches
        
        # Process this batch
        batch_results = process_batch_func(batch, **kwargs)
        
        # Update results
        if isinstance(batch_results, dict):
            results.update(batch_results)
        else:
            # If batch_results is not a dict, store with batch index as key
            results[f"batch_{batch_idx}"] = batch_results
        
        # Save checkpoint
        save_checkpoint(checkpoint_file, batch_idx, results)
        
        print(f"Completed batch {batch_idx + 1}/{len(batches)}")
    
    # Clear checkpoint after successful completion
    clear_checkpoint(checkpoint_file)
    
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