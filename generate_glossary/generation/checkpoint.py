"""
Enhanced checkpoint utility for generation modules with reliability improvements.

Features:
- File locking to prevent race conditions
- Enhanced atomic operations with validation
- Corruption detection and recovery
- Integration with failure tracking
- Cross-platform compatibility
- Backward compatibility with existing code

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
import os
import time
from pathlib import Path
from typing import Dict, Set, Tuple, Any, List, Callable, Optional
import hashlib

try:
    from filelock import FileLock
except ImportError:
    FileLock = None

# Checkpoint version for backward compatibility
CHECKPOINT_VERSION = "1.2"


def _log_failure(module: str, function: str, error_type: str, error_message: str, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log failure to the failure tracking system.
    
    Args:
        module: Module name
        function: Function name
        error_type: Error type
        error_message: Error message
        context: Additional context
    """
    try:
        from ..utils.failure_tracker import save_failure
        save_failure(module, function, error_type, error_message, context)
    except ImportError:
        # Failure tracking not available, use logging instead
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"{module}.{function}: {error_type}: {error_message}")


def _aux_path(path: Path, ext: str) -> Path:
    """
    Helper to create auxiliary file paths by appending suffixes.
    
    Args:
        path: Base file path
        ext: Extension to append (e.g., '.tmp', '.lock')
        
    Returns:
        Path with appended extension
    """
    return path.with_name(path.name + ext)


def _get_lock_file(checkpoint_file: Path) -> Optional[object]:
    """
    Get a file lock for checkpoint operations.
    
    Args:
        checkpoint_file: Path to checkpoint file
        
    Returns:
        FileLock object or None if filelock not available
    """
    if FileLock is None:
        return None
    
    lock_file = _aux_path(checkpoint_file, '.lock')
    return FileLock(str(lock_file))


def _calculate_checksum(data: str) -> str:
    """
    Calculate MD5 checksum of data for corruption detection.
    
    Args:
        data: String data to checksum
        
    Returns:
        MD5 checksum as hex string
    """
    return hashlib.md5(data.encode('utf-8')).hexdigest()


def _validate_file_integrity(file_path: Path, expected_checksum: Optional[str] = None) -> bool:
    """
    Validate file integrity using size and checksum checks.
    
    Args:
        file_path: Path to file to validate
        expected_checksum: Optional expected checksum
        
    Returns:
        True if file appears valid
    """
    if not file_path.exists():
        return False
    
    try:
        # Basic size check - empty files are invalid
        if file_path.stat().st_size == 0:
            return False
        
        # If we have expected checksum, verify it
        if expected_checksum:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            actual_checksum = _calculate_checksum(content)
            return actual_checksum == expected_checksum
        
        # Basic JSON validation
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True
    
    except (OSError, json.JSONDecodeError):
        return False


def cleanup_orphaned_temps(checkpoint_dir: Path) -> int:
    """
    Clean up orphaned .tmp files from interrupted operations with improved detection.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        Number of orphaned temp files cleaned
    """
    if not checkpoint_dir.exists():
        return 0
    
    cleaned_count = 0
    current_time = time.time()
    
    for tmp_file in checkpoint_dir.glob("*.tmp"):
        try:
            # Check if corresponding .json file exists
            json_file = tmp_file.with_suffix('.json')
            
            # Check file age (orphaned if > 1 hour old)
            file_age = current_time - tmp_file.stat().st_mtime
            is_old = file_age > 3600  # 1 hour
            
            # Check if this is truly orphaned
            is_orphaned = not json_file.exists() or is_old
            
            if is_orphaned:
                # Validate the temp file isn't corrupted before deciding to clean
                if not _validate_file_integrity(tmp_file):
                    print(f"Cleaning corrupted temp file: {tmp_file.name}")
                else:
                    print(f"Cleaning orphaned temp file: {tmp_file.name} (age: {file_age:.1f}s)")
                
                tmp_file.unlink()
                cleaned_count += 1
                
        except OSError as e:
            print(f"Warning: Could not process temp file {tmp_file.name}: {e}")
            _log_failure("checkpoint", "cleanup_orphaned_temps", "OSError", str(e), 
                        {"temp_file": str(tmp_file)})
    
    return cleaned_count


def detect_interrupted_checkpoint(checkpoint_file: Path) -> bool:
    """
    Detect if there's an interrupted checkpoint (temp file without json file).
    
    Args:
        checkpoint_file: Path to checkpoint file
        
    Returns:
        True if interrupted checkpoint detected
    """
    temp_file = _aux_path(checkpoint_file, '.tmp')
    return temp_file.exists() and not checkpoint_file.exists()


def save_checkpoint(checkpoint_file: Path, batch_idx: int, results: Dict[str, Any], max_retries: int = 3) -> None:
    """
    Save progress checkpoint with file locking, validation, and enhanced error recovery.
    
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
    
    # Serialize data once for checksum calculation
    json_data = json.dumps(checkpoint_data, indent=2, ensure_ascii=False, sort_keys=True)
    checksum = _calculate_checksum(json_data)
    checkpoint_data["checksum"] = checksum
    
    # Re-serialize with checksum
    json_data = json.dumps(checkpoint_data, indent=2, ensure_ascii=False, sort_keys=True)
    
    # Get file lock if available
    lock = _get_lock_file(checkpoint_file)
    temp_file = _aux_path(checkpoint_file, '.tmp')
    
    for attempt in range(max_retries):
        try:
            if lock:
                with lock:
                    _save_checkpoint_atomic(temp_file, checkpoint_file, json_data)
            else:
                _save_checkpoint_atomic(temp_file, checkpoint_file, json_data)
            
            # Validate the saved file
            if not _validate_file_integrity(checkpoint_file):
                raise OSError("Checkpoint validation failed after save")
                
            return  # Success
            
        except (OSError, IOError, TypeError, json.JSONDecodeError) as e:
            error_context = {
                "checkpoint_file": str(checkpoint_file),
                "batch_idx": batch_idx,
                "attempt": attempt + 1,
                "max_retries": max_retries
            }
            
            if attempt < max_retries - 1:
                print(f"Warning: Failed to save checkpoint (attempt {attempt + 1}/{max_retries}): {e}")
                _log_failure("checkpoint", "save_checkpoint", type(e).__name__, str(e), error_context)
                time.sleep(0.5)  # Brief pause before retry
            else:
                print(f"Error: Failed to save checkpoint after {max_retries} attempts: {e}")
                _log_failure("checkpoint", "save_checkpoint", type(e).__name__, str(e), error_context)
                raise


def _save_checkpoint_atomic(temp_file: Path, final_file: Path, json_data: str) -> None:
    """
    Perform atomic checkpoint save with validation.
    
    Args:
        temp_file: Temporary file path
        final_file: Final checkpoint file path
        json_data: JSON data to write
    """
    # Clean up any existing temp file first
    temp_file.unlink(missing_ok=True)
    
    try:
        # Write to temp file
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(json_data)
            f.flush()  # Ensure data is written to disk
            os.fsync(f.fileno())  # Force OS to write to disk
        
        # Validate temp file before moving
        if not _validate_file_integrity(temp_file):
            raise OSError("Temporary checkpoint file failed validation")
        
        # Atomic rename
        temp_file.replace(final_file)
        
    except Exception as e:
        # Clean up failed temp file
        temp_file.unlink(missing_ok=True)
        raise


def load_checkpoint(checkpoint_file: Path) -> Tuple[Dict[str, Any], Set[int]]:
    """
    Load checkpoint with enhanced validation and recovery capabilities.
    
    Args:
        checkpoint_file: Path to checkpoint file
        
    Returns:
        Tuple of (results dict, set of completed batch indices)
    """
    # Get file lock if available
    lock = _get_lock_file(checkpoint_file)
    
    try:
        if lock:
            with lock:
                return _load_checkpoint_locked(checkpoint_file)
        else:
            return _load_checkpoint_locked(checkpoint_file)
    except Exception as e:
        _log_failure("checkpoint", "load_checkpoint", type(e).__name__, str(e), 
                    {"checkpoint_file": str(checkpoint_file)})
        print(f"Warning: Failed to load checkpoint: {e}")
        return {}, set()


def _load_checkpoint_locked(checkpoint_file: Path) -> Tuple[Dict[str, Any], Set[int]]:
    """
    Load checkpoint implementation with enhanced validation.
    
    Args:
        checkpoint_file: Path to checkpoint file
        
    Returns:
        Tuple of (results dict, set of completed batch indices)
    """
    # Check for interrupted checkpoint first
    if detect_interrupted_checkpoint(checkpoint_file):
        temp_file = _aux_path(checkpoint_file, '.tmp')
        print(f"Found interrupted checkpoint, attempting recovery from: {temp_file.name}")
        
        recovery_result = _attempt_recovery_from_temp(temp_file, checkpoint_file)
        if recovery_result:
            return recovery_result
        
        print("Warning: Could not recover from interrupted checkpoint, starting fresh")
        return {}, set()
    
    if not checkpoint_file.exists():
        return {}, set()
    
    # Validate file integrity before loading
    if not _validate_file_integrity(checkpoint_file):
        print("Warning: Checkpoint file appears corrupted, attempting recovery")
        
        # Try to find backup or temp file
        temp_file = _aux_path(checkpoint_file, '.tmp')
        if temp_file.exists():
            recovery_result = _attempt_recovery_from_temp(temp_file, checkpoint_file)
            if recovery_result:
                return recovery_result
        
        print("Warning: Could not recover checkpoint, starting fresh")
        # Move corrupted file to backup
        backup_file = _aux_path(checkpoint_file, '.corrupted')
        try:
            checkpoint_file.rename(backup_file)
            print(f"Moved corrupted checkpoint to: {backup_file.name}")
        except OSError:
            pass
        
        return {}, set()
    
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            content = f.read()
            data = json.loads(content)
        
        # Validate checksum if present
        if "checksum" in data:
            # Remove checksum for validation
            checksum = data.pop("checksum")
            expected_checksum = _calculate_checksum(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))
            
            if checksum != expected_checksum:
                print("Warning: Checkpoint checksum mismatch, data may be corrupted")
                _log_failure("checkpoint", "_load_checkpoint_locked", "ChecksumMismatch", 
                           f"Expected {expected_checksum}, got {checksum}",
                           {"checkpoint_file": str(checkpoint_file)})
                # Continue loading but warn user
        
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
        _log_failure("checkpoint", "_load_checkpoint_locked", type(e).__name__, str(e),
                    {"checkpoint_file": str(checkpoint_file)})
        return {}, set()


def _attempt_recovery_from_temp(temp_file: Path, checkpoint_file: Path) -> Optional[Tuple[Dict[str, Any], Set[int]]]:
    """
    Attempt to recover checkpoint data from temporary file.
    
    Args:
        temp_file: Path to temporary file
        checkpoint_file: Path to main checkpoint file
        
    Returns:
        Recovery result or None if failed
    """
    try:
        # Validate temp file integrity first
        if not _validate_file_integrity(temp_file):
            print("Warning: Temp file appears corrupted")
            return None
            
        with open(temp_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate structure
        if _validate_checkpoint_structure(data):
            print(f"Successfully recovered from interrupted checkpoint (batch {data.get('last_batch', 0)})")
            # Move temp to main file for next save
            temp_file.replace(checkpoint_file)
            return data.get("results", {}), set(data.get("completed_batches", []))
        else:
            print("Warning: Interrupted checkpoint has invalid structure")
            return None
            
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Could not recover from interrupted checkpoint: {e}")
        _log_failure("checkpoint", "_attempt_recovery_from_temp", type(e).__name__, str(e),
                    {"temp_file": str(temp_file)})
        return None


def _validate_checkpoint_structure(data: Any) -> bool:
    """
    Validate checkpoint data structure with comprehensive checks.
    
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
    
    # Additional validation
    try:
        # Validate batch indices are non-negative
        if data["last_batch"] < 0:
            return False
        
        # Validate completed_batches contains only integers
        for batch_idx in data["completed_batches"]:
            if not isinstance(batch_idx, int) or batch_idx < 0:
                return False
        
        # Validate timestamp if present
        if "timestamp" in data:
            timestamp = data["timestamp"]
            if not isinstance(timestamp, (int, float)) or timestamp <= 0:
                return False
        
        # Validate version if present
        if "version" in data:
            version = data["version"]
            if not isinstance(version, str):
                return False
        
        # Basic size check - extremely large checkpoints are suspicious
        import sys
        checkpoint_size = sys.getsizeof(data)
        if checkpoint_size > 500 * 1024 * 1024:  # 500MB limit
            print(f"Warning: Checkpoint is very large ({checkpoint_size // (1024*1024)}MB)")
    
    except (TypeError, ValueError):
        return False
    
    return True


def clear_checkpoint(checkpoint_file: Path) -> None:
    """Remove checkpoint file and related files after successful completion."""
    # Use file locking if available
    lock = _get_lock_file(checkpoint_file)
    
    try:
        if lock:
            with lock:
                _clear_checkpoint_locked(checkpoint_file)
        else:
            _clear_checkpoint_locked(checkpoint_file)
    except Exception as e:
        print(f"Warning: Could not fully clear checkpoint files: {e}")
        _log_failure("checkpoint", "clear_checkpoint", type(e).__name__, str(e),
                    {"checkpoint_file": str(checkpoint_file)})


def _clear_checkpoint_locked(checkpoint_file: Path) -> None:
    """Clear checkpoint files with proper cleanup."""
    # Remove main checkpoint file
    checkpoint_file.unlink(missing_ok=True)
    
    # Remove temp file if it exists
    temp_file = _aux_path(checkpoint_file, '.tmp')
    temp_file.unlink(missing_ok=True)
    
    # Remove lock file if it exists (cleanup)
    lock_file = _aux_path(checkpoint_file, '.lock')
    lock_file.unlink(missing_ok=True)
    
    # Remove any corrupted backup files
    corrupted_file = _aux_path(checkpoint_file, '.corrupted')
    corrupted_file.unlink(missing_ok=True)


def process_with_checkpoint(
    items: List[Any],
    batch_size: int,
    checkpoint_file: Path,
    process_batch_func: Callable[[List[Any]], Dict[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    """
    Process items in batches with enhanced checkpointing, file locking, and error recovery.
    
    This is a convenience function that handles the checkpoint save/load
    cycle automatically, with improved reliability features:
    - File locking to prevent race conditions
    - Enhanced error recovery and logging  
    - Orphan cleanup and corruption detection
    - Resume detection with validation
    
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
    
    try:
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
        
        # Validate batch size
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if total_batches == 0:
            print("Warning: No items to process")
            return {}
        
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
        failed_batches = []
        
        # Process remaining batches
        for batch_idx, batch in enumerate(batches):
            if batch_idx in completed_batches:
                continue  # Skip already completed batches
            
            batch_start = time.time()
            
            try:
                # Process this batch
                batch_results = process_batch_func(batch, **kwargs)
                
                # Validate batch results
                if batch_results is None:
                    print(f"Warning: Batch {batch_idx} returned None results")
                    batch_results = {}
                
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
                
                # Save checkpoint with retry logic
                try:
                    save_checkpoint(checkpoint_file, batch_idx, results)
                except Exception as checkpoint_error:
                    print(f"Critical error: Failed to save checkpoint for batch {batch_idx}: {checkpoint_error}")
                    _log_failure("checkpoint", "process_with_checkpoint", 
                                type(checkpoint_error).__name__, str(checkpoint_error),
                                {"batch_idx": batch_idx, "checkpoint_file": str(checkpoint_file)})
                    raise  # Re-raise as this is critical
                
            except Exception as batch_error:
                print(f"Error processing batch {batch_idx}: {batch_error}")
                failed_batches.append(batch_idx)
                _log_failure("checkpoint", "process_with_checkpoint", 
                           f"BatchProcessingError", str(batch_error),
                           {"batch_idx": batch_idx, "batch_size": len(batch)})
                
                # For now, continue with other batches - could be made configurable
                continue
            
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
        
        # Report any failures
        if failed_batches:
            print(f"Warning: {len(failed_batches)} batches failed: {failed_batches}")
            _log_failure("checkpoint", "process_with_checkpoint", "BatchFailures",
                        f"Failed batches: {failed_batches}",
                        {"failed_count": len(failed_batches), "total_batches": total_batches})
        
        # Clear checkpoint after successful completion
        clear_checkpoint(checkpoint_file)
        
        total_time = time.time() - start_time
        success_rate = ((total_batches - len(failed_batches)) / total_batches) * 100 if total_batches > 0 else 0
        print(f"Processing complete! Total time: {total_time:.1f}s, Success rate: {success_rate:.1f}%")
        
        return results
        
    except Exception as e:
        _log_failure("checkpoint", "process_with_checkpoint", type(e).__name__, str(e),
                    {"checkpoint_file": str(checkpoint_file), "total_items": len(items)})
        raise


# Optional: Enhanced context manager for checkpoint cleanup
class CheckpointContext:
    """
    Enhanced context manager for automatic checkpoint cleanup with file locking support.
    
    Features:
    - Automatic checkpoint loading/saving
    - File locking when available
    - Enhanced error handling and logging
    - Automatic cleanup on success
    
    Usage:
        with CheckpointContext(checkpoint_file) as checkpoint:
            for batch_idx, batch in enumerate(batches):
                if not checkpoint.is_completed(batch_idx):
                    results = process(batch)
                    checkpoint.save(batch_idx, results)
    """
    
    def __init__(self, checkpoint_file: Path):
        self.checkpoint_file = checkpoint_file
        self.results = {}
        self.completed = set()
        self.lock = _get_lock_file(checkpoint_file)
        self._error_occurred = False
    
    def __enter__(self):
        try:
            # Clean up orphaned files first
            cleaned = cleanup_orphaned_temps(self.checkpoint_file.parent)
            if cleaned > 0:
                print(f"CheckpointContext: Cleaned {cleaned} orphaned temp file(s)")
            
            # Load existing checkpoint
            self.results, self.completed = load_checkpoint(self.checkpoint_file)
            return self
        except Exception as e:
            _log_failure("checkpoint", "CheckpointContext.__enter__", type(e).__name__, str(e),
                        {"checkpoint_file": str(self.checkpoint_file)})
            self._error_occurred = True
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and not self._error_occurred:
            # No exception, clear checkpoint
            try:
                clear_checkpoint(self.checkpoint_file)
            except Exception as e:
                print(f"Warning: Could not clear checkpoint: {e}")
                _log_failure("checkpoint", "CheckpointContext.__exit__", type(e).__name__, str(e),
                            {"checkpoint_file": str(self.checkpoint_file)})
        # If there was an exception, keep the checkpoint for resume
    
    def save(self, batch_idx: int, batch_results: Dict[str, Any]):
        """
        Save progress for a batch with enhanced error handling.
        
        Args:
            batch_idx: Index of the completed batch
            batch_results: Results from processing the batch
        """
        try:
            # Validate input
            if not isinstance(batch_results, dict):
                print(f"Warning: Batch {batch_idx} results are not a dictionary, converting")
                batch_results = {"result": batch_results}
            
            # Update results
            overlapping_keys = set(self.results.keys()) & set(batch_results.keys())
            if overlapping_keys:
                print(f"Warning: Batch {batch_idx} has {len(overlapping_keys)} overlapping keys")
            
            self.results.update(batch_results)
            
            # Save checkpoint
            save_checkpoint(self.checkpoint_file, batch_idx, self.results)
            
            # Update completed set
            self.completed.add(batch_idx)
            
        except Exception as e:
            self._error_occurred = True
            _log_failure("checkpoint", "CheckpointContext.save", type(e).__name__, str(e),
                        {"batch_idx": batch_idx, "checkpoint_file": str(self.checkpoint_file)})
            raise
    
    def is_completed(self, batch_idx: int) -> bool:
        """
        Check if a batch was already completed.
        
        Args:
            batch_idx: Index of the batch to check
            
        Returns:
            True if the batch was already completed
        """
        return batch_idx in self.completed
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get current progress information.
        
        Returns:
            Dictionary with progress information
        """
        return {
            "completed_batches": len(self.completed),
            "completed_indices": sorted(list(self.completed)),
            "results_count": len(self.results),
            "checkpoint_file": str(self.checkpoint_file),
            "file_locking_available": FileLock is not None
        }