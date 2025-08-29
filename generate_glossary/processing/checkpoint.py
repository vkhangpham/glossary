"""
Checkpoint and rollback system for expensive LLM processing operations.

This module provides granular checkpointing within processing steps to prevent loss
of expensive LLM work during pipeline failures. Each checkpoint stores partial results
that can be resumed from the exact failure point.
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime

from generate_glossary.utils.logger import setup_logger

logger = setup_logger("checkpoint")

@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint files"""
    step: str  # e.g., "lv0_s1", "lv2_s3" 
    operation: str  # e.g., "extract_concepts", "verify_keywords"
    created_at: str
    total_items: int
    processed_items: int
    batch_size: int
    config_hash: str  # Hash of processing config to detect config changes
    provider_info: Dict[str, Any] = None

@dataclass 
class CheckpointData:
    """Container for checkpoint data"""
    metadata: CheckpointMetadata
    completed_batches: List[int]  # List of completed batch indices
    partial_results: Dict[str, Any]  # Partial processing results
    failed_items: List[str] = None  # Items that failed processing
    resume_index: int = 0  # Next item index to process

class CheckpointManager:
    """Manages checkpointing and recovery for LLM processing operations"""
    
    def __init__(self, checkpoint_dir: Union[Path, str]):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_checkpoint_path(self, step: str, operation: str) -> Path:
        """Get file path for checkpoint"""
        filename = f"{step}_{operation}_checkpoint.json"
        return self.checkpoint_dir / filename
        
    def _get_config_hash(self, config: Dict[str, Any]) -> str:
        """Generate hash of configuration to detect changes"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
        
    def create_checkpoint(
        self,
        step: str,
        operation: str, 
        total_items: int,
        batch_size: int,
        config: Dict[str, Any],
        provider_info: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Create new checkpoint file"""
        config_hash = self._get_config_hash(config)
        
        metadata = CheckpointMetadata(
            step=step,
            operation=operation,
            created_at=datetime.now().isoformat(),
            total_items=total_items,
            processed_items=0,
            batch_size=batch_size,
            config_hash=config_hash,
            provider_info=provider_info or {}
        )
        
        checkpoint_data = CheckpointData(
            metadata=metadata,
            completed_batches=[],
            partial_results={},
            failed_items=[]
        )
        
        checkpoint_path = self._get_checkpoint_path(step, operation)
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(checkpoint_data), f, indent=2)
            
        logger.info(f"Created checkpoint: {checkpoint_path}")
        return checkpoint_path
        
    def save_batch_progress(
        self,
        step: str,
        operation: str,
        batch_index: int,
        batch_results: Dict[str, Any],
        failed_items: Optional[List[str]] = None
    ) -> bool:
        """Save progress after completing a batch"""
        checkpoint_path = self._get_checkpoint_path(step, operation)
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return False
            
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            checkpoint_data = CheckpointData(**data)
            
            # Update progress
            checkpoint_data.completed_batches.append(batch_index)
            checkpoint_data.partial_results.update(batch_results)
            checkpoint_data.metadata.processed_items += len(batch_results)
            
            if failed_items:
                checkpoint_data.failed_items.extend(failed_items)
                
            # Update resume index to next batch start
            checkpoint_data.resume_index = (batch_index + 1) * checkpoint_data.metadata.batch_size
            
            # Save updated checkpoint
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(checkpoint_data), f, indent=2)
                
            progress_pct = (checkpoint_data.metadata.processed_items / 
                          checkpoint_data.metadata.total_items * 100)
            logger.info(f"Saved batch {batch_index} progress: {progress_pct:.1f}% complete")
            return True
            
        except Exception as e:
            logger.error(f"Error saving batch progress: {e}")
            return False
            
    def load_checkpoint(self, step: str, operation: str) -> Optional[CheckpointData]:
        """Load existing checkpoint if available"""
        checkpoint_path = self._get_checkpoint_path(step, operation)
        
        if not checkpoint_path.exists():
            logger.info(f"No checkpoint found: {checkpoint_path}")
            return None
            
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            checkpoint_data = CheckpointData(**data)
            logger.info(f"Loaded checkpoint: {len(checkpoint_data.completed_batches)} batches completed")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None
            
    def can_resume(
        self, 
        step: str, 
        operation: str, 
        current_config: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Check if checkpoint can be resumed with current configuration"""
        checkpoint_data = self.load_checkpoint(step, operation)
        
        if not checkpoint_data:
            return False, "No checkpoint found"
            
        current_hash = self._get_config_hash(current_config)
        if checkpoint_data.metadata.config_hash != current_hash:
            return False, "Configuration changed - checkpoint incompatible"
            
        if checkpoint_data.metadata.processed_items >= checkpoint_data.metadata.total_items:
            return False, "Checkpoint already complete"
            
        return True, None
        
    def get_resume_info(self, step: str, operation: str) -> Optional[Dict[str, Any]]:
        """Get information needed to resume processing"""
        checkpoint_data = self.load_checkpoint(step, operation)
        
        if not checkpoint_data:
            return None
            
        return {
            "completed_batches": set(checkpoint_data.completed_batches),
            "partial_results": checkpoint_data.partial_results,
            "resume_index": checkpoint_data.resume_index,
            "failed_items": checkpoint_data.failed_items or [],
            "progress": {
                "processed": checkpoint_data.metadata.processed_items,
                "total": checkpoint_data.metadata.total_items,
                "percentage": (checkpoint_data.metadata.processed_items / 
                             checkpoint_data.metadata.total_items * 100)
            }
        }
        
    def cleanup_checkpoint(self, step: str, operation: str) -> bool:
        """Remove checkpoint file after successful completion"""
        checkpoint_path = self._get_checkpoint_path(step, operation)
        
        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info(f"Cleaned up checkpoint: {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"Error cleaning up checkpoint: {e}")
            return False
            
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints"""
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("*_checkpoint.json"):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                metadata = data.get("metadata", {})
                checkpoints.append({
                    "file": checkpoint_file.name,
                    "step": metadata.get("step"),
                    "operation": metadata.get("operation"),
                    "created_at": metadata.get("created_at"),
                    "progress": f"{metadata.get('processed_items', 0)}/{metadata.get('total_items', 0)}",
                    "size_mb": checkpoint_file.stat().st_size / (1024 * 1024)
                })
            except Exception as e:
                logger.warning(f"Could not read checkpoint {checkpoint_file}: {e}")
                
        return sorted(checkpoints, key=lambda x: x.get("created_at", ""))

def with_checkpoint(
    checkpoint_manager: CheckpointManager,
    step: str,
    operation: str,
    config: Dict[str, Any],
    force_restart: bool = False
):
    """Decorator to add checkpointing support to processing functions
    
    Usage:
        @with_checkpoint(checkpoint_manager, "lv0_s1", "extract_concepts", config_dict)
        def process_items(items, resume_info=None):
            # Function receives resume_info if resuming from checkpoint
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Check for existing checkpoint
            if not force_restart:
                can_resume, reason = checkpoint_manager.can_resume(step, operation, config)
                if can_resume:
                    resume_info = checkpoint_manager.get_resume_info(step, operation)
                    logger.info(f"Resuming from checkpoint: {resume_info['progress']['percentage']:.1f}% complete")
                    kwargs['resume_info'] = resume_info
                elif reason and "No checkpoint found" not in reason:
                    logger.warning(f"Cannot resume checkpoint: {reason}")
                    
            return func(*args, **kwargs)
        return wrapper
    return decorator