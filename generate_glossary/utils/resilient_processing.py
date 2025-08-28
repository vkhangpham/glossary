"""
Resilient processing utilities with checkpoint support for LLM operations.

This module provides wrapper functions for common LLM processing patterns
with automatic checkpointing and recovery capabilities.
"""

import time
import json
from typing import List, Dict, Any, Optional, Callable, Tuple
from pathlib import Path
from tqdm import tqdm

from generate_glossary.utils.logger import setup_logger
from generate_glossary.utils.checkpoint import CheckpointManager, with_checkpoint

logger = setup_logger("resilient_processing")

class ResilientProcessor:
    """Base class for resilient processing with checkpoint support"""
    
    def __init__(self, checkpoint_dir: Path, step: str, operation: str):
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.step = step
        self.operation = operation
        
    def process_with_checkpoints(
        self,
        items: List[Any],
        process_batch_func: Callable,
        batch_size: int,
        config: Dict[str, Any],
        cooldown_period: int = 2,
        cooldown_frequency: int = 5,
        force_restart: bool = False,
        **batch_func_kwargs
    ) -> Dict[str, Any]:
        """
        Process items in batches with automatic checkpointing
        
        Args:
            items: List of items to process
            process_batch_func: Function to process each batch
            batch_size: Number of items per batch
            config: Processing configuration (used for checkpoint validation)
            cooldown_period: Seconds to sleep between batches
            cooldown_frequency: Number of batches before cooldown
            force_restart: Ignore existing checkpoints and start fresh
            **batch_func_kwargs: Additional arguments for process_batch_func
            
        Returns:
            Dictionary with all processing results
        """
        total_items = len(items)
        all_results = {}
        failed_items = []
        
        # Check for existing checkpoint
        resume_info = None
        if not force_restart:
            can_resume, reason = self.checkpoint_manager.can_resume(
                self.step, self.operation, config
            )
            if can_resume:
                resume_info = self.checkpoint_manager.get_resume_info(
                    self.step, self.operation
                )
                logger.info(f"Resuming from checkpoint: {resume_info['progress']['percentage']:.1f}% complete")
                all_results.update(resume_info['partial_results'])
                failed_items.extend(resume_info['failed_items'])
            elif reason and "No checkpoint found" not in reason:
                logger.warning(f"Cannot resume checkpoint: {reason}")
                
        # Create new checkpoint if not resuming
        if resume_info is None:
            self.checkpoint_manager.create_checkpoint(
                self.step, self.operation, total_items, batch_size, config
            )
            start_index = 0
            completed_batches = set()
        else:
            start_index = resume_info['resume_index']
            completed_batches = resume_info['completed_batches']
            
        # Process batches
        num_batches = (total_items + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc=f"Processing {self.operation}"):
            # Skip already completed batches
            if batch_idx in completed_batches:
                continue
                
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, total_items)
            
            # Skip if this batch is before our resume point
            if batch_start < start_index:
                continue
                
            batch_items = items[batch_start:batch_end]
            
            try:
                logger.info(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_items)} items)")
                
                # Process the batch
                batch_results = process_batch_func(
                    batch_items, 
                    **batch_func_kwargs
                )
                
                # Handle different return types from batch functions
                if isinstance(batch_results, tuple):
                    batch_data, batch_failures = batch_results
                    failed_items.extend(batch_failures)
                else:
                    batch_data = batch_results
                    batch_failures = []
                
                # Update results
                all_results.update(batch_data)
                
                # Save checkpoint
                success = self.checkpoint_manager.save_batch_progress(
                    self.step, self.operation, batch_idx, batch_data, batch_failures
                )
                
                if not success:
                    logger.warning(f"Failed to save checkpoint for batch {batch_idx}")
                
                # Apply cooldown
                if (batch_idx + 1) % cooldown_frequency == 0 and batch_idx < num_batches - 1:
                    logger.info(f"Cooldown period: {cooldown_period}s")
                    time.sleep(cooldown_period)
                    
            except Exception as e:
                logger.error(f"Batch {batch_idx} failed: {e}")
                # Mark all items in this batch as failed
                batch_item_keys = [str(item) for item in batch_items]
                failed_items.extend(batch_item_keys)
                
                # Save checkpoint with failures
                self.checkpoint_manager.save_batch_progress(
                    self.step, self.operation, batch_idx, {}, batch_item_keys
                )
                
                # Continue processing remaining batches
                continue
        
        # Processing complete - cleanup checkpoint
        self.checkpoint_manager.cleanup_checkpoint(self.step, self.operation)
        
        return {
            "results": all_results,
            "failed_items": failed_items,
            "total_processed": len(all_results),
            "total_failed": len(failed_items)
        }

class ConceptExtractionProcessor(ResilientProcessor):
    """Specialized processor for concept extraction operations"""
    
    def __init__(self, checkpoint_dir: Path, level: int):
        super().__init__(checkpoint_dir, f"lv{level}_s1", "extract_concepts")
        
    def process_sources(
        self,
        sources: List[str],
        process_chunk_func: Callable,
        batch_size: int,
        chunk_size: int,
        max_workers: int,
        llm_attempts: int,
        config: Dict[str, Any],
        force_restart: bool = False
    ) -> Dict[str, List[str]]:
        """
        Process sources for concept extraction with checkpointing
        
        Returns:
            Dictionary mapping sources to extracted concepts
        """
        from pydash import chunk as create_chunks
        
        # Split sources into chunks for parallel processing
        source_chunks = list(create_chunks(sources, chunk_size))
        
        def process_batch(chunks_batch):
            """Process a batch of source chunks"""
            batch_results = {}
            
            for chunk in chunks_batch:
                try:
                    chunk_results = process_chunk_func(
                        (chunk, llm_attempts)
                    )
                    batch_results.update(chunk_results)
                except Exception as e:
                    logger.error(f"Chunk processing failed: {e}")
                    # Mark all sources in chunk as failed
                    for source in chunk:
                        batch_results[source] = []
                        
            return batch_results
            
        # Process with checkpoints
        processing_result = self.process_with_checkpoints(
            source_chunks,
            process_batch,
            batch_size=1,  # Process one chunk at a time for granular checkpointing
            config=config,
            force_restart=force_restart
        )
        
        return processing_result["results"]

class KeywordVerificationProcessor(ResilientProcessor):
    """Specialized processor for keyword verification operations"""
    
    def __init__(self, checkpoint_dir: Path, level: int):
        super().__init__(checkpoint_dir, f"lv{level}_s3", "verify_keywords")
        
    def verify_keywords(
        self,
        keywords: List[str],
        concept_journals: Dict[str, List[str]],
        verify_func: Callable,
        batch_size: int,
        cooldown_period: int,
        cooldown_frequency: int,
        config: Dict[str, Any],
        provider: Optional[str] = None,
        force_restart: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Verify keywords with checkpointing support
        
        Returns:
            Dictionary with verification results for each keyword
        """
        
        def process_batch(keywords_batch):
            """Process a batch of keywords for verification"""
            batch_results = {}
            batch_failures = []
            
            for keyword in keywords_batch:
                try:
                    journals = concept_journals.get(keyword, [])
                    is_verified = verify_func(keyword, journals, provider)
                    
                    batch_results[keyword] = {
                        "is_verified": is_verified,
                        "journals": journals,
                        "provider": provider or "openai",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                except Exception as e:
                    logger.error(f"Failed to verify '{keyword}': {e}")
                    batch_failures.append(keyword)
                    batch_results[keyword] = {
                        "is_verified": False,
                        "journals": [],
                        "provider": provider or "openai", 
                        "error": str(e),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
            return batch_results, batch_failures
            
        # Process with checkpoints
        processing_result = self.process_with_checkpoints(
            keywords,
            process_batch,
            batch_size=batch_size,
            config=config,
            cooldown_period=cooldown_period,
            cooldown_frequency=cooldown_frequency,
            force_restart=force_restart
        )
        
        return processing_result["results"]

def create_processing_config(processing_config_obj) -> Dict[str, Any]:
    """Convert processing config object to dictionary for checkpointing"""
    if hasattr(processing_config_obj, '__dict__'):
        return processing_config_obj.__dict__.copy()
    else:
        return dict(processing_config_obj)

def get_checkpoint_dir(level_config) -> Path:
    """Get checkpoint directory for a level"""
    checkpoint_dir = level_config.data_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir