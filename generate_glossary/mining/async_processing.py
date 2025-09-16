"""Async processing and concurrency management module.

This module provides advanced concurrency patterns, resource-aware scheduling,
streaming processing for large datasets, pipeline execution with parallel stages,
and intelligent batching and aggregation for the mining system.
"""

import time
import asyncio
import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional, Callable
from firecrawl import FirecrawlApp

from .models import ApiUsageStats
from generate_glossary.utils.error_handler import processing_context
from generate_glossary.utils.logger import get_logger, log_with_context


logger = get_logger(__name__)


class ConcurrencyManager:
    """Advanced semaphore-based concurrency control for different operation types."""

    def __init__(self):
        self.semaphores = {}
        self.active_operations = defaultdict(int)
        self.operation_history = defaultdict(list)
        self.resource_limits = {
            'search': 10,
            'map': 8,
            'scrape': 15,
            'extract': 6,
            'batch_scrape': 3
        }
        self.adaptive_limits = self.resource_limits.copy()

    def get_semaphore(self, operation_type: str) -> asyncio.Semaphore:
        """Get or create semaphore for operation type with adaptive limits."""
        if operation_type not in self.semaphores:
            limit = self.adaptive_limits.get(operation_type, 5)
            self.semaphores[operation_type] = asyncio.Semaphore(limit)
        return self.semaphores[operation_type]

    async def acquire_resource(self, operation_type: str, correlation_id: str = None):
        """Acquire resource with performance tracking."""
        semaphore = self.get_semaphore(operation_type)
        start_time = time.time()

        await semaphore.acquire()

        # Track acquisition time and active operations
        acquisition_time = time.time() - start_time
        self.active_operations[operation_type] += 1
        self.operation_history[operation_type].append({
            'timestamp': time.time(),
            'action': 'acquire',
            'wait_time': acquisition_time,
            'correlation_id': correlation_id
        })

        if acquisition_time > 1.0:  # Log if waiting more than 1 second
            log_with_context(
                logger,
                logging.DEBUG,
                f"Resource acquisition for {operation_type} took {acquisition_time:.1f}s",
                correlation_id=correlation_id
            )

    def release_resource(self, operation_type: str, correlation_id: str = None):
        """Release resource and update tracking."""
        if operation_type in self.semaphores:
            semaphore = self.semaphores[operation_type]
            semaphore.release()

            # Update tracking
            self.active_operations[operation_type] = max(0, self.active_operations[operation_type] - 1)
            self.operation_history[operation_type].append({
                'timestamp': time.time(),
                'action': 'release',
                'correlation_id': correlation_id
            })

            # Clean old history
            if len(self.operation_history[operation_type]) > 1000:
                self.operation_history[operation_type] = self.operation_history[operation_type][-500:]

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all operation types."""
        stats = {}

        for op_type, history in self.operation_history.items():
            if not history:
                continue

            # Calculate wait times from acquisitions
            wait_times = [entry['wait_time'] for entry in history
                         if entry['action'] == 'acquire' and 'wait_time' in entry]

            if wait_times:
                stats[op_type] = {
                    'active_operations': self.active_operations[op_type],
                    'resource_limit': self.adaptive_limits.get(op_type, 5),
                    'avg_wait_time': sum(wait_times) / len(wait_times),
                    'max_wait_time': max(wait_times),
                    'total_operations': len([e for e in history if e['action'] == 'acquire']),
                    'utilization': self.active_operations[op_type] / max(1, self.adaptive_limits.get(op_type, 5))
                }

        return stats

    def adapt_limits_based_on_performance(self):
        """Dynamically adjust resource limits based on performance metrics."""
        stats = self.get_performance_stats()

        for op_type, stat in stats.items():
            current_limit = self.adaptive_limits.get(op_type, 5)
            utilization = stat['utilization']
            avg_wait_time = stat['avg_wait_time']

            # Increase limit if high utilization but low wait time (room for more)
            if utilization > 0.8 and avg_wait_time < 0.5:
                new_limit = min(current_limit + 2, 25)  # Cap at 25
                self.adaptive_limits[op_type] = new_limit
                # Recreate semaphore with new limit
                self.semaphores[op_type] = asyncio.Semaphore(new_limit)

            # Decrease limit if high wait times (resource contention)
            elif avg_wait_time > 2.0:
                new_limit = max(current_limit - 1, 2)  # Minimum 2
                self.adaptive_limits[op_type] = new_limit
                # Recreate semaphore with new limit
                self.semaphores[op_type] = asyncio.Semaphore(new_limit)

    def reset_stats(self):
        """Reset performance statistics."""
        self.operation_history.clear()
        self.active_operations.clear()


class AsyncResultAggregator:
    """Async context manager for streaming result aggregation with intelligent buffering."""

    def __init__(self, buffer_size: int = 100, flush_interval: float = 5.0):
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.results = []
        self.buffer = []
        self.last_flush = time.time()
        self.stats = {
            'items_processed': 0,
            'flush_count': 0,
            'errors': 0
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.buffer:
            await self._flush_buffer()

    async def add_result(self, result: Any, metadata: Dict[str, Any] = None):
        """Add result to aggregator with optional metadata."""
        try:
            item = {
                'result': result,
                'timestamp': time.time(),
                'metadata': metadata or {}
            }
            self.buffer.append(item)
            self.stats['items_processed'] += 1

            # Check if we need to flush
            current_time = time.time()
            should_flush = (
                len(self.buffer) >= self.buffer_size or
                current_time - self.last_flush >= self.flush_interval
            )

            if should_flush:
                await self._flush_buffer()

        except Exception as e:
            self.stats['errors'] += 1
            logger.warning(f"Failed to add result to aggregator: {e}")

    async def _flush_buffer(self):
        """Flush buffer to main results with performance optimization."""
        if not self.buffer:
            return

        # Move buffer to results efficiently
        self.results.extend(self.buffer)
        self.buffer.clear()
        self.last_flush = time.time()
        self.stats['flush_count'] += 1

        logger.debug(f"Flushed {len(self.buffer)} results to aggregator")

    def get_results(self) -> List[Any]:
        """Get all aggregated results."""
        return [item['result'] for item in self.results]

    def get_results_with_metadata(self) -> List[Dict[str, Any]]:
        """Get all results with their metadata."""
        return self.results.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics."""
        return {
            **self.stats,
            'total_results': len(self.results),
            'buffer_size': len(self.buffer),
            'avg_flush_size': self.stats['items_processed'] / max(1, self.stats['flush_count'])
        }


# Global concurrency manager instance
_concurrency_manager = ConcurrencyManager()


async def execute_with_resource_management(operation_type: str, operation_func: Callable,
                                         api_usage_stats: Optional[ApiUsageStats] = None,
                                         correlation_id: str = None, *args, **kwargs):
    """Execute operation with intelligent resource management and monitoring.

    Args:
        operation_type: Type of operation for resource management
        operation_func: Function to execute (can be async or sync)
        api_usage_stats: Optional API usage stats for tracking
        correlation_id: Correlation ID for logging
        *args: Arguments for operation_func
        **kwargs: Keyword arguments for operation_func

    Returns:
        Result of operation_func
    """
    await _concurrency_manager.acquire_resource(operation_type, correlation_id)

    try:
        start_time = time.time()

        # Handle both async and sync functions
        if asyncio.iscoroutinefunction(operation_func):
            result = await operation_func(*args, **kwargs)
        else:
            result = await asyncio.to_thread(operation_func, *args, **kwargs)

        duration = time.time() - start_time

        # Track performance if stats provided
        if api_usage_stats:
            api_usage_stats.add_call(operation_type, duration=duration)

        return result

    except Exception as e:
        duration = time.time() - start_time
        if api_usage_stats:
            api_usage_stats.add_call(operation_type, duration=duration, error=True)
        raise
    finally:
        _concurrency_manager.release_resource(operation_type, correlation_id)


async def process_with_streaming(items: List[Any], processor_func: Callable,
                               batch_size: int = 10, operation_type: str = "batch",
                               api_usage_stats: Optional[ApiUsageStats] = None,
                               correlation_id: str = None) -> List[Any]:
    """Process items in batches with streaming results and intelligent work distribution.

    Args:
        items: List of items to process
        processor_func: Function to process each item (can be async or sync)
        batch_size: Number of items per batch
        operation_type: Operation type for resource management
        api_usage_stats: Optional API usage stats for tracking
        correlation_id: Correlation ID for logging

    Returns:
        List of processed results
    """
    if not items:
        return []

    results = []
    total_items = len(items)
    processed_count = 0

    # Adaptive batch size based on current system load
    current_stats = _concurrency_manager.get_performance_stats()
    if operation_type in current_stats:
        avg_wait_time = current_stats[operation_type].get('avg_wait_time', 0)
        if avg_wait_time > 1.0:  # High contention
            batch_size = max(5, batch_size // 2)
        elif avg_wait_time < 0.2:  # Low contention
            batch_size = min(20, batch_size * 2)

    log_with_context(
        logger,
        logging.INFO,
        f"Starting batch processing: {total_items} items, batch_size={batch_size}",
        correlation_id=correlation_id
    )

    for i in range(0, total_items, batch_size):
        batch = items[i:i + batch_size]
        batch_start_time = time.time()

        # Process batch concurrently with resource management
        batch_tasks = []
        for item in batch:
            task = execute_with_resource_management(
                operation_type, processor_func, api_usage_stats, correlation_id, item
            )
            batch_tasks.append(task)

        # Execute batch and collect results
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # Process results and handle exceptions
        for j, result in enumerate(batch_results):
            if isinstance(result, Exception):
                log_with_context(
                    logger,
                    logging.WARNING,
                    f"Item {i + j} failed: {result}",
                    correlation_id=correlation_id
                )
                results.append(None)  # Or handle error differently
            else:
                results.append(result)

        processed_count += len(batch)
        batch_duration = time.time() - batch_start_time

        # Progress logging
        progress_pct = (processed_count / total_items) * 100
        items_per_sec = len(batch) / batch_duration
        log_with_context(
            logger,
            logging.INFO,
            f"Batch progress: {processed_count}/{total_items} ({progress_pct:.1f}%), "
            f"rate: {items_per_sec:.1f} items/sec",
            correlation_id=correlation_id
        )

        # Adaptive performance tuning every 3 batches
        if i % (batch_size * 3) == 0:
            _concurrency_manager.adapt_limits_based_on_performance()

    log_with_context(
        logger,
        logging.INFO,
        f"Batch processing completed: {processed_count} items processed",
        correlation_id=correlation_id
    )

    return results


async def execute_parallel_pipeline(stages: List[Tuple[str, Callable]],
                                   initial_data: Any,
                                   api_usage_stats: Optional[ApiUsageStats] = None,
                                   correlation_id: str = None) -> Any:
    """Execute a multi-stage processing pipeline with parallel execution where possible.

    Args:
        stages: List of (stage_name, stage_function) tuples
        initial_data: Initial data to process
        api_usage_stats: Optional API usage stats for tracking
        correlation_id: Correlation ID for logging

    Returns:
        Final processed data
    """
    current_data = initial_data
    pipeline_start_time = time.time()

    log_with_context(
        logger,
        logging.INFO,
        f"Starting {len(stages)} stage pipeline",
        correlation_id=correlation_id
    )

    for stage_idx, (stage_name, stage_func) in enumerate(stages):
        stage_start_time = time.time()

        try:
            # Execute stage with resource management
            if asyncio.iscoroutinefunction(stage_func):
                current_data = await execute_with_resource_management(
                    f"pipeline_{stage_name}", stage_func, api_usage_stats, correlation_id, current_data
                )
            else:
                # Wrap synchronous function
                current_data = await execute_with_resource_management(
                    f"pipeline_{stage_name}",
                    lambda data: stage_func(data),
                    api_usage_stats,
                    correlation_id,
                    current_data
                )

            stage_duration = time.time() - stage_start_time
            log_with_context(
                logger,
                logging.DEBUG,
                f"Stage {stage_name} completed in {stage_duration:.1f}s",
                correlation_id=correlation_id
            )

        except Exception as e:
            log_with_context(
                logger,
                logging.ERROR,
                f"Pipeline stage {stage_name} failed: {e}",
                correlation_id=correlation_id
            )
            raise

    pipeline_duration = time.time() - pipeline_start_time
    log_with_context(
        logger,
        logging.INFO,
        f"Pipeline completed in {pipeline_duration:.1f}s",
        correlation_id=correlation_id
    )

    return current_data


async def parallel_map(items: List[Any], mapper_func: Callable,
                      max_concurrency: int = 10,
                      operation_type: str = "map",
                      api_usage_stats: Optional[ApiUsageStats] = None,
                      correlation_id: str = None) -> List[Any]:
    """Execute mapping function in parallel with concurrency control.

    Args:
        items: Items to map
        mapper_func: Mapping function (can be async or sync)
        max_concurrency: Maximum concurrent operations
        operation_type: Operation type for resource management
        api_usage_stats: Optional API usage stats for tracking
        correlation_id: Correlation ID for logging

    Returns:
        List of mapped results
    """
    if not items:
        return []

    # Create semaphore for this specific operation
    semaphore = asyncio.Semaphore(max_concurrency)

    async def map_with_semaphore(item):
        async with semaphore:
            return await execute_with_resource_management(
                operation_type, mapper_func, api_usage_stats, correlation_id, item
            )

    # Execute all mappings concurrently
    tasks = [map_with_semaphore(item) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Separate successful results from exceptions
    successful_results = []
    exceptions = []

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            exceptions.append((i, result))
            successful_results.append(None)
        else:
            successful_results.append(result)

    # Log exceptions if any
    if exceptions:
        log_with_context(
            logger,
            logging.WARNING,
            f"parallel_map had {len(exceptions)} failures out of {len(items)} items",
            correlation_id=correlation_id
        )

    return successful_results


async def throttled_execution(operations: List[Callable],
                            operations_per_second: float = 10.0,
                            api_usage_stats: Optional[ApiUsageStats] = None,
                            correlation_id: str = None) -> List[Any]:
    """Execute operations with rate limiting.

    Args:
        operations: List of operations to execute
        operations_per_second: Rate limit (operations per second)
        api_usage_stats: Optional API usage stats for tracking
        correlation_id: Correlation ID for logging

    Returns:
        List of operation results
    """
    if not operations:
        return []

    results = []
    interval = 1.0 / operations_per_second
    start_time = time.time()

    log_with_context(
        logger,
        logging.INFO,
        f"Starting throttled execution: {len(operations)} operations at {operations_per_second} ops/sec",
        correlation_id=correlation_id
    )

    for i, operation in enumerate(operations):
        operation_start = time.time()

        try:
            # Execute operation
            if asyncio.iscoroutinefunction(operation):
                result = await operation()
            else:
                result = await asyncio.to_thread(operation)

            results.append(result)

            # Track in usage stats if provided
            if api_usage_stats:
                duration = time.time() - operation_start
                api_usage_stats.add_call("throttled_operation", duration=duration)

        except Exception as e:
            log_with_context(
                logger,
                logging.ERROR,
                f"Throttled operation {i} failed: {e}",
                correlation_id=correlation_id
            )
            results.append(None)

            if api_usage_stats:
                duration = time.time() - operation_start
                api_usage_stats.add_call("throttled_operation", duration=duration, error=True)

        # Apply throttling delay
        if i < len(operations) - 1:  # Don't wait after the last operation
            elapsed = time.time() - operation_start
            delay = max(0, interval - elapsed)
            if delay > 0:
                await asyncio.sleep(delay)

    total_duration = time.time() - start_time
    actual_rate = len(operations) / total_duration

    log_with_context(
        logger,
        logging.INFO,
        f"Throttled execution completed: {len(operations)} operations in {total_duration:.1f}s "
        f"(actual rate: {actual_rate:.1f} ops/sec)",
        correlation_id=correlation_id
    )

    return results


def get_concurrency_manager() -> ConcurrencyManager:
    """Get the global concurrency manager instance."""
    return _concurrency_manager


def reset_concurrency_state():
    """Reset concurrency management state - useful for testing."""
    global _concurrency_manager
    _concurrency_manager = ConcurrencyManager()


def search_concepts_batch(concepts: List[str], max_results_per_concept: int = 3, **kwargs) -> Dict[str, List[Dict[str, Any]]]:
    """Search for concepts in batch mode.
    
    This function provides an async processing interface to concept search.
    
    Args:
        concepts: List of concepts to search for
        max_results_per_concept: Maximum results per concept
        **kwargs: Additional arguments
        
    Returns:
        Dictionary mapping concepts to search results
    """
    from .core_mining import search_concepts_batch as _search_batch_core
    from .client import get_client
    
    client = get_client()
    if not client:
        return {}
    
    return _search_batch_core(client, concepts, max_results_per_concept)
__all__ = [
    'ConcurrencyManager',
    'AsyncResultAggregator',
    'execute_with_resource_management',
    'process_with_streaming',
    'execute_parallel_pipeline',
    'parallel_map',
    'throttled_execution',
    'get_concurrency_manager',
    'reset_concurrency_state',
    'search_concepts_batch'
]