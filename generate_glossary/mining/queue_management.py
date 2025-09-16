"""Queue management and monitoring module.

This module provides comprehensive queue status monitoring, predictive queue management,
adaptive polling strategies, and intelligent throttling for optimal resource utilization
in the Firecrawl-based mining system.
"""

import asyncio
import time
import logging
import statistics
from typing import Dict, Any, List, Optional
from firecrawl import FirecrawlApp

from .models import QueueStatus, QueuePredictor, PerformanceProfile, ApiUsageStats
from .client import get_client
from generate_glossary.utils.error_handler import (
    ExternalServiceError, handle_error, processing_context
)
from generate_glossary.utils.logger import get_logger, log_processing_step, log_with_context


logger = get_logger(__name__)

# Global queue management state
_queue_predictor = QueuePredictor()


def get_queue_status(app: Optional[FirecrawlApp] = None) -> Optional[QueueStatus]:
    """Enhanced queue status monitoring with predictive queue management.

    Monitors job queues providing visibility into:
    - Jobs in queue count with trend analysis
    - Active jobs being processed with performance metrics
    - Waiting jobs with prediction capabilities
    - Max concurrency limits with adaptive recommendations
    - Most recent successful job timestamp with completion rate analysis
    - Real-time queue health monitoring and alerting

    Args:
        app: Initialized FirecrawlApp client, will get default if None

    Returns:
        QueueStatus object with enhanced metrics or None if failed
    """
    if app is None:
        app = get_client()
        if app is None:
            return None

    with processing_context("get_queue_status") as correlation_id:
        start_time = time.time()
        log_processing_step(
            logger,
            "get_queue_status",
            "started",
            {"endpoint": "/v2/team/queue-status", "predictive_monitoring": True},
            correlation_id=correlation_id
        )

        try:
            # Track API usage with timing
            duration = 0.0

            try:
                # Attempt to use new v2.2.0 queue status method
                status_response = app.get_queue_status()
                duration = time.time() - start_time
            except (AttributeError, TypeError):
                # Fallback for SDK versions that don't support queue status yet
                duration = time.time() - start_time
                log_with_context(
                    logger,
                    logging.WARNING,
                    "Queue status endpoint not available in current SDK version",
                    correlation_id=correlation_id
                )
                return None

            # Parse response into QueueStatus model with enhanced validation
            if isinstance(status_response, dict):
                # Extract and compute queue status fields
                jobs_in_queue = status_response.get("jobs_in_queue", 0)
                active_jobs = status_response.get("active_jobs", 0)
                waiting_jobs = status_response.get("waiting_jobs", jobs_in_queue)
                max_concurrency = status_response.get("max_concurrency", 10)

                # Compute derived fields
                queue_utilization = waiting_jobs / max(1, max_concurrency)
                estimated_wait_time = int(waiting_jobs * 30)  # Rough estimate: 30 seconds per job

                queue_status = QueueStatus(
                    jobs_in_queue=jobs_in_queue,
                    active_jobs=active_jobs,
                    waiting_jobs=waiting_jobs,
                    queued_jobs=waiting_jobs,  # Alias for compatibility
                    completed_jobs=status_response.get("completed_jobs", 0),
                    failed_jobs=status_response.get("failed_jobs", 0),
                    max_concurrency=max_concurrency,
                    processing_time_avg=status_response.get("processing_time_avg", 0.0),
                    queue_utilization=status_response.get("queue_utilization", queue_utilization),
                    estimated_wait_time=status_response.get("estimated_wait_time", estimated_wait_time),
                    most_recent_success=status_response.get("most_recent_success")
                )

                # Update predictor with current status
                _queue_predictor.add_queue_status(queue_status)

                # Generate predictive insights
                queue_insights = generate_queue_insights(queue_status)

                # Check for queue health alerts
                health_alerts = check_queue_health(queue_status)

                log_processing_step(
                    logger,
                    "get_queue_status",
                    "completed",
                    {
                        "jobs_in_queue": queue_status.jobs_in_queue,
                        "active_jobs": queue_status.active_jobs,
                        "waiting_jobs": queue_status.waiting_jobs,
                        "max_concurrency": queue_status.max_concurrency,
                        "queue_utilization": queue_status.waiting_jobs / max(1, queue_status.max_concurrency),
                        "predicted_completion_time": queue_insights.get("predicted_completion_time", 0),
                        "recommended_concurrency": queue_insights.get("recommended_concurrency", queue_status.max_concurrency),
                        "health_score": queue_insights.get("health_score", 100),
                        "alerts": len(health_alerts)
                    },
                    correlation_id=correlation_id
                )

                # Log health alerts if any
                for alert in health_alerts:
                    log_with_context(
                        logger,
                        logging.WARNING,
                        f"Queue health alert: {alert}",
                        correlation_id=correlation_id
                    )

                return queue_status
            else:
                log_with_context(
                    logger,
                    logging.WARNING,
                    f"Unexpected queue status response format: {type(status_response)}",
                    correlation_id=correlation_id
                )
                return None

        except Exception as e:
            handle_error(
                ExternalServiceError(f"Enhanced queue status request failed: {e}", service="firecrawl"),
                context={"predictive_monitoring": True},
                operation="get_queue_status"
            )
            log_with_context(
                logger,
                logging.ERROR,
                f"Enhanced queue status request failed: {e}",
                correlation_id=correlation_id
            )
            return None


async def get_queue_status_async(app: Optional[FirecrawlApp] = None) -> Optional[QueueStatus]:
    """Async wrapper for get_queue_status that runs in a thread to avoid blocking.

    Args:
        app: Initialized FirecrawlApp client

    Returns:
        QueueStatus object with enhanced metrics or None if failed
    """
    if app is None:
        app = get_client()
        if app is None:
            return None

    return await asyncio.to_thread(get_queue_status, app)


def generate_queue_insights(queue_status: QueueStatus) -> Dict[str, Any]:
    """Generate predictive insights from current queue status.

    Args:
        queue_status: Current queue status

    Returns:
        Dictionary with predictive insights and recommendations
    """
    insights = {}

    try:
        # Predict completion time for current queue
        avg_job_size = max(queue_status.active_jobs + queue_status.waiting_jobs, 1)
        predicted_time = _queue_predictor.predict_completion_time(avg_job_size)
        insights["predicted_completion_time"] = predicted_time

        # Recommend optimal concurrency
        recommended_concurrency = _recommend_optimal_concurrency(queue_status.max_concurrency)
        insights["recommended_concurrency"] = recommended_concurrency

        # Calculate queue health score (0-100)
        utilization = queue_status.waiting_jobs / max(1, queue_status.max_concurrency)
        if utilization <= 0.5:
            health_score = 100  # Excellent
        elif utilization <= 0.8:
            health_score = 80   # Good
        elif utilization <= 1.0:
            health_score = 60   # Fair
        else:
            health_score = max(20, 60 - (utilization - 1.0) * 40)  # Poor to Critical

        insights["health_score"] = int(health_score)

        # Queue trend analysis
        if len(_queue_predictor.queue_history) >= 3:
            recent_loads = [s['total_jobs'] for s in list(_queue_predictor.queue_history)[-3:]]
            trend = ("increasing" if recent_loads[-1] > recent_loads[0] else
                    "decreasing" if recent_loads[-1] < recent_loads[0] else "stable")
            insights["load_trend"] = trend
        else:
            insights["load_trend"] = "unknown"

        # Optimal delay recommendation
        optimal_delay = _queue_predictor.get_optimal_delay()
        insights["recommended_delay"] = optimal_delay

    except Exception as e:
        logger.debug(f"Failed to generate queue insights: {e}")
        insights = {"error": str(e)}

    return insights


def check_queue_health(queue_status: QueueStatus) -> List[str]:
    """Check queue health and return list of alerts.

    Args:
        queue_status: Current queue status

    Returns:
        List of health alert messages
    """
    alerts = []

    try:
        # Critical overload alert
        utilization = queue_status.waiting_jobs / max(1, queue_status.max_concurrency)
        if utilization > 2.0:
            alerts.append(f"CRITICAL: Queue severely overloaded ({utilization:.1f}x capacity)")
        elif utilization > 1.5:
            alerts.append(f"HIGH: Queue significantly overloaded ({utilization:.1f}x capacity)")
        elif utilization > 1.0:
            alerts.append(f"MEDIUM: Queue over capacity ({utilization:.1f}x capacity)")

        # Stagnation detection
        if len(_queue_predictor.queue_history) >= 5:
            recent_statuses = list(_queue_predictor.queue_history)[-5:]
            if (all(s['active_jobs'] == 0 for s in recent_statuses) and
                any(s['waiting_jobs'] > 0 for s in recent_statuses)):
                alerts.append("CRITICAL: Queue appears stagnant - no active jobs despite waiting jobs")

        # Performance degradation detection
        if len(_queue_predictor.completion_times) >= 10:
            recent_times = list(_queue_predictor.completion_times)[-5:]
            older_times = list(_queue_predictor.completion_times)[-10:-5]
            if len(older_times) >= 3 and len(recent_times) >= 3:
                avg_recent = statistics.mean(recent_times)
                avg_older = statistics.mean(older_times)
                if avg_recent > avg_older * 1.5:
                    degradation_pct = ((avg_recent / avg_older - 1) * 100)
                    alerts.append(f"MEDIUM: Performance degradation detected "
                                f"(completion time increased by {degradation_pct:.0f}%)")

        # Concurrency recommendations
        recommended_concurrency = _recommend_optimal_concurrency(queue_status.max_concurrency)
        if recommended_concurrency != queue_status.max_concurrency:
            direction = "increase" if recommended_concurrency > queue_status.max_concurrency else "decrease"
            alerts.append(f"INFO: Consider {direction} concurrency to {recommended_concurrency} "
                         "for better performance")

    except Exception as e:
        logger.debug(f"Failed to check queue health: {e}")
        alerts.append(f"WARNING: Health check failed: {e}")

    return alerts


async def poll_job_with_adaptive_strategy(app: FirecrawlApp, job_id: str,
                                        enable_queue_monitoring: bool = False) -> Dict[str, Any]:
    """Enhanced adaptive polling with queue-aware strategy and dynamic timeout adjustment.

    Implements:
    - Exponential backoff with queue-aware adjustments
    - Dynamic timeout based on job size and complexity
    - Predictive completion time estimation
    - Circuit breaker pattern for failed polling
    - Real-time queue status integration

    Args:
        app: Firecrawl client
        job_id: Job ID to poll
        enable_queue_monitoring: Whether to use queue status for adaptive polling

    Returns:
        Job result dictionary
    """
    with processing_context("poll_job_adaptive") as correlation_id:
        try:
            start_time = time.time()
            base_interval = 1.0
            max_interval = 30.0
            current_interval = base_interval
            max_attempts = 60

            # Adaptive parameters based on queue status
            max_wait_seconds = 300.0  # Default 5 minutes wall-clock timeout
            if enable_queue_monitoring:
                queue_status = await get_queue_status_async(app)
                if queue_status:
                    _queue_predictor.add_queue_status(queue_status)

                    # Adjust polling based on queue load
                    if queue_status.waiting_jobs > queue_status.max_concurrency:
                        # High load: slower polling, longer timeouts
                        base_interval = 2.0
                        max_interval = 60.0
                        max_wait_seconds = 600.0  # 10 minutes for high load
                    elif queue_status.waiting_jobs < queue_status.max_concurrency * 0.3:
                        # Low load: faster polling
                        base_interval = 0.5
                        max_interval = 15.0
                        max_wait_seconds = 180.0  # 3 minutes for low load

                    # Get predicted completion time and adjust timeout accordingly
                    predicted_time = _queue_predictor.predict_completion_time()
                    # Set timeout based on predicted time + buffer, but with safe bounds
                    predicted_timeout = max(predicted_time * 1.5, 120.0)  # 50% buffer, minimum 2 minutes
                    max_wait_seconds = min(predicted_timeout, 900.0)  # Maximum 15 minutes

                    # Keep max_attempts as a safety fallback
                    max_attempts = min(int(max_wait_seconds / base_interval), 300)

            # Circuit breaker state
            consecutive_failures = 0
            max_failures = 5
            last_error = None

            attempt = 0
            while attempt < max_attempts:
                # Check wall-clock timeout
                elapsed_time = time.time() - start_time
                if elapsed_time >= max_wait_seconds:
                    log_with_context(
                        logger,
                        logging.WARNING,
                        f"Polling timeout after {elapsed_time:.1f}s (limit: {max_wait_seconds:.1f}s)",
                        correlation_id=correlation_id
                    )
                    break

                try:
                    # Apply circuit breaker
                    if consecutive_failures >= max_failures:
                        log_with_context(
                            logger,
                            logging.WARNING,
                            f"Circuit breaker: too many polling failures ({consecutive_failures}), backing off",
                            correlation_id=correlation_id
                        )
                        await asyncio.sleep(min(consecutive_failures * 2, 30))
                        consecutive_failures = 0

                    # Get job status with timing
                    poll_start = time.time()
                    try:
                        status = app.get_crawl_status(job_id)
                    except AttributeError:
                        # Fallback method for different SDK versions
                        status = getattr(app, 'check_crawl_status', lambda x: {"status": "unknown"})(job_id)
                    poll_duration = time.time() - poll_start

                    # Handle response format
                    current_status = "unknown"
                    if isinstance(status, dict):
                        if status.get("success") and "data" in status:
                            status_data = status["data"]
                            current_status = status_data.get("status", "unknown")
                        else:
                            current_status = status.get("status", "unknown")
                    else:
                        current_status = getattr(status, "status", "unknown")

                    log_with_context(
                        logger,
                        logging.DEBUG,
                        f"Polling attempt {attempt + 1}/{max_attempts}: status={current_status}, "
                        f"interval={current_interval:.1f}s",
                        correlation_id=correlation_id
                    )

                    # Terminal states
                    if current_status in ("completed", "success", "failed", "error"):
                        completion_time = time.time() - start_time
                        _queue_predictor.completion_times.append(completion_time)

                        log_with_context(
                            logger,
                            logging.INFO,
                            f"Job {job_id} completed with status '{current_status}' after {completion_time:.1f}s",
                            correlation_id=correlation_id
                        )

                        return status

                    # Reset consecutive failures on successful poll
                    consecutive_failures = 0
                    last_error = None

                except Exception as e:
                    consecutive_failures += 1
                    last_error = e
                    log_with_context(
                        logger,
                        logging.WARNING,
                        f"Polling attempt {attempt + 1} failed: {e}",
                        correlation_id=correlation_id
                    )

                # Adaptive wait with exponential backoff
                await asyncio.sleep(current_interval)
                current_interval = min(current_interval * 1.5, max_interval)
                attempt += 1

            # Timeout or max attempts reached
            final_error = last_error or f"Polling timed out after {max_attempts} attempts"
            log_with_context(
                logger,
                logging.ERROR,
                f"Adaptive polling failed for job {job_id}: {final_error}",
                correlation_id=correlation_id
            )

            return {"status": "timeout", "error": str(final_error)}

        except Exception as e:
            log_with_context(
                logger,
                logging.ERROR,
                f"Adaptive polling failed for job {job_id}: {e}",
                correlation_id=correlation_id
            )
            raise


async def apply_intelligent_throttling(app: Optional[FirecrawlApp] = None) -> float:
    """Apply intelligent throttling based on predictive queue management and load forecasting.

    Args:
        app: Firecrawl client, will get default if None

    Returns:
        Optimal delay in seconds before proceeding with batch operations
    """
    if app is None:
        app = get_client()
        if app is None:
            return 0.0

    with processing_context("intelligent_throttling") as correlation_id:
        try:
            current_queue_status = await get_queue_status_async(app)
            if not current_queue_status:
                return 0.0

            # Update predictor with current status
            _queue_predictor.add_queue_status(current_queue_status)

            # Calculate load metrics
            queue_utilization = current_queue_status.waiting_jobs / max(1, current_queue_status.max_concurrency)
            total_load = current_queue_status.active_jobs + current_queue_status.waiting_jobs

            # Get predictive delay recommendation
            predicted_delay = _queue_predictor.get_optimal_delay()

            # Base throttling logic
            throttle_delay = 0.0

            # Get current performance profile to avoid circular dependency
            from .performance import get_current_profile
            current_profile = get_current_profile()

            if queue_utilization > current_profile.queue_threshold:
                # Queue approaching or exceeding capacity
                base_delay = (queue_utilization - current_profile.queue_threshold) * 10
                throttle_delay = min(base_delay, 10.0)  # Max 10 second delay

                log_with_context(
                    logger,
                    logging.INFO,
                    f"Queue utilization high ({queue_utilization:.2f}), applying base throttling: {throttle_delay:.1f}s",
                    correlation_id=correlation_id
                )

            # Add predictive component
            if predicted_delay > 0:
                throttle_delay = max(throttle_delay, predicted_delay)
                log_with_context(
                    logger,
                    logging.INFO,
                    f"Predictive analysis suggests additional delay: {predicted_delay:.1f}s "
                    f"(total: {throttle_delay:.1f}s)",
                    correlation_id=correlation_id
                )

            # Load balancing across time windows
            if total_load > 100:  # Very high total load
                time_based_delay = min((total_load - 100) * 0.05, 5.0)
                throttle_delay = max(throttle_delay, time_based_delay)
                log_with_context(
                    logger,
                    logging.INFO,
                    f"High total load ({total_load}), additional time-based delay: {time_based_delay:.1f}s",
                    correlation_id=correlation_id
                )

            return throttle_delay

        except Exception as e:
            log_with_context(
                logger,
                logging.ERROR,
                f"Intelligent throttling failed: {e}",
                correlation_id=correlation_id
            )
            return 0.0


def _recommend_optimal_concurrency(current_max: int) -> int:
    """Recommend optimal concurrency based on historical performance."""
    if not _queue_predictor.completion_times:
        return current_max

    # Get recent completion times for analysis
    recent_times = list(_queue_predictor.completion_times)[-10:]
    if len(recent_times) < 3:
        return current_max

    avg_time = statistics.mean(recent_times)

    # If jobs are completing quickly, we can increase concurrency
    if avg_time < 30:  # Fast jobs (< 30 seconds)
        return min(current_max + 2, 25)  # Cap at 25
    elif avg_time > 120:  # Slow jobs (> 2 minutes)
        return max(current_max - 2, 5)   # Minimum 5
    else:
        return current_max  # Good balance


def get_queue_predictor() -> QueuePredictor:
    """Get the global queue predictor instance."""
    return _queue_predictor


def set_performance_profile(profile: PerformanceProfile) -> None:
    """Set the performance profile for queue management via performance module."""
    # Set the profile in the performance module to maintain sync
    import sys
    sys.modules['generate_glossary.mining.performance']._performance_profile = profile


def reset_queue_state() -> None:
    """Reset queue management state - useful for testing."""
    global _queue_predictor
    _queue_predictor = QueuePredictor()
    # Reset performance profile via performance module to maintain sync
    from .performance import reset_performance_state
    reset_performance_state()


__all__ = [
    'get_queue_status',
    'get_queue_status_async',
    'generate_queue_insights',
    'check_queue_health',
    'poll_job_with_adaptive_strategy',
    'apply_intelligent_throttling',
    'get_queue_predictor',
    'set_performance_profile',
    'reset_queue_state'
]