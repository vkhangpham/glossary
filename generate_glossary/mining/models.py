"""Data models and schemas for the academic glossary mining system.

This module contains all Pydantic models used throughout the mining system,
including concept definitions, web resources, queue status, API usage tracking,
performance profiles, and webhook configurations.
"""

import statistics
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field


class ConceptDefinition(BaseModel):
    """Schema for extracted academic concept definitions."""
    concept: str = Field(description="The academic term or concept")
    definition: str = Field(description="Clear, comprehensive definition")
    context: str = Field(description="Academic field or domain")
    key_points: List[str] = Field(default=[], description="Key characteristics")
    related_concepts: List[str] = Field(default=[], description="Related terms")
    source_quality: float = Field(default=0.0, ge=0.0, le=1.0, description="0..1 source quality score")


class WebResource(BaseModel):
    """Schema for web resources containing definitions."""
    url: str
    title: str = ""
    definitions: List[ConceptDefinition] = Field(default_factory=list)
    domain: str = ""

    def model_post_init(self, __context: Any) -> None:
        self.domain = urlparse(self.url).netloc


class QueueStatus(BaseModel):
    """Schema for Firecrawl v2.2.0 queue status response."""
    jobs_in_queue: int = Field(description="Number of jobs currently in queue")
    active_jobs: int = Field(description="Number of active jobs being processed")
    queued_jobs: int = Field(description="Number of jobs waiting to be processed")
    waiting_jobs: int = Field(description="Number of jobs waiting to be processed")
    completed_jobs: int = Field(default=0, description="Number of completed jobs")
    failed_jobs: int = Field(default=0, description="Number of failed jobs")
    max_concurrency: int = Field(description="Maximum concurrent jobs allowed")
    processing_time_avg: float = Field(default=0.0, description="Average processing time")
    queue_utilization: float = Field(default=0.0, description="Queue utilization ratio")
    estimated_wait_time: int = Field(default=0, description="Estimated wait time in seconds")
    most_recent_success: Optional[str] = Field(default=None, description="Timestamp of most recent successful job")


class ApiUsageStats(BaseModel):
    """Schema for API usage tracking with enhanced performance metrics."""
    search_calls: int = Field(default=0, description="Number of search API calls")
    scrape_calls: int = Field(default=0, description="Number of scrape API calls")
    extract_calls: int = Field(default=0, description="Number of extract API calls")
    map_calls: int = Field(default=0, description="Number of map API calls")
    batch_scrape_calls: int = Field(default=0, description="Number of batch scrape API calls")
    queue_status_calls: int = Field(default=0, description="Number of queue status API calls")
    total_calls: int = Field(default=0, description="Total API calls made")

    # Performance metrics
    call_durations: Dict[str, List[float]] = Field(default_factory=dict, description="API call duration tracking")
    error_counts: Dict[str, int] = Field(default_factory=dict, description="Error count by call type")
    cache_hits: int = Field(default=0, description="Number of cache hits")
    throttle_events: int = Field(default=0, description="Number of throttling events")
    concurrent_operations: int = Field(default=0, description="Peak concurrent operations")
    call_history: List[Dict[str, Any]] = Field(default_factory=list, description="Detailed call history")

    def add_call(self, call_type: str, duration: float = 0.0, success: bool = True, cached: bool = False, error: bool = False):
        """Add a call to the usage stats with performance metrics."""
        # Update specific call counter
        if hasattr(self, f"{call_type}_calls"):
            setattr(self, f"{call_type}_calls", getattr(self, f"{call_type}_calls") + 1)
        self.total_calls += 1

        # Add to call history
        call_record = {
            "type": call_type,
            "duration": duration,
            "success": success,
            "cached": cached,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        self.call_history.append(call_record)

        if duration > 0:
            if call_type not in self.call_durations:
                self.call_durations[call_type] = []
            self.call_durations[call_type].append(duration)

        if cached:
            self.cache_hits += 1

        if error:
            self.error_counts[call_type] = self.error_counts.get(call_type, 0) + 1

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics = {}
        for call_type, durations in self.call_durations.items():
            if durations:
                metrics[call_type] = {
                    'avg_duration': statistics.mean(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'p95_duration': statistics.quantiles(durations, n=20)[18] if len(durations) >= 5 else max(durations),
                    'call_count': len(durations)
                }

        # Calculate success rate
        successful_calls = sum(1 for call in self.call_history if call.get("success", True))
        success_rate = successful_calls / max(1, self.total_calls)

        # Calculate average duration across all calls
        all_durations = [call["duration"] for call in self.call_history if call["duration"] > 0]
        average_duration = statistics.mean(all_durations) if all_durations else 0.0

        # Calculate calls per method
        calls_per_method = {}
        for call in self.call_history:
            call_type = call["type"]
            calls_per_method[call_type] = calls_per_method.get(call_type, 0) + 1

        return {
            'per_endpoint': metrics,
            'overall': {
                'total_calls': self.total_calls,
                'success_rate': success_rate,
                'average_duration': average_duration,
                'calls_per_method': calls_per_method,
                'cache_hit_rate': self.cache_hits / max(1, self.total_calls),
                'error_rate': sum(self.error_counts.values()) / max(1, self.total_calls),
                'peak_concurrency': self.concurrent_operations,
                'throttle_events': self.throttle_events
            }
        }


class WebhookConfig(BaseModel):
    """Schema for webhook configuration."""
    url: str = Field(description="Webhook endpoint URL")
    events: List[str] = Field(default=["started", "page", "completed", "failed"], description="Events to subscribe to")
    secret: Optional[str] = Field(default=None, description="Secret for webhook signature verification")
    verify_signature: bool = Field(default=True, description="Whether to verify webhook signatures")
    retry_on_failure: bool = Field(default=True, description="Whether to retry failed webhook calls")


class PerformanceProfile(BaseModel):
    """Performance tuning profile for different use cases."""
    name: str = Field(description="Profile name")
    max_concurrent: int = Field(description="Maximum concurrent operations")
    polling_strategy: str = Field(default="adaptive", description="Polling strategy: adaptive, exponential, linear")
    queue_threshold: float = Field(default=0.8, description="Queue load threshold for throttling")
    cache_priority: str = Field(default="balanced", description="Cache priority: speed, accuracy, balanced")
    timeout_multiplier: float = Field(default=1.0, description="Timeout adjustment multiplier")
    retry_strategy: str = Field(default="intelligent", description="Retry strategy: aggressive, conservative, intelligent")

    @classmethod
    def speed_optimized(cls) -> 'PerformanceProfile':
        return cls(
            name="speed_optimized",
            max_concurrent=20,
            polling_strategy="aggressive",
            queue_threshold=0.9,
            cache_priority="speed",
            timeout_multiplier=0.8,
            retry_strategy="aggressive"
        )

    @classmethod
    def accuracy_focused(cls) -> 'PerformanceProfile':
        return cls(
            name="accuracy_focused",
            max_concurrent=8,
            polling_strategy="conservative",
            queue_threshold=0.6,
            cache_priority="accuracy",
            timeout_multiplier=1.5,
            retry_strategy="conservative"
        )

    @classmethod
    def balanced(cls) -> 'PerformanceProfile':
        return cls(
            name="balanced",
            max_concurrent=12,
            polling_strategy="adaptive",
            queue_threshold=0.8,
            cache_priority="balanced",
            timeout_multiplier=1.0,
            retry_strategy="intelligent"
        )


class QueuePredictor(BaseModel):
    """Predictive queue management and load forecasting."""
    queue_history: deque = Field(default_factory=lambda: deque(maxlen=100), description="Recent queue status history")
    completion_times: deque = Field(default_factory=lambda: deque(maxlen=50), description="Job completion time history")
    load_patterns: Dict[str, List[float]] = Field(default_factory=dict, description="Historical load patterns by hour")

    def add_queue_status(self, status: QueueStatus):
        """Add queue status to prediction history."""
        timestamp = datetime.now()
        self.queue_history.append({
            'timestamp': timestamp,
            'active_jobs': status.active_jobs,
            'waiting_jobs': status.waiting_jobs,
            'total_jobs': status.jobs_in_queue
        })

        # Track hourly load patterns
        hour_key = timestamp.strftime('%H')
        if hour_key not in self.load_patterns:
            self.load_patterns[hour_key] = []
        self.load_patterns[hour_key].append(status.jobs_in_queue)

        # Keep only recent patterns
        if len(self.load_patterns[hour_key]) > 30:
            self.load_patterns[hour_key] = self.load_patterns[hour_key][-30:]

    def predict_completion_time(self, job_size: int = 1) -> float:
        """Predict job completion time based on historical data."""
        if not self.completion_times:
            return 60.0  # Default 1 minute

        recent_times = list(self.completion_times)[-10:]
        avg_time = statistics.mean(recent_times)

        # Adjust for job size
        size_factor = min(job_size / 10, 3.0)  # Cap at 3x for large jobs
        return avg_time * size_factor

    def get_optimal_delay(self) -> float:
        """Calculate optimal delay before submitting new jobs."""
        if len(self.queue_history) < 2:
            return 0.0

        # Calculate queue load trend
        recent_entries = list(self.queue_history)[-5:]
        if not recent_entries:
            return 0.0

        avg_jobs = sum(entry['total_jobs'] for entry in recent_entries) / len(recent_entries)

        # If queue is growing, increase delay
        if avg_jobs > 50:
            return min(avg_jobs * 0.1, 30.0)  # Cap at 30 seconds

        return 0.0

    def get_load_forecast(self, hours_ahead: int = 1) -> float:
        """Forecast queue load for future hours."""
        current_hour = datetime.now().hour
        target_hour = (current_hour + hours_ahead) % 24
        hour_key = f"{target_hour:02d}"

        if hour_key in self.load_patterns and self.load_patterns[hour_key]:
            return statistics.mean(self.load_patterns[hour_key][-10:])

        # Fallback to overall average
        all_loads = []
        for patterns in self.load_patterns.values():
            all_loads.extend(patterns[-5:])

        return statistics.mean(all_loads) if all_loads else 20.0

    class Config:
        arbitrary_types_allowed = True


__all__ = [
    'ConceptDefinition',
    'WebResource',
    'QueueStatus',
    'ApiUsageStats',
    'WebhookConfig',
    'PerformanceProfile',
    'QueuePredictor'
]