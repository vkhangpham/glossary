"""Performance management and optimization module.

This module handles performance profiling, auto-tuning, optimization recommendations,
and system health monitoring for the mining system. It provides intelligent
performance adjustment based on usage patterns and system metrics.
"""

import time
import logging
import statistics
from typing import Dict, Any, List, Optional
from datetime import datetime

from .models import PerformanceProfile, ApiUsageStats, QueuePredictor
from generate_glossary.utils.error_handler import processing_context
from generate_glossary.utils.logger import get_logger, log_processing_step, log_with_context


logger = get_logger(__name__)

# Global performance state
_performance_profile = PerformanceProfile.balanced()
_performance_history: List[Dict[str, Any]] = []
_tuning_recommendations: List[Dict[str, Any]] = []


def configure_performance_profile(profile_name: str = None, custom_settings: Dict[str, Any] = None) -> Dict[str, Any]:
    """Configure performance profile for optimal v2.2.0 performance.

    Performance profiles for different use cases:
    - speed_optimized: Maximum performance with higher resource usage
    - accuracy_focused: Conservative approach prioritizing reliability
    - balanced: Optimal balance of speed and accuracy (default)
    - custom: User-defined settings

    Args:
        profile_name: Predefined profile name or 'custom'
        custom_settings: Dictionary of custom performance settings

    Returns:
        Dictionary containing applied configuration and performance impact
    """
    global _performance_profile

    with processing_context("configure_performance_profile") as correlation_id:
        log_processing_step(
            logger,
            "configure_performance_profile",
            "started",
            {"profile_name": profile_name, "has_custom_settings": bool(custom_settings)},
            correlation_id=correlation_id
        )

        try:
            old_profile_name = _performance_profile.name

            # Apply predefined profile
            if profile_name == "speed_optimized":
                _performance_profile = PerformanceProfile.speed_optimized()
            elif profile_name == "accuracy_focused":
                _performance_profile = PerformanceProfile.accuracy_focused()
            elif profile_name == "balanced" or profile_name is None:
                _performance_profile = PerformanceProfile.balanced()
            elif profile_name == "custom" and custom_settings:
                _performance_profile = PerformanceProfile(**custom_settings)
            else:
                raise ValueError(f"Invalid profile_name '{profile_name}' or missing custom_settings")

            # Calculate expected performance impact
            performance_impact = _calculate_profile_impact(old_profile_name, _performance_profile.name)

            config_result = {
                "previous_profile": old_profile_name,
                "new_profile": _performance_profile.name,
                "applied_settings": _performance_profile.model_dump(),
                "expected_impact": performance_impact,
                "timestamp": datetime.now().isoformat()
            }

            log_processing_step(
                logger,
                "configure_performance_profile",
                "completed",
                config_result,
                correlation_id=correlation_id
            )

            return config_result

        except Exception as e:
            log_with_context(
                logger,
                logging.ERROR,
                f"Failed to configure performance profile: {e}",
                correlation_id=correlation_id
            )
            raise


def auto_tune_performance(api_usage_stats: ApiUsageStats, queue_predictor: QueuePredictor) -> Dict[str, Any]:
    """Automatically tune performance settings based on historical data and system analysis.

    Analyzes recent performance metrics to:
    - Optimize concurrency limits based on actual usage patterns
    - Adjust polling strategies based on queue behavior
    - Recommend caching improvements
    - Suggest profile changes for better performance

    Args:
        api_usage_stats: API usage statistics for analysis
        queue_predictor: Queue predictor with historical data

    Returns:
        Dictionary containing tuning results and applied optimizations
    """
    with processing_context("auto_tune_performance") as correlation_id:
        start_time = time.time()
        log_processing_step(
            logger,
            "auto_tune_performance",
            "started",
            {"historical_data_available": bool(api_usage_stats.call_durations)},
            correlation_id=correlation_id
        )

        try:
            tuning_results = {
                "tuning_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "analysis_duration_ms": 0,
                "optimizations_applied": [],
                "recommendations": [],
                "performance_improvements": {}
            }

            # Analyze current performance metrics
            performance_metrics = api_usage_stats.get_performance_metrics()

            # Auto-tune concurrency limits based on performance data
            concurrency_optimizations = _auto_tune_concurrency(performance_metrics)
            tuning_results["optimizations_applied"].extend(concurrency_optimizations)

            # Auto-tune polling strategy based on queue history
            polling_optimizations = _auto_tune_polling_strategy(queue_predictor)
            tuning_results["optimizations_applied"].extend(polling_optimizations)

            # Generate cache optimization recommendations
            cache_recommendations = _analyze_cache_performance(performance_metrics)
            tuning_results["recommendations"].extend(cache_recommendations)

            # Estimate performance improvements
            tuning_results["performance_improvements"] = _estimate_tuning_impact(
                concurrency_optimizations, polling_optimizations
            )

            # Calculate analysis duration
            tuning_results["analysis_duration_ms"] = int((time.time() - start_time) * 1000)

            # Store tuning results in history
            _performance_history.append(tuning_results.copy())
            if len(_performance_history) > 50:
                _performance_history.pop(0)

            log_processing_step(
                logger,
                "auto_tune_performance",
                "completed",
                {
                    "optimizations_count": len(tuning_results["optimizations_applied"]),
                    "recommendations_count": len(tuning_results["recommendations"]),
                    "analysis_duration_ms": tuning_results["analysis_duration_ms"]
                },
                correlation_id=correlation_id
            )

            return tuning_results

        except Exception as e:
            log_with_context(
                logger,
                logging.ERROR,
                f"Auto-tuning failed: {e}",
                correlation_id=correlation_id
            )
            raise


def get_performance_status(api_usage_stats: ApiUsageStats, queue_predictor: QueuePredictor) -> Dict[str, Any]:
    """Get comprehensive performance status including current configuration and metrics.

    Returns detailed status of:
    - Current performance profile and settings
    - Real-time performance metrics
    - Queue status and predictions
    - Optimization opportunities
    - System health assessment

    Args:
        api_usage_stats: Current API usage statistics
        queue_predictor: Queue predictor with historical data

    Returns:
        Dictionary containing complete performance status
    """
    with processing_context("get_performance_status") as correlation_id:
        log_processing_step(
            logger,
            "get_performance_status",
            "started",
            {"current_profile": _performance_profile.name},
            correlation_id=correlation_id
        )

        try:
            # Current configuration
            current_config = {
                "performance_profile": _performance_profile.model_dump(),
                "profile_age_minutes": _get_profile_age_minutes(),
                "tuning_history_count": len(_performance_history)
            }

            # Real-time metrics
            performance_metrics = api_usage_stats.get_performance_metrics()

            # Queue status and predictions
            queue_insights = _generate_queue_insights(queue_predictor)

            # System health assessment
            health_score = _calculate_system_health_score(performance_metrics, queue_insights)

            # Optimization opportunities
            optimization_opportunities = _identify_optimization_opportunities(
                performance_metrics, queue_insights
            )

            status = {
                "timestamp": datetime.now().isoformat(),
                "configuration": current_config,
                "performance_metrics": performance_metrics,
                "queue_insights": queue_insights,
                "system_health": {
                    "score": health_score,
                    "status": _get_health_status_text(health_score),
                    "alerts": _get_health_alerts(performance_metrics, queue_insights)
                },
                "optimization_opportunities": optimization_opportunities,
                "recent_tuning": _performance_history[-1] if _performance_history else None
            }

            log_processing_step(
                logger,
                "get_performance_status",
                "completed",
                {
                    "health_score": health_score,
                    "optimization_opportunities": len(optimization_opportunities)
                },
                correlation_id=correlation_id
            )

            return status

        except Exception as e:
            log_with_context(
                logger,
                logging.ERROR,
                f"Failed to get performance status: {e}",
                correlation_id=correlation_id
            )
            raise


def _calculate_profile_impact(old_profile: str, new_profile: str) -> Dict[str, Any]:
    """Calculate expected impact of profile change."""
    profile_impacts = {
        "speed_optimized": {
            "throughput_change": "+40%",
            "resource_usage": "+60%",
            "accuracy_risk": "moderate",
            "recommended_for": "large-scale batch operations"
        },
        "accuracy_focused": {
            "throughput_change": "-20%",
            "resource_usage": "-30%",
            "accuracy_risk": "minimal",
            "recommended_for": "high-quality extraction tasks"
        },
        "balanced": {
            "throughput_change": "baseline",
            "resource_usage": "baseline",
            "accuracy_risk": "low",
            "recommended_for": "general-purpose usage"
        }
    }

    return {
        "from": profile_impacts.get(old_profile, {}),
        "to": profile_impacts.get(new_profile, {}),
        "change_magnitude": _calculate_change_magnitude(old_profile, new_profile)
    }


def _calculate_change_magnitude(old_profile: str, new_profile: str) -> str:
    """Calculate magnitude of profile change."""
    profile_scores = {"accuracy_focused": 1, "balanced": 2, "speed_optimized": 3}
    old_score = profile_scores.get(old_profile, 2)
    new_score = profile_scores.get(new_profile, 2)

    diff = abs(new_score - old_score)
    if diff == 0:
        return "none"
    elif diff == 1:
        return "moderate"
    else:
        return "significant"


def _auto_tune_concurrency(performance_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Auto-tune concurrency limits based on performance data."""
    optimizations = []

    overall_metrics = performance_metrics.get('overall', {})
    error_rate = overall_metrics.get('error_rate', 0)

    # If error rate is high, reduce concurrency
    if error_rate > 0.1:  # 10% error rate
        global _performance_profile
        new_concurrent = max(5, int(_performance_profile.max_concurrent * 0.8))
        if new_concurrent != _performance_profile.max_concurrent:
            _performance_profile.max_concurrent = new_concurrent
            optimizations.append({
                "type": "concurrency_reduction",
                "reason": f"High error rate ({error_rate:.1%})",
                "old_value": _performance_profile.max_concurrent,
                "new_value": new_concurrent,
                "expected_improvement": "reduced error rate"
            })

    return optimizations


def _auto_tune_polling_strategy(queue_predictor: QueuePredictor) -> List[Dict[str, Any]]:
    """Auto-tune polling strategy based on queue behavior."""
    optimizations = []

    if len(queue_predictor.queue_history) > 10:
        recent_entries = list(queue_predictor.queue_history)[-10:]
        avg_queue_size = sum(entry.get('total_jobs', 0) for entry in recent_entries) / len(recent_entries)

        global _performance_profile
        if avg_queue_size > 100 and _performance_profile.polling_strategy != "conservative":
            _performance_profile.polling_strategy = "conservative"
            optimizations.append({
                "type": "polling_strategy",
                "reason": f"High average queue size ({avg_queue_size:.1f})",
                "new_strategy": "conservative",
                "expected_improvement": "reduced queue pressure"
            })

    return optimizations


def _analyze_cache_performance(performance_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Analyze cache performance and generate recommendations."""
    recommendations = []

    overall_metrics = performance_metrics.get('overall', {})
    cache_hit_rate = overall_metrics.get('cache_hit_rate', 0)

    if cache_hit_rate < 0.3:  # Less than 30% cache hit rate
        recommendations.append({
            "type": "cache_optimization",
            "priority": "high",
            "recommendation": "Increase cache TTL or enable more aggressive caching",
            "current_hit_rate": cache_hit_rate,
            "target_hit_rate": 0.5,
            "potential_speedup": "2-3x faster repeated operations"
        })

    return recommendations


def _estimate_tuning_impact(concurrency_opts: List, polling_opts: List) -> Dict[str, Any]:
    """Estimate performance impact of applied optimizations."""
    impact = {
        "throughput_change_percent": 0,
        "error_rate_improvement": 0,
        "resource_efficiency": 0
    }

    # Estimate impact from optimizations
    for opt in concurrency_opts:
        if opt["type"] == "concurrency_reduction":
            impact["error_rate_improvement"] += 0.05  # 5% error rate improvement
            impact["throughput_change_percent"] -= 10   # 10% throughput reduction

    for opt in polling_opts:
        if opt["type"] == "polling_strategy":
            impact["resource_efficiency"] += 15  # 15% better resource utilization

    return impact


def _generate_queue_insights(queue_predictor: QueuePredictor) -> Dict[str, Any]:
    """Generate insights from queue predictor data."""
    if not queue_predictor.queue_history:
        return {"status": "no_data"}

    recent_entry = queue_predictor.queue_history[-1]
    predicted_time = queue_predictor.predict_completion_time()
    optimal_delay = queue_predictor.get_optimal_delay()

    return {
        "current_queue_size": recent_entry.get('total_jobs', 0),
        "predicted_completion_time": predicted_time,
        "optimal_delay": optimal_delay,
        "queue_trend": _calculate_queue_trend(queue_predictor),
        "load_forecast": queue_predictor.get_load_forecast()
    }


def _calculate_queue_trend(queue_predictor: QueuePredictor) -> str:
    """Calculate queue size trend."""
    if len(queue_predictor.queue_history) < 5:
        return "insufficient_data"

    recent_sizes = [entry.get('total_jobs', 0) for entry in list(queue_predictor.queue_history)[-5:]]
    if len(recent_sizes) < 2:
        return "stable"

    trend = recent_sizes[-1] - recent_sizes[0]
    if trend > 10:
        return "growing"
    elif trend < -10:
        return "shrinking"
    else:
        return "stable"


def _calculate_system_health_score(performance_metrics: Dict[str, Any], queue_insights: Dict[str, Any]) -> int:
    """Calculate overall system health score (0-100)."""
    score = 100

    # Deduct for high error rates
    overall_metrics = performance_metrics.get('overall', {})
    error_rate = overall_metrics.get('error_rate', 0)
    score -= int(error_rate * 500)  # 50 points for 10% error rate

    # Deduct for large queue sizes
    queue_size = queue_insights.get('current_queue_size', 0)
    if queue_size > 100:
        score -= min(30, int((queue_size - 100) / 10))

    # Deduct for low cache performance
    cache_hit_rate = overall_metrics.get('cache_hit_rate', 1.0)
    if cache_hit_rate < 0.3:
        score -= 20

    return max(0, min(100, score))


def _get_health_status_text(score: int) -> str:
    """Convert health score to text status."""
    if score >= 90:
        return "excellent"
    elif score >= 70:
        return "good"
    elif score >= 50:
        return "fair"
    elif score >= 30:
        return "poor"
    else:
        return "critical"


def _get_health_alerts(performance_metrics: Dict[str, Any], queue_insights: Dict[str, Any]) -> List[str]:
    """Generate health alerts based on metrics."""
    alerts = []

    overall_metrics = performance_metrics.get('overall', {})
    error_rate = overall_metrics.get('error_rate', 0)

    if error_rate > 0.15:
        alerts.append(f"High error rate: {error_rate:.1%}")

    queue_size = queue_insights.get('current_queue_size', 0)
    if queue_size > 200:
        alerts.append(f"Large queue size: {queue_size} jobs")

    cache_hit_rate = overall_metrics.get('cache_hit_rate', 1.0)
    if cache_hit_rate < 0.2:
        alerts.append(f"Low cache hit rate: {cache_hit_rate:.1%}")

    return alerts


def _identify_optimization_opportunities(performance_metrics: Dict[str, Any], queue_insights: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify optimization opportunities."""
    opportunities = []

    # Check for profile optimization opportunity
    if _get_profile_age_minutes() > 60:  # Profile hasn't been tuned in an hour
        opportunities.append({
            "type": "auto_tune",
            "priority": "medium",
            "description": "Run auto-tune to optimize performance based on recent usage",
            "action": "call auto_tune_performance()"
        })

    # Check for caching opportunity
    overall_metrics = performance_metrics.get('overall', {})
    cache_hit_rate = overall_metrics.get('cache_hit_rate', 1.0)
    if cache_hit_rate < 0.4:
        opportunities.append({
            "type": "caching",
            "priority": "high",
            "description": f"Cache hit rate is low ({cache_hit_rate:.1%}). Enable more aggressive caching",
            "potential_improvement": "2-3x speedup for repeated operations"
        })

    return opportunities


def _get_profile_age_minutes() -> int:
    """Get age of current profile in minutes."""
    # This is a simplified implementation - in real code you'd track profile change time
    return 30  # Placeholder


def get_current_profile() -> PerformanceProfile:
    """Get current performance profile."""
    return _performance_profile


def reset_performance_state() -> None:
    """Reset performance state - useful for testing."""
    global _performance_profile, _performance_history, _tuning_recommendations
    _performance_profile = PerformanceProfile.balanced()
    _performance_history.clear()
    _tuning_recommendations.clear()


__all__ = [
    'configure_performance_profile',
    'auto_tune_performance',
    'get_performance_status',
    'get_current_profile',
    'reset_performance_state'
]