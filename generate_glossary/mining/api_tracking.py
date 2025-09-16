"""API usage tracking and analytics module.

This module provides comprehensive API usage monitoring, performance analysis,
feature adoption tracking, cost estimation, and optimization recommendations
for the Firecrawl-based mining system.
"""

import time
import logging
import statistics
from typing import Dict, Any, List, Optional
from datetime import datetime

from .models import ApiUsageStats, PerformanceProfile, QueuePredictor
from generate_glossary.utils.error_handler import processing_context
from generate_glossary.utils.logger import get_logger, log_processing_step, log_with_context


logger = get_logger(__name__)

# Global API usage tracking
_api_usage_stats = ApiUsageStats()
_usage_history: List[Dict[str, Any]] = []
_feature_benchmarks: Dict[str, Any] = {}


def track_api_usage(api_usage_stats: Optional[ApiUsageStats] = None) -> Dict[str, Any]:
    """Get comprehensive API usage tracking statistics with v2.2.0 performance benchmarking.

    Enhanced analytics including:
    - Individual endpoint usage counts with performance metrics
    - Total API calls made with success/error rates
    - Usage patterns and trends with predictive analysis
    - v2.2.0 feature performance comparison
    - Optimization recommendations based on usage patterns
    - Real-time performance monitoring and cost analysis

    Args:
        api_usage_stats: Optional specific stats instance, uses global if None

    Returns:
        Dictionary containing comprehensive usage and performance statistics
    """
    if api_usage_stats is None:
        api_usage_stats = _api_usage_stats

    with processing_context("track_api_usage") as correlation_id:
        start_time = time.time()
        log_processing_step(
            logger,
            "track_api_usage",
            "started",
            {
                "total_calls": api_usage_stats.total_calls,
                "performance_analysis": True,
                "v2.2.0_benchmarking": True
            },
            correlation_id=correlation_id
        )

        # Enhanced usage data with performance metrics
        usage_data = api_usage_stats.model_dump()

        # Enhanced performance metrics from API usage stats
        performance_metrics = api_usage_stats.get_performance_metrics()

        # Usage patterns with v2.2.0 feature analysis
        usage_patterns = analyze_usage_patterns(usage_data, performance_metrics)

        # Feature impact calculations
        feature_impacts = {}
        for feature in ['map_calls', 'batch_scrape_calls', 'queue_status_calls']:
            feature_impacts[feature] = calculate_feature_impact(feature, performance_metrics)

        # Generate optimization recommendations
        optimization_recommendations = generate_optimization_recommendations(
            usage_data, performance_metrics
        )

        # Generate v2.2.0 benchmarks
        v220_benchmarks = generate_v220_benchmarks(performance_metrics, usage_data)

        # Estimate API costs
        cost_analysis = estimate_api_costs(usage_data, performance_metrics)

        # Compile comprehensive tracking report
        tracking_report = {
            "timestamp": datetime.now().isoformat(),
            "tracking_duration_ms": int((time.time() - start_time) * 1000),
            "usage_statistics": {
                "basic_stats": usage_data,
                "performance_metrics": performance_metrics,
                "usage_patterns": usage_patterns
            },
            "v2_2_0_analysis": {
                "feature_impacts": feature_impacts,
                "benchmarks": v220_benchmarks,
                "adoption_insights": usage_patterns.get("v2_2_0_feature_adoption", {})
            },
            "optimization": {
                "recommendations": optimization_recommendations,
                "cost_analysis": cost_analysis
            }
        }

        # Store in history for trend analysis
        _usage_history.append({
            "timestamp": time.time(),
            "stats": usage_data,
            "metrics": performance_metrics
        })

        # Keep only recent history
        if len(_usage_history) > 100:
            _usage_history.pop(0)

        log_processing_step(
            logger,
            "track_api_usage",
            "completed",
            {
                "recommendations_count": len(optimization_recommendations),
                "feature_impacts_analyzed": len(feature_impacts),
                "benchmarked_features": len(v220_benchmarks.get("feature_benchmarks", {}))
            },
            correlation_id=correlation_id
        )

        return tracking_report


def analyze_usage_patterns(usage_data: Dict[str, Any], performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze usage patterns with v2.2.0 feature insights.

    Args:
        usage_data: Raw usage statistics
        performance_metrics: Performance metrics data

    Returns:
        Dictionary with usage pattern analysis
    """
    # Filter out non-endpoint data
    endpoint_data = {k: v for k, v in usage_data.items()
                    if k.endswith('_calls') and k != 'total_calls' and v > 0}

    if not endpoint_data:
        return {"most_used_endpoint": "none", "endpoint_distribution": {}}

    patterns = {
        "most_used_endpoint": max(endpoint_data, key=endpoint_data.get).replace('_calls', ''),
        "least_used_endpoint": min(endpoint_data, key=endpoint_data.get).replace('_calls', ''),
        "endpoint_distribution": {k.replace('_calls', ''): v / max(1, usage_data.get("total_calls", 1))
                                for k, v in endpoint_data.items()},
        "v2_2_0_feature_adoption": {},
        "usage_efficiency": calculate_usage_efficiency(usage_data, performance_metrics)
    }

    # Analyze v2.2.0 feature adoption
    v220_features = {
        "map_calls": "15x faster Map endpoint",
        "batch_scrape_calls": "500% batch scrape improvement",
        "queue_status_calls": "Real-time queue monitoring"
    }

    for feature_key, feature_name in v220_features.items():
        if feature_key in usage_data and usage_data[feature_key] > 0:
            adoption_rate = usage_data[feature_key] / max(1, usage_data.get("total_calls", 1))
            patterns["v2_2_0_feature_adoption"][feature_name] = {
                "usage_count": usage_data[feature_key],
                "adoption_rate": adoption_rate,
                "performance_impact": calculate_feature_impact(feature_key, performance_metrics)
            }

    return patterns


def calculate_feature_impact(feature_key: str, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate the performance impact of v2.2.0 features.

    Args:
        feature_key: API endpoint key (e.g., 'map_calls')
        performance_metrics: Performance metrics dictionary

    Returns:
        Dictionary with feature impact analysis
    """
    impact = {"status": "unknown"}

    if feature_key in performance_metrics.get("per_endpoint", {}):
        endpoint_metrics = performance_metrics["per_endpoint"][feature_key]
        avg_duration = endpoint_metrics.get("avg_duration", 0)

        # Define expected performance improvements for v2.2.0 features
        expected_improvements = {
            "map_calls": {"baseline": 15.0, "improvement": 15},  # 15x faster
            "batch_scrape_calls": {"baseline": 10.0, "improvement": 5},  # 500% improvement
            "queue_status_calls": {"baseline": 2.0, "improvement": 1}  # Real-time monitoring
        }

        if feature_key in expected_improvements:
            expected = expected_improvements[feature_key]
            improvement_factor = expected["baseline"] / max(avg_duration, 0.1)

            impact = {
                "status": "measured",
                "avg_duration": avg_duration,
                "expected_baseline": expected["baseline"],
                "actual_improvement_factor": improvement_factor,
                "expected_improvement_factor": expected["improvement"],
                "performance_rating": ("excellent" if improvement_factor >= expected["improvement"] * 0.8
                                     else "good" if improvement_factor >= expected["improvement"] * 0.5
                                     else "needs_optimization")
            }

    return impact


def generate_optimization_recommendations(usage_data: Dict[str, Any],
                                        performance_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate optimization recommendations based on usage patterns and performance data.

    Args:
        usage_data: API usage statistics
        performance_metrics: Performance metrics

    Returns:
        List of optimization recommendations
    """
    recommendations = []

    # Cache optimization recommendations
    cache_hit_rate = performance_metrics.get("overall", {}).get("cache_hit_rate", 0)
    if cache_hit_rate < 0.2:
        recommendations.append({
            "category": "caching",
            "priority": "high",
            "title": "Improve cache hit rate",
            "description": f"Current cache hit rate is {cache_hit_rate:.1%}. "
                         "Consider increasing cache TTL or implementing smarter caching strategies.",
            "potential_impact": "20-40% performance improvement",
            "action_items": [
                "Increase cache TTL from 1 hour to 2-4 hours",
                "Implement predictive caching for frequently accessed content",
                "Use cache warming strategies during low-usage periods"
            ]
        })

    # v2.2.0 feature adoption recommendations
    map_usage = usage_data.get("map_calls", 0)
    total_usage = usage_data.get("total_calls", 1)

    if map_usage / total_usage < 0.1 and total_usage > 50:
        recommendations.append({
            "category": "feature_adoption",
            "priority": "high",
            "title": "Increase usage of 15x faster Map endpoint",
            "description": "Map endpoint usage is low but offers 15x performance improvement. "
                         "Consider using it more for URL discovery tasks.",
            "potential_impact": "Up to 15x faster URL mapping",
            "action_items": [
                "Replace manual URL discovery with Map endpoint",
                "Implement batch mapping for multiple domains",
                "Use Map endpoint for comprehensive site coverage"
            ]
        })

    # Batch scraping recommendations
    batch_usage = usage_data.get("batch_scrape_calls", 0)
    regular_scrape_usage = usage_data.get("scrape_calls", 0)

    if regular_scrape_usage > batch_usage * 10 and batch_usage > 0:
        recommendations.append({
            "category": "efficiency",
            "priority": "medium",
            "title": "Increase batch scraping usage",
            "description": f"You're using regular scraping {regular_scrape_usage} times vs batch scraping "
                         f"{batch_usage} times. Batch scraping offers 500% performance improvement.",
            "potential_impact": "5x faster scraping operations",
            "action_items": [
                "Group related URLs into batch scraping operations",
                "Set optimal batch sizes based on content complexity",
                "Use async batch processing for better throughput"
            ]
        })

    # Error rate recommendations
    error_rate = performance_metrics.get("overall", {}).get("error_rate", 0)
    if error_rate > 0.1:  # 10% error rate
        recommendations.append({
            "category": "reliability",
            "priority": "high",
            "title": "Reduce API error rate",
            "description": f"Current error rate is {error_rate:.1%}, which is above optimal threshold. "
                         "This may indicate issues with request parameters or rate limiting.",
            "potential_impact": "Improved reliability and reduced retry overhead",
            "action_items": [
                "Implement exponential backoff for failed requests",
                "Validate request parameters before API calls",
                "Monitor rate limits and implement intelligent throttling"
            ]
        })

    return recommendations


def generate_v220_benchmarks(performance_metrics: Dict[str, Any], usage_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive v2.2.0 feature benchmarks.

    Args:
        performance_metrics: Performance metrics data
        usage_data: API usage statistics

    Returns:
        Dictionary with v2.2.0 benchmarking results
    """
    benchmarks = {
        "overall_performance_score": 0,
        "feature_benchmarks": {}
    }

    # v2.2.0 feature benchmarks
    v220_features = {
        "map": {
            "calls_key": "map_calls",
            "expected_improvement": 15.0,
            "baseline_duration": 15.0,
            "description": "15x faster Map endpoint"
        },
        "batch_scrape": {
            "calls_key": "batch_scrape_calls",
            "expected_improvement": 5.0,
            "baseline_duration": 10.0,
            "description": "500% batch scrape improvement"
        },
        "queue_status": {
            "calls_key": "queue_status_calls",
            "expected_improvement": 1.0,
            "baseline_duration": 2.0,
            "description": "Real-time queue monitoring"
        }
    }

    total_score = 0
    scored_features = 0

    for feature_name, feature_config in v220_features.items():
        calls_key = feature_config["calls_key"]

        if calls_key in usage_data and usage_data[calls_key] > 0:
            feature_benchmark = {
                "usage_count": usage_data[calls_key],
                "expected_improvement": feature_config["expected_improvement"],
                "baseline_duration": feature_config["baseline_duration"],
                "description": feature_config["description"],
                "status": "active"
            }

            # Get actual performance metrics
            if calls_key in performance_metrics.get("per_endpoint", {}):
                endpoint_metrics = performance_metrics["per_endpoint"][calls_key]
                actual_duration = endpoint_metrics.get("avg_duration", feature_config["baseline_duration"])

                improvement_factor = feature_config["baseline_duration"] / max(actual_duration, 0.1)
                performance_score = min(100, (improvement_factor / feature_config["expected_improvement"]) * 100)

                feature_benchmark.update({
                    "actual_duration": actual_duration,
                    "actual_improvement_factor": improvement_factor,
                    "performance_score": performance_score
                })

                total_score += performance_score
                scored_features += 1
            else:
                feature_benchmark.update({
                    "status": "no_performance_data",
                    "performance_score": 0
                })

            benchmarks["feature_benchmarks"][feature_name] = feature_benchmark
        else:
            benchmarks["feature_benchmarks"][feature_name] = {
                "status": "unused",
                "description": feature_config["description"],
                "potential_improvement": f"{feature_config['expected_improvement']}x"
            }

    # Calculate overall performance score
    if scored_features > 0:
        benchmarks["overall_performance_score"] = total_score / scored_features
    else:
        benchmarks["overall_performance_score"] = 0

    return benchmarks


def estimate_api_costs(usage_data: Dict[str, Any], performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate API costs and efficiency metrics.

    Args:
        usage_data: API usage statistics
        performance_metrics: Performance metrics

    Returns:
        Dictionary with cost analysis
    """
    # Get cost estimates from configuration
    from generate_glossary.config import get_mining_config
    mining_config = get_mining_config()
    cost_per_call = mining_config.api_costs

    total_estimated_cost = 0.0
    cost_breakdown = {}

    for call_type, count in usage_data.items():
        if call_type in cost_per_call and count > 0:
            cost = count * cost_per_call[call_type]
            cost_breakdown[call_type] = {
                "count": count,
                "cost_per_call": cost_per_call[call_type],
                "total_cost": cost
            }
            total_estimated_cost += cost

    # Calculate efficiency metrics
    efficiency_metrics = {}
    if "scrape_calls" in usage_data and "batch_scrape_calls" in usage_data:
        regular_scrapes = usage_data["scrape_calls"]
        batch_scrapes = usage_data["batch_scrape_calls"]
        total_scrapes = regular_scrapes + batch_scrapes

        if total_scrapes > 0:
            batch_efficiency = batch_scrapes / total_scrapes
            potential_savings = (regular_scrapes * 0.02) - (regular_scrapes * 0.1 / 10)  # Assuming 10:1 efficiency
            efficiency_metrics = {
                "batch_usage_rate": batch_efficiency,
                "potential_cost_savings": max(0, potential_savings),
                "efficiency_rating": ("excellent" if batch_efficiency > 0.7
                                    else "good" if batch_efficiency > 0.3
                                    else "needs_improvement")
            }

    return {
        "estimated_total_cost": round(total_estimated_cost, 4),
        "cost_breakdown": cost_breakdown,
        "efficiency_metrics": efficiency_metrics,
        "cost_optimization_opportunities": generate_cost_optimization_tips(usage_data, cost_breakdown)
    }


def calculate_usage_efficiency(usage_data: Dict[str, Any], performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate overall usage efficiency metrics.

    Args:
        usage_data: API usage statistics
        performance_metrics: Performance metrics

    Returns:
        Dictionary with efficiency analysis
    """
    efficiency_score = 100
    factors = []

    # Cache efficiency
    cache_hit_rate = performance_metrics.get("overall", {}).get("cache_hit_rate", 0)
    cache_factor = cache_hit_rate * 30  # Cache contributes up to 30 points
    efficiency_score = efficiency_score - 30 + cache_factor
    factors.append(f"Cache hit rate: {cache_hit_rate:.1%}")

    # Error rate efficiency
    error_rate = performance_metrics.get("overall", {}).get("error_rate", 0)
    error_factor = (1 - error_rate) * 25  # Error rate contributes up to 25 points
    efficiency_score = efficiency_score - 25 + error_factor
    factors.append(f"Error rate: {error_rate:.1%}")

    # Feature adoption efficiency
    total_calls = usage_data.get("total_calls", 1)
    advanced_calls = usage_data.get("batch_scrape_calls", 0) + usage_data.get("map_calls", 0)
    adoption_rate = advanced_calls / max(1, total_calls)
    adoption_factor = adoption_rate * 25  # Feature adoption contributes up to 25 points
    efficiency_score = efficiency_score - 25 + adoption_factor
    factors.append(f"Advanced feature adoption: {adoption_rate:.1%}")

    # Performance consistency
    consistency_factor = 20  # Default 20 points for consistency
    if "per_endpoint" in performance_metrics:
        durations = []
        for endpoint_metrics in performance_metrics["per_endpoint"].values():
            if "avg_duration" in endpoint_metrics:
                durations.append(endpoint_metrics["avg_duration"])

        if durations and len(durations) > 1:
            cv = statistics.stdev(durations) / statistics.mean(durations)  # Coefficient of variation
            consistency_factor = max(0, 20 * (1 - cv))  # Lower variation = higher score

    efficiency_score = efficiency_score - 20 + consistency_factor
    factors.append(f"Performance consistency: {(consistency_factor/20):.1%}")

    return {
        "overall_score": max(0, min(100, efficiency_score)),
        "score_factors": factors,
        "rating": ("excellent" if efficiency_score >= 90
                  else "good" if efficiency_score >= 70
                  else "fair" if efficiency_score >= 50
                  else "needs_improvement")
    }


def generate_cost_optimization_tips(usage_data: Dict[str, Any], cost_breakdown: Dict[str, Any]) -> List[str]:
    """Generate cost optimization tips based on usage patterns.

    Args:
        usage_data: API usage statistics
        cost_breakdown: Cost breakdown by endpoint

    Returns:
        List of cost optimization tips
    """
    tips = []

    # Check batch vs individual scraping
    regular_scrapes = usage_data.get("scrape_calls", 0)
    batch_scrapes = usage_data.get("batch_scrape_calls", 0)

    if regular_scrapes > batch_scrapes * 5:
        tips.append("Consider using batch scraping more - it's 5x more efficient despite higher per-call cost")

    # Check map endpoint usage
    map_calls = usage_data.get("map_calls", 0)
    total_calls = usage_data.get("total_calls", 1)

    if map_calls / total_calls < 0.05:
        tips.append("Map endpoint is underutilized - it's 15x faster and cheaper per operation")

    # Check for high-cost endpoints
    highest_cost_endpoint = None
    highest_cost = 0
    for endpoint, breakdown in cost_breakdown.items():
        if breakdown["total_cost"] > highest_cost:
            highest_cost = breakdown["total_cost"]
            highest_cost_endpoint = endpoint

    if highest_cost_endpoint and highest_cost > 0.5:
        tips.append(f"Consider optimizing {highest_cost_endpoint.replace('_calls', '')} usage - "
                   f"it's your highest cost endpoint at ${highest_cost:.2f}")

    return tips


def get_api_usage_stats() -> ApiUsageStats:
    """Get the global API usage stats instance."""
    return _api_usage_stats


def reset_api_tracking() -> None:
    """Reset API tracking state - useful for testing."""
    global _api_usage_stats, _usage_history, _feature_benchmarks
    _api_usage_stats = ApiUsageStats()
    _usage_history.clear()
    _feature_benchmarks.clear()


def get_usage_trends(hours_back: int = 24) -> Dict[str, Any]:
    """Get usage trends over time.

    Args:
        hours_back: Number of hours to look back for trends

    Returns:
        Dictionary with trend analysis
    """
    cutoff_time = time.time() - (hours_back * 3600)
    recent_history = [h for h in _usage_history if h["timestamp"] >= cutoff_time]

    if len(recent_history) < 2:
        return {"status": "insufficient_data"}

    # Calculate trends
    first_stats = recent_history[0]["stats"]
    last_stats = recent_history[-1]["stats"]

    trends = {}
    for key in ["total_calls", "search_calls", "scrape_calls", "map_calls", "batch_scrape_calls"]:
        first_val = first_stats.get(key, 0)
        last_val = last_stats.get(key, 0)
        if first_val > 0:
            change_pct = ((last_val - first_val) / first_val) * 100
            trends[key] = {
                "change_percent": change_pct,
                "direction": "increasing" if change_pct > 5 else "decreasing" if change_pct < -5 else "stable"
            }

    return {
        "status": "available",
        "time_range_hours": hours_back,
        "data_points": len(recent_history),
        "trends": trends
    }


__all__ = [
    'track_api_usage',
    'analyze_usage_patterns',
    'calculate_feature_impact',
    'generate_optimization_recommendations',
    'generate_v220_benchmarks',
    'estimate_api_costs',
    'calculate_usage_efficiency',
    'get_api_usage_stats',
    'reset_api_tracking',
    'get_usage_trends'
]