"""
Optimization Analytics Module for LLM Performance Tracking.

This module provides comprehensive tracking for all LLM operations including
performance metrics, optimization statistics, and analytics without cost tracking.

Features:
- Performance tracking for all LLM operations
- Smart consensus savings metrics
- Cache performance analytics
- Response time statistics
- Optimization insights and recommendations
- Integration hooks with existing LLM utilities
"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from datetime import datetime, timedelta

from .logger import get_logger

logger = get_logger("optimization_tracker")


@dataclass
class PerformanceMetric:
    """Individual performance metric for an LLM operation."""
    timestamp: float
    operation_type: str  # "consensus", "single", "cache_hit", "cache_miss"
    model: str
    response_time: float
    success: bool
    optimization_type: Optional[str] = None  # "smart_consensus", "cache_hit", etc.
    api_calls_made: int = 1
    api_calls_saved: int = 0
    cache_hit: bool = False
    semantic_cache_hit: bool = False
    confidence_score: Optional[float] = None
    agreement_score: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationStats:
    """Aggregated optimization statistics."""
    total_operations: int = 0
    total_api_calls_made: int = 0
    total_api_calls_saved: int = 0
    cache_hits: int = 0
    semantic_cache_hits: int = 0
    smart_consensus_reductions: int = 0
    
    total_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    
    success_count: int = 0
    error_count: int = 0
    
    # Consensus-specific stats
    consensus_operations: int = 0
    early_consensus_count: int = 0
    average_confidence_score: float = 0.0
    average_agreement_score: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_operations == 0:
            return 0.0
        return (self.success_count / self.total_operations) * 100
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        if self.total_operations == 0:
            return 0.0
        return self.total_response_time / self.total_operations
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        if self.total_operations == 0:
            return 0.0
        return (self.cache_hits / self.total_operations) * 100
    
    @property
    def semantic_cache_hit_rate(self) -> float:
        """Calculate semantic cache hit rate percentage."""
        if self.total_operations == 0:
            return 0.0
        return (self.semantic_cache_hits / self.total_operations) * 100
    
    @property
    def api_call_savings_rate(self) -> float:
        """Calculate percentage of API calls saved through optimization."""
        total_potential_calls = self.total_api_calls_made + self.total_api_calls_saved
        if total_potential_calls == 0:
            return 0.0
        return (self.total_api_calls_saved / total_potential_calls) * 100
    
    @property
    def smart_consensus_effectiveness(self) -> float:
        """Calculate effectiveness of smart consensus (percentage of consensus operations that used early stopping)."""
        if self.consensus_operations == 0:
            return 0.0
        return (self.early_consensus_count / self.consensus_operations) * 100


class OptimizationTracker:
    """Thread-safe optimization tracker for LLM performance analytics."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else Path("data/optimization")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
        self._metrics: List[PerformanceMetric] = []
        self._session_stats = OptimizationStats()
        self._model_stats: Dict[str, OptimizationStats] = defaultdict(OptimizationStats)
        self._operation_stats: Dict[str, OptimizationStats] = defaultdict(OptimizationStats)
        
        # Recent operations for quick analysis
        self._recent_operations = deque(maxlen=1000)
        
        self.session_start_time = time.time()
        
        # Load previous session data if available
        self._load_session_data()
        
        logger.debug(f"OptimizationTracker initialized with storage at {self.storage_path}")
    
    def record_operation(
        self,
        operation_type: str,
        model: str,
        response_time: float,
        success: bool,
        optimization_type: Optional[str] = None,
        api_calls_made: int = 1,
        api_calls_saved: int = 0,
        cache_hit: bool = False,
        semantic_cache_hit: bool = False,
        confidence_score: Optional[float] = None,
        agreement_score: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Record a single LLM operation with performance metrics."""
        with self._lock:
            metric = PerformanceMetric(
                timestamp=time.time(),
                operation_type=operation_type,
                model=model,
                response_time=response_time,
                success=success,
                optimization_type=optimization_type,
                api_calls_made=api_calls_made,
                api_calls_saved=api_calls_saved,
                cache_hit=cache_hit,
                semantic_cache_hit=semantic_cache_hit,
                confidence_score=confidence_score,
                agreement_score=agreement_score,
                context=context or {}
            )
            
            self._metrics.append(metric)
            self._recent_operations.append(metric)
            
            # Update session stats
            self._update_stats(self._session_stats, metric)
            
            # Update model-specific stats
            self._update_stats(self._model_stats[model], metric)
            
            # Update operation-specific stats
            self._update_stats(self._operation_stats[operation_type], metric)
            
            logger.debug(f"Recorded {operation_type} operation: {model}, {response_time:.2f}s, success={success}")
    
    def _update_stats(self, stats: OptimizationStats, metric: PerformanceMetric):
        """Update statistics with a new metric."""
        stats.total_operations += 1
        stats.total_api_calls_made += metric.api_calls_made
        stats.total_api_calls_saved += metric.api_calls_saved
        
        if metric.cache_hit:
            stats.cache_hits += 1
        if metric.semantic_cache_hit:
            stats.semantic_cache_hits += 1
        
        if metric.optimization_type == "smart_consensus":
            stats.smart_consensus_reductions += 1
        
        stats.total_response_time += metric.response_time
        stats.min_response_time = min(stats.min_response_time, metric.response_time)
        stats.max_response_time = max(stats.max_response_time, metric.response_time)
        
        if metric.success:
            stats.success_count += 1
        else:
            stats.error_count += 1
        
        # Consensus-specific updates
        if metric.operation_type == "consensus":
            stats.consensus_operations += 1
            if metric.optimization_type == "smart_consensus":
                stats.early_consensus_count += 1
            
            if metric.confidence_score is not None:
                # Update running average
                old_count = max(1, stats.consensus_operations - 1)
                stats.average_confidence_score = (
                    (stats.average_confidence_score * old_count + metric.confidence_score) / 
                    stats.consensus_operations
                )
            
            if metric.agreement_score is not None:
                # Update running average
                old_count = max(1, stats.consensus_operations - 1)
                stats.average_agreement_score = (
                    (stats.average_agreement_score * old_count + metric.agreement_score) /
                    stats.consensus_operations
                )
    
    def get_session_stats(self) -> OptimizationStats:
        """Get current session optimization statistics."""
        with self._lock:
            return OptimizationStats(**asdict(self._session_stats))
    
    def get_model_stats(self, model: Optional[str] = None) -> Dict[str, OptimizationStats]:
        """Get model-specific optimization statistics."""
        with self._lock:
            if model:
                return {model: OptimizationStats(**asdict(self._model_stats[model]))}
            return {
                model: OptimizationStats(**asdict(stats)) 
                for model, stats in self._model_stats.items()
            }
    
    def get_operation_stats(self, operation_type: Optional[str] = None) -> Dict[str, OptimizationStats]:
        """Get operation-specific optimization statistics."""
        with self._lock:
            if operation_type:
                return {operation_type: OptimizationStats(**asdict(self._operation_stats[operation_type]))}
            return {
                op_type: OptimizationStats(**asdict(stats))
                for op_type, stats in self._operation_stats.items()
            }
    
    def get_recent_operations(self, limit: int = 100) -> List[PerformanceMetric]:
        """Get recent operations for analysis."""
        with self._lock:
            return list(self._recent_operations)[-limit:]
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report."""
        with self._lock:
            stats = self._session_stats
            session_duration = time.time() - self.session_start_time
            
            report = []
            report.append("=== LLM Optimization Performance Report ===")
            report.append(f"Session Duration: {session_duration:.1f} seconds")
            report.append("")
            
            # Overall Statistics
            report.append("Overall Performance:")
            report.append(f"  Total Operations: {stats.total_operations}")
            report.append(f"  Success Rate: {stats.success_rate:.1f}%")
            report.append(f"  Average Response Time: {stats.average_response_time:.2f}s")
            report.append(f"  Min Response Time: {stats.min_response_time:.2f}s")
            report.append(f"  Max Response Time: {stats.max_response_time:.2f}s")
            report.append("")
            
            # API Call Optimization
            report.append("API Call Optimization:")
            report.append(f"  Total API Calls Made: {stats.total_api_calls_made}")
            report.append(f"  Total API Calls Saved: {stats.total_api_calls_saved}")
            report.append(f"  API Call Savings Rate: {stats.api_call_savings_rate:.1f}%")
            report.append("")
            
            # Caching Performance
            report.append("Caching Performance:")
            report.append(f"  Cache Hits: {stats.cache_hits}")
            report.append(f"  Semantic Cache Hits: {stats.semantic_cache_hits}")
            report.append(f"  Cache Hit Rate: {stats.cache_hit_rate:.1f}%")
            report.append(f"  Semantic Cache Hit Rate: {stats.semantic_cache_hit_rate:.1f}%")
            report.append("")
            
            # Smart Consensus Performance
            if stats.consensus_operations > 0:
                report.append("Smart Consensus Performance:")
                report.append(f"  Total Consensus Operations: {stats.consensus_operations}")
                report.append(f"  Early Consensus Count: {stats.early_consensus_count}")
                report.append(f"  Smart Consensus Effectiveness: {stats.smart_consensus_effectiveness:.1f}%")
                report.append(f"  Average Confidence Score: {stats.average_confidence_score:.2f}")
                report.append(f"  Average Agreement Score: {stats.average_agreement_score:.2f}")
                report.append("")
            
            # Model Performance Breakdown
            if self._model_stats:
                report.append("Model Performance Breakdown:")
                for model, model_stats in self._model_stats.items():
                    report.append(f"  {model}:")
                    report.append(f"    Operations: {model_stats.total_operations}")
                    report.append(f"    Success Rate: {model_stats.success_rate:.1f}%")
                    report.append(f"    Avg Response Time: {model_stats.average_response_time:.2f}s")
                    report.append(f"    API Calls Saved: {model_stats.total_api_calls_saved}")
                report.append("")
            
            # Operation Type Performance
            if self._operation_stats:
                report.append("Operation Type Performance:")
                for op_type, op_stats in self._operation_stats.items():
                    report.append(f"  {op_type}:")
                    report.append(f"    Operations: {op_stats.total_operations}")
                    report.append(f"    Success Rate: {op_stats.success_rate:.1f}%")
                    report.append(f"    Avg Response Time: {op_stats.average_response_time:.2f}s")
                    if op_type == "consensus" and op_stats.consensus_operations > 0:
                        report.append(f"    Smart Consensus Rate: {op_stats.smart_consensus_effectiveness:.1f}%")
                report.append("")
            
            return "\n".join(report)
    
    def generate_optimization_insights(self) -> List[str]:
        """Generate optimization insights and recommendations."""
        insights = []
        stats = self._session_stats
        
        # API Call Optimization insights
        if stats.api_call_savings_rate < 20:
            insights.append("Consider enabling more aggressive optimization settings to increase API call savings")
        elif stats.api_call_savings_rate > 50:
            insights.append("Excellent API call optimization! Current settings are working well")
        
        # Cache performance insights
        if stats.cache_hit_rate < 30:
            insights.append("Cache hit rate is low. Consider increasing cache TTL or enabling persistent caching")
        elif stats.cache_hit_rate > 70:
            insights.append("Great cache performance! Consider sharing cache insights with the team")
        
        # Smart consensus insights
        if stats.consensus_operations > 0:
            if stats.smart_consensus_effectiveness < 40:
                insights.append("Smart consensus is not triggering frequently. Consider lowering confidence thresholds")
            elif stats.smart_consensus_effectiveness > 70:
                insights.append("Smart consensus is very effective! Consider using it for more operations")
            
            if stats.average_confidence_score < 0.7:
                insights.append("Low average confidence scores. Consider reviewing model selection or prompts")
        
        # Performance insights
        if stats.average_response_time > 10.0:
            insights.append("Average response time is high. Consider using faster model tiers or reducing batch sizes")
        elif stats.average_response_time < 2.0:
            insights.append("Excellent response times! Current model selection is optimal for speed")
        
        # Success rate insights
        if stats.success_rate < 95:
            insights.append("Success rate could be improved. Review error patterns and consider fallback strategies")
        elif stats.success_rate > 98:
            insights.append("Excellent success rate! Current error handling is working well")
        
        return insights
    
    def export_metrics(self, format: str = "json") -> str:
        """Export all metrics in the specified format."""
        with self._lock:
            data = {
                "session_start_time": self.session_start_time,
                "export_time": time.time(),
                "session_stats": asdict(self._session_stats),
                "model_stats": {
                    model: asdict(stats) for model, stats in self._model_stats.items()
                },
                "operation_stats": {
                    op_type: asdict(stats) for op_type, stats in self._operation_stats.items()
                },
                "recent_operations": [asdict(metric) for metric in self._recent_operations]
            }
            
            if format.lower() == "json":
                return json.dumps(data, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")
    
    def reset_session(self):
        """Reset all session statistics and start a new tracking session."""
        with self._lock:
            self._metrics.clear()
            self._recent_operations.clear()
            self._session_stats = OptimizationStats()
            self._model_stats.clear()
            self._operation_stats.clear()
            self.session_start_time = time.time()
            
            logger.info("Optimization tracking session reset")
    
    def save_session_data(self):
        """Save current session data to storage."""
        try:
            session_file = self.storage_path / f"session_{int(self.session_start_time)}.json"
            with open(session_file, 'w') as f:
                f.write(self.export_metrics())
            logger.debug(f"Session data saved to {session_file}")
        except Exception as e:
            logger.error(f"Failed to save session data: {e}")
    
    def _load_session_data(self):
        """Load previous session data if available."""
        try:
            session_files = list(self.storage_path.glob("session_*.json"))
            if not session_files:
                return
            
            # Load the most recent session
            latest_file = max(session_files, key=lambda f: f.stat().st_mtime)
            logger.debug(f"Loading previous session data from {latest_file}")
            
            # Note: For now, we just log that previous data exists
            # In future versions, we could aggregate historical data
            
        except Exception as e:
            logger.debug(f"Could not load previous session data: {e}")


# Global optimization tracker instance
_global_tracker: Optional[OptimizationTracker] = None
_tracker_lock = threading.Lock()


def get_optimization_tracker() -> OptimizationTracker:
    """Get the global optimization tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        with _tracker_lock:
            if _global_tracker is None:
                from generate_glossary.config import get_llm_config
                llm_config = get_llm_config()
                storage_path = getattr(llm_config, 'cache_storage_path', 'data/cache')
                _global_tracker = OptimizationTracker(storage_path)
    return _global_tracker


def record_llm_operation(
    operation_type: str,
    model: str,
    response_time: float,
    success: bool,
    optimization_type: Optional[str] = None,
    api_calls_made: int = 1,
    api_calls_saved: int = 0,
    cache_hit: bool = False,
    semantic_cache_hit: bool = False,
    confidence_score: Optional[float] = None,
    agreement_score: Optional[float] = None,
    context: Optional[Dict[str, Any]] = None
):
    """Convenience function to record an LLM operation."""
    tracker = get_optimization_tracker()
    tracker.record_operation(
        operation_type=operation_type,
        model=model,
        response_time=response_time,
        success=success,
        optimization_type=optimization_type,
        api_calls_made=api_calls_made,
        api_calls_saved=api_calls_saved,
        cache_hit=cache_hit,
        semantic_cache_hit=semantic_cache_hit,
        confidence_score=confidence_score,
        agreement_score=agreement_score,
        context=context
    )


def get_performance_report() -> str:
    """Get a performance report from the global tracker."""
    tracker = get_optimization_tracker()
    return tracker.generate_performance_report()


def get_optimization_insights() -> List[str]:
    """Get optimization insights from the global tracker."""
    tracker = get_optimization_tracker()
    return tracker.generate_optimization_insights()


def reset_optimization_tracking():
    """Reset the global optimization tracker."""
    tracker = get_optimization_tracker()
    tracker.reset_session()


def export_optimization_data(format: str = "json") -> str:
    """Export optimization data from the global tracker."""
    tracker = get_optimization_tracker()
    return tracker.export_metrics(format)