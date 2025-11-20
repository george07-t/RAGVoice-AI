"""
Metrics Logging System
=====================================

This module provides utilities for tracking and logging performance metrics
across the entire voice AI pipeline including:
- STT (Speech-to-Text) latency
- LLM inference time
- Tool execution time
- RAG retrieval time
- TTS (Text-to-Speech) generation time
- End-to-end response time

Usage:
    from metrics_logger import MetricsLogger
    
    metrics = MetricsLogger()
    metrics.start_timer("stt")
    # ... STT processing ...
    metrics.end_timer("stt")
    metrics.log_summary()
"""

import time
import logging
import json
from typing import Dict, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MetricEntry:
    """
    Represents a single metric measurement.
    
    Attributes:
        name: Metric identifier (e.g., 'stt_latency', 'llm_inference')
        duration: Time elapsed in seconds
        timestamp: When the metric was recorded
        metadata: Additional context (e.g., model name, query text)
    """
    name: str
    duration: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'duration': round(self.duration, 4),
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


class MetricsLogger:
    """
    Tracks and logs performance metrics for the entire AI pipeline.
    
    This class provides:
    - Timer management for measuring operation durations
    - Metric aggregation and statistics
    - JSON export for analysis
    - Real-time logging
    
    Example:
        >>> metrics = MetricsLogger()
        >>> metrics.start_timer("stt")
        >>> # ... perform STT ...
        >>> metrics.end_timer("stt", metadata={"model": "nova-3"})
        >>> print(metrics.get_summary())
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize the metrics logger.
        
        Args:
            log_file: Optional path to save metrics JSON file
        """
        self.metrics: List[MetricEntry] = []
        self.active_timers: Dict[str, float] = {}
        self.log_file = log_file
        
        # Session-level tracking
        self.session_start = time.time()
        self.total_queries = 0
        
    def start_timer(self, name: str) -> None:
        """
        Start timing a metric.
        
        Args:
            name: Metric identifier (e.g., 'stt', 'llm', 'rag_retrieval')
        """
        self.active_timers[name] = time.time()
        logger.debug(f"Timer started: {name}")
    
    def end_timer(
        self, 
        name: str, 
        metadata: Optional[Dict] = None
    ) -> float:
        """
        End timing a metric and record the duration.
        
        Args:
            name: Metric identifier (must match start_timer call)
            metadata: Additional context to log with the metric
            
        Returns:
            Duration in seconds
            
        Raises:
            ValueError: If timer was never started
        """
        if name not in self.active_timers:
            raise ValueError(f"Timer '{name}' was never started")
        
        start_time = self.active_timers.pop(name)
        duration = time.time() - start_time
        
        entry = MetricEntry(
            name=name,
            duration=duration,
            metadata=metadata or {}
        )
        
        self.metrics.append(entry)
        
        logger.info(f"✓ {name}: {duration:.3f}s {metadata or ''}")
        
        return duration
    
    def record_metric(
        self, 
        name: str, 
        duration: float, 
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Directly record a metric without using timers.
        
        Args:
            name: Metric identifier
            duration: Pre-calculated duration in seconds
            metadata: Additional context
        """
        entry = MetricEntry(
            name=name,
            duration=duration,
            metadata=metadata or {}
        )
        
        self.metrics.append(entry)
        logger.info(f"✓ {name}: {duration:.3f}s")
    
    def get_metrics_by_name(self, name: str) -> List[MetricEntry]:
        """
        Retrieve all metrics with a specific name.
        
        Args:
            name: Metric identifier to filter by
            
        Returns:
            List of matching metric entries
        """
        return [m for m in self.metrics if m.name == name]
    
    def get_average(self, name: str) -> Optional[float]:
        """
        Calculate average duration for a metric.
        
        Args:
            name: Metric identifier
            
        Returns:
            Average duration in seconds, or None if no metrics found
        """
        metrics = self.get_metrics_by_name(name)
        if not metrics:
            return None
        
        total = sum(m.duration for m in metrics)
        return total / len(metrics)
    
    def get_summary(self) -> Dict:
        """
        Generate a comprehensive summary of all metrics.
        
        Returns:
            Dictionary containing:
            - Total queries processed
            - Average times for each metric type
            - Min/max times
            - Session duration
        """
        summary = {
            'session_duration': time.time() - self.session_start,
            'total_queries': self.total_queries,
            'metrics': {}
        }
        
        # Group metrics by name
        metric_names = set(m.name for m in self.metrics)
        
        for name in metric_names:
            metrics = self.get_metrics_by_name(name)
            durations = [m.duration for m in metrics]
            
            summary['metrics'][name] = {
                'count': len(durations),
                'total': sum(durations),
                'average': sum(durations) / len(durations),
                'min': min(durations),
                'max': max(durations)
            }
        
        return summary
    
    def log_summary(self) -> None:
        """
        Log a formatted summary of all metrics to console.
        """
        summary = self.get_summary()
        
        logger.info("=" * 60)
        logger.info("PERFORMANCE METRICS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Session Duration: {summary['session_duration']:.2f}s")
        logger.info(f"Total Queries: {summary['total_queries']}")
        logger.info("-" * 60)
        
        for name, stats in summary['metrics'].items():
            logger.info(f"{name}:")
            logger.info(f"  Count: {stats['count']}")
            logger.info(f"  Average: {stats['average']:.3f}s")
            logger.info(f"  Min: {stats['min']:.3f}s")
            logger.info(f"  Max: {stats['max']:.3f}s")
            logger.info(f"  Total: {stats['total']:.3f}s")
        
        logger.info("=" * 60)
    
    def save_to_json(self, filepath: Optional[str] = None) -> None:
        """
        Save all metrics to a JSON file for analysis.
        
        Args:
            filepath: Output file path (overrides constructor log_file)
        """
        output_path = filepath or self.log_file
        
        if not output_path:
            logger.warning("No log file specified, skipping JSON export")
            return
        
        data = {
            'session_start': datetime.fromtimestamp(self.session_start).isoformat(),
            'session_duration': time.time() - self.session_start,
            'total_queries': self.total_queries,
            'metrics': [m.to_dict() for m in self.metrics],
            'summary': self.get_summary()
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metrics saved to {output_path}")
    
    def increment_query_count(self) -> None:
        """Increment the total query counter."""
        self.total_queries += 1
    
    def reset(self) -> None:
        """Clear all metrics and reset session."""
        self.metrics.clear()
        self.active_timers.clear()
        self.session_start = time.time()
        self.total_queries = 0
        logger.info("Metrics logger reset")


# Singleton instance for global access
_global_metrics_logger: Optional[MetricsLogger] = None


def get_metrics_logger(log_file: Optional[str] = None) -> MetricsLogger:
    """
    Get the global metrics logger instance (singleton pattern).
    
    Args:
        log_file: Optional log file path (only used on first call)
        
    Returns:
        Global MetricsLogger instance
    """
    global _global_metrics_logger
    
    if _global_metrics_logger is None:
        _global_metrics_logger = MetricsLogger(log_file=log_file)
        logger.info("Global metrics logger initialized")
    
    return _global_metrics_logger


# Example usage and testing
if __name__ == "__main__":
    # Create logger
    metrics = MetricsLogger(log_file="metrics/test_metrics.json")
    
    # Simulate STT
    metrics.start_timer("stt")
    time.sleep(0.5)
    metrics.end_timer("stt", metadata={"model": "nova-3", "language": "en"})
    
    # Simulate RAG retrieval
    metrics.start_timer("rag_retrieval")
    time.sleep(0.8)
    metrics.end_timer("rag_retrieval", metadata={"num_docs": 3, "query": "What is AI?"})
    
    # Simulate LLM inference
    metrics.start_timer("llm_inference")
    time.sleep(1.2)
    metrics.end_timer("llm_inference", metadata={"model": "gpt-4o-mini", "tokens": 150})
    
    # Simulate TTS
    metrics.start_timer("tts")
    time.sleep(0.3)
    metrics.end_timer("tts", metadata={"voice": "cartesia"})
    
    metrics.increment_query_count()
    
    # Another query
    metrics.start_timer("stt")
    time.sleep(0.6)
    metrics.end_timer("stt", metadata={"model": "nova-3", "language": "en"})
    
    metrics.start_timer("llm_inference")
    time.sleep(1.0)
    metrics.end_timer("llm_inference", metadata={"model": "gpt-4o-mini", "tokens": 200})
    
    metrics.increment_query_count()
    
    # Log summary
    metrics.log_summary()
    
    # Save to JSON
    metrics.save_to_json()
    
    print("\nTest completed! Check metrics/test_metrics.json for output.")
