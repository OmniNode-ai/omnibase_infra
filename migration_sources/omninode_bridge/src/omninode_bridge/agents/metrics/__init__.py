"""
Performance Metrics Framework.

Provides comprehensive metrics collection, storage, and alerting with <10ms overhead.

Components:
- MetricsCollector: Core metrics collection with ring buffer
- RingBuffer: Lock-free ring buffer for high performance
- Kafka/PostgreSQL storage: Dual storage for real-time and historical analysis
- AlertRuleEngine: Threshold-based alerting
- Decorators: @timed, @counted for zero-boilerplate instrumentation

Performance:
- Collection overhead: <1ms per metric (target: <10ms)
- Batch flush: <50ms
- Throughput: 1000+ metrics/second

Usage:
    from omninode_bridge.agents.metrics import MetricsCollector, timed

    collector = MetricsCollector()

    # Decorator-based timing
    @timed("operation_time_ms")
    async def my_operation():
        await do_work()

    # Manual recording
    await collector.record_timing("routing_decision_time_ms", 5.2)
"""

from omninode_bridge.agents.metrics.collector import MetricsCollector
from omninode_bridge.agents.metrics.decorators import counted, timed, timing
from omninode_bridge.agents.metrics.models import (
    Alert,
    AlertRule,
    AlertSeverity,
    Metric,
    MetricType,
)

__all__ = [
    "MetricsCollector",
    "timed",
    "counted",
    "timing",
    "Metric",
    "MetricType",
    "Alert",
    "AlertSeverity",
    "AlertRule",
]
