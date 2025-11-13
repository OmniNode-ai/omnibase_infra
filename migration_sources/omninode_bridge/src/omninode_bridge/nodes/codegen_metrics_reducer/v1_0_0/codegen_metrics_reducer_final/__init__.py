"""
codegen_metrics_reducer - Aggregates code generation metrics from event streams for analytics, monitoring, and trend analysis. Consumes CODEGEN_* events from Kafka, aggregates by time window (hourly/daily/weekly), computes performance, quality, and cost metrics, and publishes GENERATION_METRICS_RECORDED events. Architecture: Pure aggregation logic (MetricsAggregator) + coordination I/O via MixinIntentPublisher + Intent executor publishes via EFFECT.

Generated: 2025-11-05T18:02:41.557888+00:00
ONEX v2.0 Compliant
"""

from .node import NodeCodegenMetricsReducerReducer

__all__ = ["NodeCodegenMetricsReducerReducer"]
