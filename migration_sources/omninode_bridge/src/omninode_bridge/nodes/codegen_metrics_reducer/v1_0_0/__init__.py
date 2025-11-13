"""
NodeCodegenMetricsReducer - Code Generation Metrics Aggregator.

Aggregates code generation metrics across workflows for analytics, monitoring,
and trend analysis. Part of the omninode_bridge code generation MVP.

ONEX v2.0 Compliance:
- Contract-first architecture with event-driven design
- Pure reducer logic with streaming aggregation
- FSM state tracking for workflow metrics
- 70% code reuse from NodeBridgeReducer patterns
"""

from .node import NodeCodegenMetricsReducer

__all__ = ["NodeCodegenMetricsReducer"]
