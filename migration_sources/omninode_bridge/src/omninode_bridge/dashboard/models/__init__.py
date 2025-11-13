"""
Models for dashboard event tracing and metrics.

This package provides Pydantic models for type-safe event tracing,
session metrics, and correlated event analysis.

Models:
    ModelEventTrace: Event trace results for debugging sessions
    ModelSessionMetrics: Performance metrics for code generation sessions
    ModelCorrelatedEvent: Individual correlated event data
"""

from omninode_bridge.dashboard.models.model_correlated_event import ModelCorrelatedEvent
from omninode_bridge.dashboard.models.model_event_trace import ModelEventTrace
from omninode_bridge.dashboard.models.model_session_metrics import ModelSessionMetrics

__all__ = ["ModelEventTrace", "ModelSessionMetrics", "ModelCorrelatedEvent"]
