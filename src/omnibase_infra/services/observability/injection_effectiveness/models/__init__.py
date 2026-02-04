# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pydantic models for injection effectiveness events.

These models correspond to the event payloads emitted by OMN-1889
(omniclaude hooks) and consumed by OMN-1890 (this consumer).

Event Types:
    - ModelContextUtilizationEvent: Context utilization detection results
    - ModelAgentMatchEvent: Agent routing accuracy metrics
    - ModelLatencyBreakdownEvent: Per-prompt latency breakdowns
"""

from omnibase_infra.services.observability.injection_effectiveness.models.model_agent_match import (
    ModelAgentMatchEvent,
)
from omnibase_infra.services.observability.injection_effectiveness.models.model_context_utilization import (
    ModelContextUtilizationEvent,
)
from omnibase_infra.services.observability.injection_effectiveness.models.model_latency_breakdown import (
    ModelLatencyBreakdownEvent,
)
from omnibase_infra.services.observability.injection_effectiveness.models.model_pattern_utilization import (
    ModelPatternUtilization,
)

__all__ = [
    "ModelAgentMatchEvent",
    "ModelContextUtilizationEvent",
    "ModelLatencyBreakdownEvent",
    "ModelPatternUtilization",
]
