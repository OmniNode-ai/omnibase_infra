# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pydantic models for injection effectiveness events and query results.

These models correspond to the event payloads emitted by OMN-1889
(omniclaude hooks) and consumed by OMN-1890 (this consumer), plus
query result models added by OMN-2078.

Event Types:
    - ModelContextUtilizationEvent: Context utilization detection results
    - ModelAgentMatchEvent: Agent routing accuracy metrics
    - ModelLatencyBreakdownEvent: Per-prompt latency breakdowns

Query Models (OMN-2078):
    - ModelInjectionEffectivenessRow: Single row from injection_effectiveness table
    - ModelLatencyBreakdownRow: Single row from latency_breakdowns table
    - ModelPatternHitRateRow: Single row from pattern_hit_rates table
    - ModelInjectionEffectivenessQuery: Query filter parameters
    - ModelInjectionEffectivenessQueryResult: Paginated query result
"""

from omnibase_infra.services.observability.injection_effectiveness.models.model_agent_match import (
    ModelAgentMatchEvent,
)
from omnibase_infra.services.observability.injection_effectiveness.models.model_context_utilization import (
    ModelContextUtilizationEvent,
)
from omnibase_infra.services.observability.injection_effectiveness.models.model_injection_effectiveness_query import (
    ModelInjectionEffectivenessQuery,
)
from omnibase_infra.services.observability.injection_effectiveness.models.model_injection_effectiveness_query_result import (
    ModelInjectionEffectivenessQueryResult,
)
from omnibase_infra.services.observability.injection_effectiveness.models.model_injection_effectiveness_row import (
    ModelInjectionEffectivenessRow,
)
from omnibase_infra.services.observability.injection_effectiveness.models.model_invalidation_event import (
    ModelEffectivenessInvalidationEvent,
)
from omnibase_infra.services.observability.injection_effectiveness.models.model_latency_breakdown import (
    ModelLatencyBreakdownEvent,
)
from omnibase_infra.services.observability.injection_effectiveness.models.model_latency_breakdown_row import (
    ModelLatencyBreakdownRow,
)
from omnibase_infra.services.observability.injection_effectiveness.models.model_pattern_hit_rate_row import (
    ModelPatternHitRateRow,
)
from omnibase_infra.services.observability.injection_effectiveness.models.model_pattern_utilization import (
    ModelPatternUtilization,
)

__all__ = [
    "ModelAgentMatchEvent",
    "ModelContextUtilizationEvent",
    "ModelEffectivenessInvalidationEvent",
    "ModelInjectionEffectivenessQuery",
    "ModelInjectionEffectivenessQueryResult",
    "ModelInjectionEffectivenessRow",
    "ModelLatencyBreakdownEvent",
    "ModelLatencyBreakdownRow",
    "ModelPatternHitRateRow",
    "ModelPatternUtilization",
]
