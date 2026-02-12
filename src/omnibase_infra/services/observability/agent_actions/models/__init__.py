# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Observability Models for Agent Actions Consumer.

This package contains Pydantic models for the agent_actions observability
consumer. These models define the schema for events consumed from Kafka
and persisted to PostgreSQL.

Model Categories:
    - Envelope (strict): ModelObservabilityEnvelope - common metadata fields
    - Payload (strict): All other models - frozen, extra="forbid"

Design Decisions:
    - All models use frozen=True for thread safety
    - All models use extra="forbid" for strict schema compliance
    - All models use from_attributes=True for ORM/pytest-xdist compatibility
    - All models have created_at for TTL readiness
    - ModelExecutionLog has updated_at for lifecycle tracking
    - Zero dict[str, Any] - use dict[str, object] when needed

Idempotency Keys (per table):
    - agent_actions: id (UUID)
    - agent_routing_decisions: id (UUID)
    - agent_transformation_events: id (UUID)
    - router_performance_metrics: id (UUID)
    - agent_detection_failures: correlation_id (UUID)
    - agent_execution_logs: execution_id (UUID)
    - agent_status_events: id (UUID)

Example:
    >>> from omnibase_infra.services.observability.agent_actions.models import (
    ...     ModelObservabilityEnvelope,
    ...     ModelAgentAction,
    ...     ModelRoutingDecision,
    ... )
    >>> from datetime import datetime, UTC
    >>> from uuid import uuid4
    >>>
    >>> # Strict envelope validation
    >>> envelope = ModelObservabilityEnvelope(
    ...     event_id=uuid4(),
    ...     event_time=datetime.now(UTC),
    ...     producer_id="agent-observability-postgres",
    ...     schema_version="1.0.0",
    ... )
    >>>
    >>> # Strict payload - no extra fields allowed
    >>> action = ModelAgentAction(
    ...     id=uuid4(),
    ...     correlation_id=uuid4(),
    ...     agent_name="polymorphic-agent",
    ...     action_type="tool_call",
    ...     action_name="Read",
    ...     created_at=datetime.now(UTC),
    ... )
"""

from omnibase_infra.services.observability.agent_actions.models.model_agent_action import (
    ModelAgentAction,
)
from omnibase_infra.services.observability.agent_actions.models.model_agent_status_event import (
    ModelAgentStatusEvent,
)
from omnibase_infra.services.observability.agent_actions.models.model_detection_failure import (
    ModelDetectionFailure,
)
from omnibase_infra.services.observability.agent_actions.models.model_envelope import (
    ModelObservabilityEnvelope,
)
from omnibase_infra.services.observability.agent_actions.models.model_execution_log import (
    ModelExecutionLog,
)
from omnibase_infra.services.observability.agent_actions.models.model_performance_metric import (
    ModelPerformanceMetric,
)
from omnibase_infra.services.observability.agent_actions.models.model_routing_decision import (
    ModelRoutingDecision,
)
from omnibase_infra.services.observability.agent_actions.models.model_transformation_event import (
    ModelTransformationEvent,
)
from omnibase_infra.services.observability.agent_actions.models.model_ttl_cleanup_result import (
    ModelTTLCleanupResult,
)

__all__ = [
    "ModelAgentAction",
    "ModelAgentStatusEvent",
    "ModelDetectionFailure",
    "ModelExecutionLog",
    "ModelObservabilityEnvelope",
    "ModelPerformanceMetric",
    "ModelRoutingDecision",
    "ModelTTLCleanupResult",
    "ModelTransformationEvent",
]
