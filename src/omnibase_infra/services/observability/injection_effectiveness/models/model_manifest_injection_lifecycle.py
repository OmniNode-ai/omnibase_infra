# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Manifest injection lifecycle event models.

Represents payloads from the three manifest injection lifecycle topics
emitted by omniclaude hooks (OMN-1888 audit trail).

Topics consumed:
    - onex.evt.omniclaude.manifest-injection-started.v1
    - onex.evt.omniclaude.manifest-injected.v1
    - onex.evt.omniclaude.manifest-injection-failed.v1

The same Pydantic model handles all three topics. The ``event_type``
discriminator field identifies which lifecycle stage was recorded.

Related Tickets:
    - OMN-1888: Manifest injection effectiveness measurement loop
    - OMN-2942: Add consumer for manifest injection lifecycle events
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelManifestInjectionLifecycleEvent(BaseModel):
    """Manifest injection lifecycle event from omniclaude hooks.

    Emitted at each stage of agent manifest loading:
        - ``manifest_injection_started``: Injection attempt begins.
        - ``manifest_injected``: Injection completed successfully.
        - ``manifest_injection_failed``: Injection failed with error.

    This event populates the ``manifest_injection_lifecycle`` table,
    providing the end-to-end audit trail required by OMN-1888.

    Attributes:
        event_type: Lifecycle stage discriminator.
        entity_id: Session identifier as UUID (partition key).
        session_id: Session identifier as UUID.
        correlation_id: Correlation ID for distributed tracing.
        causation_id: ID of the prompt event that triggered injection.
        emitted_at: Timestamp when the hook emitted this event (UTC).
        agent_label: Label/display name of the agent being loaded.
        agent_domain: Domain of the agent.
        injection_success: Whether the manifest injection succeeded.
            None for ``manifest_injection_started`` events (outcome unknown).
        injection_duration_ms: Time to load and inject manifest (ms).
            None for ``manifest_injection_started`` events (not yet complete).
        routing_source: How the agent was selected.
        agent_version: Version of the agent definition if specified.
        yaml_path: Path to the agent YAML file (optional, for debugging).
        error_message: Error details if injection failed.
        error_type: Error classification if injection failed.

    Example:
        >>> from datetime import UTC, datetime
        >>> from uuid import uuid4
        >>> event = ModelManifestInjectionLifecycleEvent(
        ...     event_type="manifest_injected",
        ...     entity_id=uuid4(),
        ...     session_id=uuid4(),
        ...     correlation_id=uuid4(),
        ...     causation_id=uuid4(),
        ...     emitted_at=datetime(2026, 2, 27, 12, 0, 0, tzinfo=UTC),
        ...     agent_label="agent-api-architect",
        ...     agent_domain="api-development",
        ...     injection_success=True,
        ...     injection_duration_ms=45,
        ...     routing_source="explicit",
        ... )
    """

    model_config = ConfigDict(frozen=True, extra="ignore", from_attributes=True)

    event_type: Literal[
        "manifest_injection_started",
        "manifest_injected",
        "manifest_injection_failed",
    ] = Field(
        ...,
        description="Lifecycle stage discriminator",
    )

    # Entity identification
    entity_id: UUID = Field(
        ...,
        description="Session identifier as UUID (partition key for ordering)",
    )
    session_id: UUID = Field(
        ...,
        description="Session identifier as UUID",
    )

    # Tracing
    correlation_id: UUID = Field(
        default_factory=uuid4,
        description="Correlation ID for distributed tracing",
    )
    causation_id: UUID = Field(
        default_factory=uuid4,
        description="ID of the prompt event that triggered manifest injection",
    )

    # Timestamp
    emitted_at: datetime = Field(
        ...,
        description="Timestamp when the hook emitted this event (UTC)",
    )

    # Agent identification
    agent_label: str = Field(
        ...,
        min_length=1,
        description="Label/display name of the agent being loaded (e.g., 'agent-api-architect')",
    )
    agent_domain: str = Field(
        default="",
        description="Domain of the agent",
    )

    # Outcome (None for started events — outcome not yet known)
    injection_success: bool | None = Field(
        default=None,
        description="Whether the manifest injection succeeded. None for started events.",
    )
    injection_duration_ms: int | None = Field(
        default=None,
        ge=0,
        description="Time to load and inject manifest in milliseconds. None for started events.",
    )

    # Optional metadata
    routing_source: str | None = Field(
        default=None,
        description="How the agent was selected (explicit, fuzzy_match, fallback)",
    )
    agent_version: str | None = Field(
        default=None,
        description="Version of the agent definition if specified",
    )
    yaml_path: str | None = Field(
        default=None,
        description="Path to the agent YAML file (for debugging)",
    )

    # Error tracking (populated for manifest_injection_failed events)
    error_message: str | None = Field(
        default=None,
        description="Error details if injection failed",
    )
    error_type: str | None = Field(
        default=None,
        description="Error classification if injection failed",
    )
