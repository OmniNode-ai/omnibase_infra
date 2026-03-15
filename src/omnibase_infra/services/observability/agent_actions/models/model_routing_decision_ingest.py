# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# EXPIRY: 2026-04-01 or v0.6.0 (whichever first) — see OMN-3469 for retirement criteria
# BLOCKED ON: handler_routing_emitter.py (omniclaude) still emits old-shape fields
#   (session_id, confidence, emitted_at) — shim cannot be retired until Producer 1 is aligned
"""Permissive ingest model for the routing-decision Kafka topic.

Used exclusively at the Kafka boundary for onex.evt.omniclaude.routing-decision.v1.
Maps the omniclaude producer payload to internal field conventions without relaxing
the strict ModelRoutingDecision contract downstream.

Design Decisions:
    - extra="ignore": tolerates producer fields not declared here
    - Field aliases: map producer key names to internal names
    - Server-generated defaults: id (uuid4), created_at (from emitted_at else UTC now)
    - frozen=False: mutable at ingest boundary (immutability not required pre-persist)

Compatibility shim — see Known Debt in OMN-3422 for retirement criteria:
    1. Producer alignment: omniclaude emits contract-shaped fields directly
    2. Cross-repo schema handshake gate via CI
    3. Health check rule 5 redesign

Producers mapped:
    - handler_routing_emitter.py: correlation_id, session_id, selected_agent,
      confidence, emitted_at, confidence_breakdown, routing_policy, ...
    - route_via_events_wrapper.py: correlation_id, session_id, selected_agent,
      confidence, domain, reasoning, routing_method, latency_ms, ...
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_core.types import JsonType

logger = logging.getLogger(__name__)


class ModelRoutingDecisionIngest(BaseModel):
    """Permissive ingest model used exclusively at the Kafka boundary.

    Maps omniclaude producer payload to internal field conventions without
    relaxing the strict ModelRoutingDecision contract downstream.

    extra="ignore" tolerates producer fields not declared here (e.g.,
    confidence_breakdown, routing_policy, routing_path, prompt_preview,
    prompt_length, event_attempted).

    Field aliases map producer key names to internal names:
        - confidence → confidence_score
        - reasoning → routing_reason
        - session_id → claude_session_id

    Server-generated defaults:
        - id: uuid4() if absent from producer
        - created_at: emitted_at if present, else UTC now

    Attributes:
        id: Unique identifier (server-generated if absent from producer).
        created_at: Event timestamp (from emitted_at or UTC now if absent).
        correlation_id: Request correlation ID (always valid UUID from producers).
        selected_agent: Name of the agent selected.
        confidence_score: Confidence score (0.0-1.0), aliased from "confidence".
        routing_reason: Routing explanation, aliased from "reasoning".
        claude_session_id: Claude session ID, aliased from "session_id".
        domain: Domain classification for the request.
        routing_method: Routing method used (new field from producer 2).
        latency_ms: End-to-end routing duration (new field from producer 2).
        request_type: Type of request being routed (not populated by producers).
        alternatives: Alternative agents considered (not populated by producers).
        metadata: Additional metadata (not populated by producers).
        raw_payload: Complete raw payload (not populated by producers).
        project_path: Absolute path to the project (not populated by producers).
        project_name: Human-readable project name (not populated by producers).
    """

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
    )

    # Server-generated if absent from producer
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Required fields (same name in producer and internal model)
    # Invariant: correlation_id is always a valid UUID string from these producers.
    correlation_id: UUID
    selected_agent: str

    # Aliased: producer name -> internal name
    confidence_score: float = Field(alias="confidence", default=0.0)
    routing_reason: str | None = Field(alias="reasoning", default=None)
    claude_session_id: str | None = Field(alias="session_id", default=None)

    # Direct optional fields (same name in producer and internal model)
    domain: str | None = None
    routing_method: str | None = None
    latency_ms: int | None = None

    # Fields present in strict model; unset here (no producers populate them)
    request_type: str | None = None
    alternatives: tuple[str, ...] | None = None
    metadata: dict[str, JsonType] | None = None
    raw_payload: dict[str, JsonType] | None = None
    project_path: str | None = None
    project_name: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_timestamps(cls, data: object) -> object:
        """Map emitted_at -> created_at to preserve producer event time.

        Producer 1 (handler_routing_emitter.py) emits emitted_at but not created_at.
        This validator copies emitted_at to created_at before field validation runs,
        preserving the original event timestamp rather than defaulting to server time.

        The input dict is copied before mutation to avoid modifying the caller's object.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)  # copy; never mutate caller's dict
        if "emitted_at" in data and "created_at" not in data:
            data["created_at"] = data["emitted_at"]
        return data

    @field_validator("created_at", mode="after")
    @classmethod
    def _ensure_utc(cls, v: datetime) -> datetime:
        """Ensure created_at is timezone-aware UTC.

        Treats naive datetimes as UTC (producer 1 may emit naive ISO strings).
        """
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v.astimezone(UTC)

    @field_validator("confidence_score", mode="before")
    @classmethod
    def _normalize_confidence(cls, v: object) -> float:
        """Clamp confidence to [0.0, 1.0]; coerce non-numeric to 0.0."""
        try:
            f = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, f))

    def __str__(self) -> str:
        """Return concise string representation for logging."""
        id_short = str(self.id)[:8]
        domain_part = f", domain={self.domain}" if self.domain else ""
        return (
            f"RoutingDecisionIngest(id={id_short}, agent={self.selected_agent}, "
            f"confidence={self.confidence_score:.2f}{domain_part})"
        )


__all__ = ["ModelRoutingDecisionIngest"]
