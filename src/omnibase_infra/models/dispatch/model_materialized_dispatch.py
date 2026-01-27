# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Materialized dispatch message model.

This module defines the canonical runtime contract for dispatched messages.
All handlers receive a materialized dict that conforms to this shape.

Design Rationale:
    The dispatch boundary is a **serialization boundary**. This model enforces
    that all data crossing the dispatch layer is transport-safe:
    - JSON-serializable (can be logged, replayed, transported to Kafka)
    - Deterministic (same input produces same serialized output)
    - Inspectable (can be examined offline without Python runtime)

    Handlers that need rich Pydantic models should hydrate them locally:
        ``ModelFoo.model_validate(dispatch["payload"])``

    This separation keeps the runtime decoupled from handler internals and
    enables distributed dispatch, event replay, and observability tooling.

.. versionadded:: 0.2.6
    Added as part of OMN-1518 - Declarative operation bindings.

.. versionchanged:: 0.2.8
    Changed to strict JSON-safe contract:
    - Removed arbitrary_types_allowed
    - Changed payload/bindings to JsonType
    - Renamed __debug_original_envelope to __debug_trace (serialized snapshot)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType


class ModelMaterializedDispatch(BaseModel):
    """Canonical dispatch message shape (strictly JSON-safe).

    This is the runtime contract for all handlers. After materialization,
    every handler receives a dict that conforms to this structure:

    - ``payload``: The event payload as a JSON-safe dict (required)
    - ``__bindings``: Resolved binding parameters (always present, may be empty)
    - ``__debug_trace``: Serialized trace metadata snapshot (debug only)

    The double-underscore prefix on ``__bindings`` and ``__debug_trace``
    signals that these are infrastructure-level fields, not business data.

    Transport Safety:
        All fields are JSON-serializable. This enables:
        - Event replay from logs or Kafka
        - Distributed dispatch across processes
        - Observability tooling (logging, tracing, dashboards)
        - Offline inspection and debugging

    Handler Hydration:
        Handlers that need typed Pydantic models should hydrate locally:

        >>> payload = dispatch["payload"]
        >>> event = UserCreatedEvent.model_validate(payload)

        This keeps the dispatch boundary clean and transport-safe.

    Warning:
        ``__debug_trace`` is provided ONLY for debugging and observability.
        It is a serialized snapshot of trace metadata, NOT the live envelope.

        **DO NOT**:
        - Use ``__debug_trace`` for business logic
        - Assume ``__debug_trace`` reflects complete envelope state
        - Depend on specific fields being present

    Example:
        >>> materialized = {
        ...     "payload": {"user_id": "123", "action": "login"},
        ...     "__bindings": {"user_id": "123", "timestamp": "2025-01-27T12:00:00Z"},
        ...     "__debug_trace": {
        ...         "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
        ...         "topic": "dev.user.events.v1",
        ...     },
        ... }
        >>> validated = ModelMaterializedDispatch.model_validate(materialized)
        >>> validated.payload
        {'user_id': '123', 'action': 'login'}

    Attributes:
        payload: The event payload as a JSON-safe dict. Handlers that need
            typed models should call ``model_validate()`` on this dict.
            Primitives are wrapped under ``{"_raw": value}`` to maintain
            dict structure (this is a last-resort escape hatch).
        bindings: Resolved binding parameters from contract.yaml operation_bindings.
            Always present (empty dict if no bindings configured). All values
            are JSON-safe (UUIDs and datetimes are serialized to strings).
        debug_trace: Serialized trace metadata snapshot for debugging.
            Contains correlation_id, trace_id, topic, etc. as strings.
            This is NOT authoritative data and should NOT be used for
            business logic. Excluded from repr() to prevent log bloat.

    .. versionadded:: 0.2.6

    .. versionchanged:: 0.2.8
        Changed to strict JSON-safe contract. See module docstring.
    """

    model_config = ConfigDict(
        frozen=True,
        populate_by_name=True,
        extra="forbid",
        # NOTE: arbitrary_types_allowed is intentionally NOT set (defaults to False).
        # This enforces that all data crossing the dispatch boundary is JSON-safe.
        # See module docstring for design rationale.
    )

    payload: JsonType = Field(
        ...,
        description=(
            "Event payload as JSON-safe dict. "
            "Handlers should hydrate typed models via model_validate()."
        ),
    )

    bindings: dict[str, JsonType] = Field(
        default_factory=dict,
        alias="__bindings",
        description=(
            "Resolved binding parameters. Always present, may be empty dict. "
            "Values are JSON-safe (UUIDs/datetimes serialized to strings)."
        ),
    )

    debug_trace: dict[str, str | None] | None = Field(
        default=None,
        alias="__debug_trace",
        description=(
            "Serialized trace metadata snapshot for debugging only. "
            "NOT authoritative. Do NOT use for business logic."
        ),
        repr=False,  # Prevent log bloat when stringifying model
    )
