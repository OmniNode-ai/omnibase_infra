# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Materialized dispatch message model.

This module defines the canonical runtime contract for dispatched messages.
All handlers receive a materialized dict that conforms to this shape.

The model serves dual purposes:
1. **Schema validation**: Ensures consistent message structure at dispatch time
2. **Documentation**: Explicitly defines the dispatch contract invariants

.. versionadded:: 0.2.6
    Added as part of OMN-1518 - Declarative operation bindings.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType


class ModelMaterializedDispatch(BaseModel):
    """Canonical dispatch message shape.

    This is the runtime contract for all handlers. After materialization,
    every handler receives a dict that conforms to this structure:

    - ``payload``: The original event payload (required)
    - ``__bindings``: Resolved binding parameters (always present, may be empty)
    - ``__debug_original_envelope``: Trace-only reference (do NOT use for business logic)

    The double-underscore prefix on ``__bindings`` and ``__debug_original_envelope``
    signals that these are infrastructure-level fields, not business data.

    Warning:
        ``__debug_original_envelope`` is provided ONLY for distributed tracing
        (accessing correlation_id, trace_id). It is NOT part of the handler's
        business contract and may be removed or changed without notice.

        **DO NOT**:
        - Branch business logic on ``__debug_original_envelope`` type
        - Access payload data through ``__debug_original_envelope``
        - Assume ``__debug_original_envelope`` will always be present

    Example:
        >>> materialized = {
        ...     "payload": {"user_id": "123", "action": "login"},
        ...     "__bindings": {"user_id": "123", "timestamp": "2025-01-27T12:00:00Z"},
        ...     "__debug_original_envelope": original_envelope,  # trace-only
        ... }
        >>> validated = ModelMaterializedDispatch.model_validate(materialized)
        >>> validated.payload
        {'user_id': '123', 'action': 'login'}

    Attributes:
        payload: The original event payload. Can be any object type - dict, list,
            Pydantic model, or primitive. This is typed as ``object`` to accept
            arbitrary payloads while maintaining type safety (per ONEX guidelines,
            ``object`` is preferred over ``Any``). At runtime, payloads are typically
            either dicts (JSON) or Pydantic models (domain events).
        bindings: Resolved binding parameters from contract.yaml operation_bindings.
            Always present (empty dict if no bindings configured). Handlers can
            access pre-resolved parameters without parsing expressions.
        debug_original_envelope: Reference to the original ModelEventEnvelope.
            Used ONLY for accessing trace metadata (correlation_id, trace_id).
            Excluded from repr() to prevent log bloat.

    .. versionadded:: 0.2.6
    """

    model_config = ConfigDict(
        frozen=True,
        populate_by_name=True,
        extra="forbid",
        arbitrary_types_allowed=True,  # Allow Pydantic models as payload
    )

    # NOTE: payload is typed as object (not JsonType) to allow arbitrary payloads
    # including Pydantic models (e.g., UserCreatedEvent), dicts, lists, etc.
    # Per ONEX guidelines, object is preferred over Any for "unknown type" semantics.
    payload: object = Field(
        ...,
        description="Original event payload. Can be dict, list, Pydantic model, or primitive.",
    )

    # NOTE: bindings values are typed as object (not JsonType) because resolved
    # bindings can include UUIDs, datetimes, and other non-JSON-primitive types.
    bindings: dict[str, object] = Field(
        default_factory=dict,
        alias="__bindings",
        description="Resolved binding parameters. Always present, may be empty dict.",
    )

    debug_original_envelope: object | None = Field(
        default=None,
        alias="__debug_original_envelope",
        description=(
            "Trace-only reference to original envelope. "
            "DO NOT use for business logic. May be removed without notice."
        ),
        repr=False,  # Prevent log bloat when stringifying model
    )
