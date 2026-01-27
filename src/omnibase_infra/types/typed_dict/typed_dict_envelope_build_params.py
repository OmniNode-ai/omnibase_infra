# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""TypedDict definition for envelope build parameters.

This module provides a TypedDict that represents the parameters passed to the
_build_envelope method in AdapterProtocolEventPublisherInmemory, replacing the
loose dict[str, object] typing with proper type safety.

The TypedDictEnvelopeBuildParams ensures type-checked access to envelope
construction parameters without requiring type: ignore comments or unsafe casts.

Key Features:
    - Full type annotations for all envelope build parameters
    - Proper handling of nullable fields (correlation_id, causation_id, metadata)
    - Integration with JsonType for payload typing
    - Uses TYPE_CHECKING import for ContextValue to avoid circular imports

Usage:
    This TypedDict is primarily used for:
    - Type-safe parameter passing to envelope builder methods
    - Eliminating dict[str, object] loose typing
    - Enabling mypy verification of parameter access

Example:
    ```python
    from omnibase_infra.types.typed_dict import TypedDictEnvelopeBuildParams

    def build_envelope(params: TypedDictEnvelopeBuildParams) -> ModelEventEnvelope:
        # Type-safe access to all fields
        event_type = params["event_type"]
        payload = params["payload"]
        correlation_id = params.get("correlation_id")
        return ...
    ```

See Also:
    - AdapterProtocolEventPublisherInmemory: Primary consumer of this TypedDict
    - ModelEventEnvelope: Target envelope model constructed from these params
    - ProtocolEventPublisher: SPI protocol defining publish interface
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NotRequired, TypedDict

from omnibase_core.types import JsonType

if TYPE_CHECKING:
    from omnibase_spi.protocols.types.protocol_core_types import ContextValue

__all__ = ["TypedDictEnvelopeBuildParams"]


class TypedDictEnvelopeBuildParams(TypedDict):
    """TypedDict representing parameters for building a ModelEventEnvelope.

    This type provides a type-safe alternative to dict[str, object] for passing
    envelope construction parameters. It enables proper type checking when
    building event envelopes in publisher adapters.

    Required fields (event_type, payload) are enforced by TypedDict's default
    total=True behavior. Optional fields use NotRequired[] to allow omission
    while maintaining type safety.

    Attributes:
        event_type: Fully-qualified event type identifier
            (e.g., "omninode.user.event.created.v1"). Required field - must
            always be provided when constructing the TypedDict.
        payload: Event payload data as JSON-compatible types. Required field -
            must always be provided. Can be dict, list, str, int, float, bool,
            or None.
        correlation_id: Optional correlation ID for request tracing.
            Used to link related events across service boundaries.
            May be omitted or set to None.
        causation_id: Optional causation ID for event sourcing chains.
            Links this event to the event that caused it.
            May be omitted or set to None.
        metadata: Optional additional context values for the envelope.
            Keys are string identifiers, values are ContextValue protocol
            implementations (e.g., ProtocolContextStringValue).
            May be omitted or set to None.

    Note:
        The metadata field uses ContextValue from omnibase_spi which is imported
        under TYPE_CHECKING to avoid runtime circular import issues. At runtime,
        the dict values are duck-typed protocol implementations.

    Example:
        ```python
        # Minimal valid TypedDict (only required fields)
        params: TypedDictEnvelopeBuildParams = {
            "event_type": "user.created.v1",
            "payload": {"user_id": "123", "email": "user@example.com"},
        }

        # Full TypedDict with all fields
        params_full: TypedDictEnvelopeBuildParams = {
            "event_type": "user.created.v1",
            "payload": {"user_id": "123", "email": "user@example.com"},
            "correlation_id": "corr-abc-123",
            "causation_id": None,
            "metadata": {"trace_id": trace_value},
        }

        # Safe field access
        event_type: str = params["event_type"]  # Always present
        correlation_id: str | None = params.get("correlation_id")  # May be absent
        ```
    """

    event_type: str
    payload: JsonType
    correlation_id: NotRequired[str | None]
    causation_id: NotRequired[str | None]
    metadata: NotRequired[dict[str, ContextValue] | None]
