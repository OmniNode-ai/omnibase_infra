# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Envelope validation for runtime dispatch.

This module provides minimal envelope validation before dispatch to handlers.
Validation enforces ONLY:
    1. Operation presence and non-empty
    2. Prefix validity against registry (unknown prefix = error)
    3. Correlation ID normalization to UUID
    4. Payload existence when required by operation type

Validation MUST NOT inspect handler-specific payload schemas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    EnvelopeValidationError,
    ModelInfraErrorContext,
    UnknownHandlerTypeError,
)

if TYPE_CHECKING:
    from omnibase_infra.runtime.handler_registry import RegistryProtocolBinding


def normalize_correlation_id(raw_value: object) -> UUID:
    """Normalize a correlation ID value to a UUID.

    Handles multiple input types and ensures a valid UUID is always returned:
    - If already a UUID, returns it unchanged
    - If a valid UUID string, parses and returns the UUID
    - If invalid/missing/wrong type, generates a new UUID

    This is the single source of truth for correlation ID normalization
    across the runtime infrastructure.

    Args:
        raw_value: The raw correlation_id value from an envelope.
            Can be UUID, str, None, or any other type.

    Returns:
        A valid UUID - either the normalized input or a newly generated one.

    Example:
        >>> from uuid import UUID
        >>> normalize_correlation_id(None)  # Returns new UUID
        UUID('...')
        >>> normalize_correlation_id("invalid")  # Returns new UUID
        UUID('...')
        >>> existing = UUID("12345678-1234-5678-1234-567812345678")
        >>> normalize_correlation_id(existing) == existing
        True
        >>> normalize_correlation_id(str(existing)) == existing
        True
    """
    if isinstance(raw_value, UUID):
        return raw_value
    if isinstance(raw_value, str):
        try:
            return UUID(raw_value)
        except ValueError:
            # Invalid UUID string - generate new one
            return uuid4()
    # None or unknown type - generate new one
    return uuid4()


# Operations that REQUIRE payload to be present and non-empty
PAYLOAD_REQUIRED_OPERATIONS: frozenset[str] = frozenset(
    {
        # Database operations
        "db.query",
        "db.execute",
        # HTTP operations with body
        "http.post",
        "http.put",
        "http.patch",
        # Kafka operations
        "kafka.produce",
        # Consul operations
        "consul.kv_put",
        "consul.register",
        # Vault operations
        "vault.write",
        "vault.encrypt",
        "vault.decrypt",
    }
)


def validate_envelope(
    envelope: dict[str, object],
    registry: RegistryProtocolBinding,
) -> None:
    """Validate envelope before dispatch to handler.

    Performs minimal validation required for safe dispatch:
    1. Operation presence and type validation
    2. Handler prefix validation against registry
    3. Payload requirement validation for specific operations
    4. Correlation ID normalization

    This function MUTATES the envelope dict to normalize correlation_id.

    Args:
        envelope: Envelope dict to validate. Will be mutated to normalize
            correlation_id to UUID.
        registry: Handler registry to validate operation prefix against.

    Raises:
        EnvelopeValidationError: If validation fails (missing operation,
            missing required payload).
        UnknownHandlerTypeError: If operation prefix is not registered
            in the handler registry.

    Example:
        >>> from omnibase_infra.runtime.handler_registry import get_handler_registry
        >>> envelope = {"operation": "db.query", "payload": {"sql": "SELECT 1"}}
        >>> validate_envelope(envelope, get_handler_registry())
        >>> # envelope["correlation_id"] is now a UUID
    """
    # 1. Validate operation presence and type
    operation = envelope.get("operation")
    if not operation or not isinstance(operation, str):
        raise EnvelopeValidationError(
            "operation is required and must be non-empty string",
            context=ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="validate_envelope",
            ),
            received_type=type(operation).__name__ if operation is not None else "None",
        )

    # 2. Validate operation prefix against registry
    prefix = operation.split(".")[0]
    if not registry.is_registered(prefix):
        registered = registry.list_protocols()
        raise UnknownHandlerTypeError(
            f"No handler registered for prefix: {prefix!r}",
            context=ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="validate_envelope",
                target_name=prefix,
            ),
            handler_prefix=prefix,
            operation=operation,
            registered_prefixes=registered,
        )

    # 3. Validate payload for operations that require it
    if operation in PAYLOAD_REQUIRED_OPERATIONS:
        payload = envelope.get("payload")
        if payload is None or (isinstance(payload, dict) and not payload):
            raise EnvelopeValidationError(
                f"payload is required for operation: {operation}",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="validate_envelope",
                    target_name=operation,
                ),
                operation=operation,
            )

    # 4. Normalize correlation_id to UUID (MUTATES envelope)
    envelope["correlation_id"] = normalize_correlation_id(
        envelope.get("correlation_id")
    )


__all__: list[str] = [
    "PAYLOAD_REQUIRED_OPERATIONS",
    "normalize_correlation_id",
    "validate_envelope",
]
