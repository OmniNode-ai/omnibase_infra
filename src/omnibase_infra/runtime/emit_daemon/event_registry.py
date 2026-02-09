# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Event Registry - Event type to Kafka topic mapping.

This module provides a generic event registry that maps semantic event types
to Kafka topics and handles metadata injection.

The registry is the central configuration point for:
- Event type -> topic routing
- Partition key extraction
- Payload validation
- Metadata injection (correlation IDs, timestamps, schema versions)

The registry ships with no default registrations. Consumers must register
their own event types via ``register()`` or ``register_batch()``.

Example Usage:
    ```python
    from omnibase_infra.runtime.emit_daemon.event_registry import (
        EventRegistry,
        ModelEventRegistration,
    )

    # Create registry
    registry = EventRegistry(environment="dev")

    # Register event types
    registry.register_batch([
        ModelEventRegistration(
            event_type="myapp.submitted",
            topic_template="onex.evt.myapp.submitted.v1",
            partition_key_field="session_id",
            required_fields=["session_id", "payload"],
        ),
        ModelEventRegistration(
            event_type="myapp.completed",
            topic_template="onex.evt.myapp.completed.v1",
            partition_key_field="session_id",
            required_fields=["session_id"],
        ),
    ])

    # Resolve topic for event type (realm-agnostic, no env prefix)
    topic = registry.resolve_topic("myapp.submitted")
    # Returns: "onex.evt.myapp.submitted.v1"

    # Inject metadata into payload
    enriched = registry.inject_metadata(
        event_type="myapp.submitted",
        payload={"session_id": "abc123", "payload": "data"},
        correlation_id="corr-123",
    )
    # Returns payload with correlation_id, causation_id, emitted_at, schema_version
    ```

Note:
    Topics are realm-agnostic in ONEX. The environment/realm is enforced via
    envelope identity, not topic naming.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.errors import OnexError


class ModelEventRegistration(BaseModel):
    """Registration configuration for a single event type.

    Defines how a semantic event type maps to Kafka infrastructure including
    topic naming, partition keys, and payload validation rules.

    Attributes:
        event_type: Semantic event type identifier (e.g., "myapp.submitted").
            This is the logical name used by event emitters.
        topic_template: Kafka topic name (realm-agnostic, no environment prefix).
            Example: "onex.evt.myapp.submitted.v1"
            Note: Topics are realm-agnostic in ONEX. The environment/realm is
            enforced via envelope identity, not topic naming.
        partition_key_field: Optional field name in payload to use as partition key.
            When set, ensures events with same key go to same partition for ordering.
        required_fields: List of field names that must be present in payload.
            Validation will fail if any required field is missing.
        schema_version: Semantic version of the event schema (default: "1.0.0").
            Injected into event metadata for schema evolution tracking.

    Example:
        >>> reg = ModelEventRegistration(
        ...     event_type="myapp.submitted",
        ...     topic_template="onex.evt.myapp.submitted.v1",
        ...     partition_key_field="session_id",
        ...     required_fields=["session_id", "payload"],
        ...     schema_version="1.0.0",
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
    )

    event_type: str = Field(
        description="Semantic event type identifier (e.g., 'myapp.submitted')",
    )
    topic_template: str = Field(
        description="Kafka topic name (realm-agnostic, no environment prefix)",
    )
    partition_key_field: str | None = Field(
        default=None,
        description="Optional field name in payload to use as partition key",
    )
    required_fields: list[str] = Field(
        default_factory=list,
        description="List of field names that must be present in payload",
    )
    schema_version: str = Field(
        default="1.0.0",
        description="Semantic version of the event schema",
    )


class EventRegistry:
    """Registry for event type to Kafka topic mappings.

    Manages the mapping between semantic event types and Kafka infrastructure,
    including topic resolution, partition key extraction, payload validation,
    and metadata injection.

    The registry starts empty. Consumers must register their own event types
    via ``register()`` or ``register_batch()``.

    Attributes:
        environment: Deployment environment name (e.g., "dev", "staging", "prod").
            Stored for potential use in consumer group derivation by related
            components.

    Example:
        >>> registry = EventRegistry(environment="dev")
        >>> registry.register(
        ...     ModelEventRegistration(
        ...         event_type="myapp.submitted",
        ...         topic_template="onex.evt.myapp.submitted.v1",
        ...         required_fields=["payload"],
        ...     )
        ... )
        >>> registry.resolve_topic("myapp.submitted")
        'onex.evt.myapp.submitted.v1'

    Note:
        Topics are realm-agnostic in ONEX. The environment is stored for
        potential use in consumer group derivation by related components,
        but topics themselves do not include environment prefixes.
    """

    def __init__(self, environment: str = "dev") -> None:
        """Initialize an empty event registry.

        Args:
            environment: Deployment environment name. Stored for potential
                use in consumer group derivation. Defaults to "dev".

        Note:
            The registry starts with no registrations. Use ``register()``
            or ``register_batch()`` to add event type mappings.
        """
        self._environment = environment
        self._registrations: dict[str, ModelEventRegistration] = {}

    def register(self, registration: ModelEventRegistration) -> None:
        """Register an event type mapping.

        Adds or updates a registration for the given event type.
        Existing registrations for the same event type are overwritten.

        Args:
            registration: Event registration configuration.

        Example:
            >>> registry = EventRegistry()
            >>> registry.register(
            ...     ModelEventRegistration(
            ...         event_type="custom.event",
            ...         topic_template="onex.evt.custom.event.v1",
            ...     )
            ... )
            >>> registry.resolve_topic("custom.event")
            'onex.evt.custom.event.v1'
        """
        self._registrations[registration.event_type] = registration

    def register_batch(self, registrations: Iterable[ModelEventRegistration]) -> None:
        """Register multiple event type mappings.

        Convenience method to register multiple event types in a single call.
        Each registration is added via ``register()``, overwriting existing
        registrations for the same event type.

        Args:
            registrations: Iterable of event registration configurations.

        Example:
            >>> registry = EventRegistry()
            >>> registry.register_batch([
            ...     ModelEventRegistration(
            ...         event_type="custom.one",
            ...         topic_template="onex.evt.custom.one.v1",
            ...     ),
            ...     ModelEventRegistration(
            ...         event_type="custom.two",
            ...         topic_template="onex.evt.custom.two.v1",
            ...     ),
            ... ])
            >>> registry.resolve_topic("custom.one")
            'onex.evt.custom.one.v1'
        """
        for registration in registrations:
            self.register(registration)

    def resolve_topic(self, event_type: str) -> str:
        """Get the Kafka topic for an event type (realm-agnostic).

        Topics are realm-agnostic in ONEX. The environment/realm is enforced via
        envelope identity, not topic naming. This enables cross-environment event
        routing when needed while maintaining proper isolation through identity.

        Args:
            event_type: Semantic event type identifier.

        Returns:
            Kafka topic name (no environment prefix).

        Raises:
            OnexError: If the event type is not registered.

        Example:
            >>> registry = EventRegistry()
            >>> registry.register(
            ...     ModelEventRegistration(
            ...         event_type="myapp.submitted",
            ...         topic_template="onex.evt.myapp.submitted.v1",
            ...     )
            ... )
            >>> registry.resolve_topic("myapp.submitted")
            'onex.evt.myapp.submitted.v1'
        """
        registration = self._registrations.get(event_type)
        if registration is None:
            registered = list(self._registrations.keys())
            raise OnexError(
                f"Unknown event type: '{event_type}'. Registered types: {registered}"
            )
        return registration.topic_template

    def get_partition_key(
        self,
        event_type: str,
        payload: dict[str, object],
    ) -> str | None:
        """Extract partition key from payload based on registration.

        Uses the configured partition_key_field to extract the value
        from the payload. Returns None if no partition key is configured
        or the field is not present in the payload.

        Args:
            event_type: Semantic event type identifier.
            payload: Event payload dictionary.

        Returns:
            Partition key value as string, or None if not applicable.

        Raises:
            OnexError: If the event type is not registered.

        Example:
            >>> registry = EventRegistry()
            >>> registry.register(
            ...     ModelEventRegistration(
            ...         event_type="myapp.submitted",
            ...         topic_template="onex.evt.myapp.submitted.v1",
            ...         partition_key_field="session_id",
            ...     )
            ... )
            >>> registry.get_partition_key(
            ...     "myapp.submitted",
            ...     {"session_id": "sess-123"},
            ... )
            'sess-123'
        """
        registration = self._registrations.get(event_type)
        if registration is None:
            registered = list(self._registrations.keys())
            raise OnexError(
                f"Unknown event type: '{event_type}'. Registered types: {registered}"
            )

        if registration.partition_key_field is None:
            return None

        value = payload.get(registration.partition_key_field)
        if value is None:
            return None

        return str(value)

    def validate_payload(
        self,
        event_type: str,
        payload: dict[str, object],
    ) -> bool:
        """Validate payload has all required fields.

        Checks that all fields specified in the registration's required_fields
        are present in the payload.

        Args:
            event_type: Semantic event type identifier.
            payload: Event payload dictionary to validate.

        Returns:
            True if validation passes.

        Raises:
            OnexError: If the event type is not registered or if any
                required field is missing from the payload.

        Example:
            >>> registry = EventRegistry()
            >>> registry.register(
            ...     ModelEventRegistration(
            ...         event_type="myapp.submitted",
            ...         topic_template="onex.evt.myapp.submitted.v1",
            ...         required_fields=["payload"],
            ...     )
            ... )
            >>> registry.validate_payload("myapp.submitted", {"payload": "data"})
            True
        """
        registration = self._registrations.get(event_type)
        if registration is None:
            registered = list(self._registrations.keys())
            raise OnexError(
                f"Unknown event type: '{event_type}'. Registered types: {registered}"
            )

        missing_fields = [
            field for field in registration.required_fields if field not in payload
        ]

        if missing_fields:
            raise OnexError(
                f"Missing required fields for '{event_type}': {missing_fields}"
            )

        return True

    def inject_metadata(
        self,
        event_type: str,
        payload: dict[str, object],
        correlation_id: str | None = None,
        causation_id: str | None = None,
    ) -> dict[str, object]:
        """Inject correlation_id, causation_id, emitted_at, and schema_version.

        Creates a new payload dictionary with metadata fields added.
        The original payload is not modified.

        Injected fields:
        - correlation_id: Trace ID for the event chain (auto-generated if None)
        - causation_id: ID of the event that caused this event (None if root event)
        - emitted_at: ISO-8601 timestamp of when the event was emitted
        - schema_version: Version of the event schema from registration

        Args:
            event_type: Semantic event type identifier.
            payload: Event payload dictionary to enrich.
            correlation_id: Optional correlation ID for tracing. If None,
                a new UUID will be generated.
            causation_id: Optional ID of the event that directly caused this event.
                This parameter enables event chain tracing by linking derived events
                back to their source. When None (the default), indicates this is a
                root event with no direct cause in the event stream.

        Returns:
            New dictionary with original payload plus injected metadata.

        Raises:
            OnexError: If the event type is not registered.

        Example:
            >>> registry = EventRegistry()
            >>> registry.register(
            ...     ModelEventRegistration(
            ...         event_type="myapp.submitted",
            ...         topic_template="onex.evt.myapp.submitted.v1",
            ...     )
            ... )
            >>> enriched = registry.inject_metadata(
            ...     "myapp.submitted",
            ...     {"data": "value"},
            ...     correlation_id="corr-123",
            ... )
            >>> enriched["correlation_id"]
            'corr-123'
            >>> enriched["causation_id"] is None
            True
        """
        registration = self._registrations.get(event_type)
        if registration is None:
            registered = list(self._registrations.keys())
            raise OnexError(
                f"Unknown event type: '{event_type}'. Registered types: {registered}"
            )

        # Create new dict with original payload
        enriched: dict[str, object] = dict(payload)

        # Inject metadata
        enriched["correlation_id"] = correlation_id or str(uuid4())
        enriched["causation_id"] = causation_id
        enriched["emitted_at"] = datetime.now(UTC).isoformat()
        enriched["schema_version"] = registration.schema_version

        return enriched

    def get_registration(self, event_type: str) -> ModelEventRegistration | None:
        """Get the registration for an event type.

        Args:
            event_type: Semantic event type identifier.

        Returns:
            The registration configuration, or None if not registered.

        Example:
            >>> registry = EventRegistry()
            >>> registry.register(
            ...     ModelEventRegistration(
            ...         event_type="myapp.submitted",
            ...         topic_template="onex.evt.myapp.submitted.v1",
            ...         partition_key_field="session_id",
            ...     )
            ... )
            >>> reg = registry.get_registration("myapp.submitted")
            >>> reg.topic_template
            'onex.evt.myapp.submitted.v1'
        """
        return self._registrations.get(event_type)

    def list_event_types(self) -> list[str]:
        """List all registered event types.

        Returns:
            List of registered event type identifiers.

        Example:
            >>> registry = EventRegistry()
            >>> registry.register(
            ...     ModelEventRegistration(
            ...         event_type="myapp.submitted",
            ...         topic_template="onex.evt.myapp.submitted.v1",
            ...     )
            ... )
            >>> "myapp.submitted" in registry.list_event_types()
            True
        """
        return list(self._registrations.keys())


__all__: list[str] = [
    "EventRegistry",
    "ModelEventRegistration",
]
