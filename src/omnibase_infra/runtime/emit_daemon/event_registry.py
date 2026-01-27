# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Event Registry - Event type to Kafka topic mapping for Hook Event Daemon.

This module provides the event registry that maps semantic event types
(e.g., "prompt.submitted") to Kafka topics and handles metadata injection.

The registry is the central configuration point for:
- Event type → topic routing
- Partition key extraction
- Payload validation
- Metadata injection (correlation IDs, timestamps, schema versions)

Example Usage:
    ```python
    from omnibase_infra.runtime.emit_daemon.event_registry import (
        EventRegistry,
        ModelEventRegistration,
    )

    # Create registry with environment prefix
    registry = EventRegistry(environment="dev")

    # Register a custom event type
    registry.register(
        ModelEventRegistration(
            event_type="custom.event",
            topic_template="{env}.onex.evt.custom.event.v1",
            partition_key_field="session_id",
            required_fields=["session_id", "user_id"],
        )
    )

    # Resolve topic for event type
    topic = registry.resolve_topic("prompt.submitted")
    # Returns: "dev.onex.evt.omniclaude.prompt-submitted.v1"

    # Inject metadata into payload
    enriched = registry.inject_metadata(
        event_type="prompt.submitted",
        payload={"prompt": "Hello", "session_id": "abc123"},
        correlation_id="corr-123",
    )
    # Returns payload with correlation_id, causation_id, emitted_at, schema_version
    ```

Integration Points:
- EmitDaemon uses this registry to route events to correct Kafka topics
- Hook events from OmniClaude are routed through this registry
- Metadata injection ensures event traceability across the system
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.errors import OnexError


class ModelEventRegistration(BaseModel):
    """Registration configuration for a single event type.

    Defines how a semantic event type (e.g., "prompt.submitted") maps to
    Kafka infrastructure including topic naming, partition keys, and
    payload validation rules.

    Attributes:
        event_type: Semantic event type identifier (e.g., "prompt.submitted").
            This is the logical name used by event emitters.
        topic_template: Kafka topic name template with {env} placeholder.
            Example: "{env}.onex.evt.omniclaude.prompt-submitted.v1"
        partition_key_field: Optional field name in payload to use as partition key.
            When set, ensures events with same key go to same partition for ordering.
        required_fields: List of field names that must be present in payload.
            Validation will fail if any required field is missing.
        schema_version: Semantic version of the event schema (default: "1.0.0").
            Injected into event metadata for schema evolution tracking.

    Example:
        >>> reg = ModelEventRegistration(
        ...     event_type="prompt.submitted",
        ...     topic_template="{env}.onex.evt.omniclaude.prompt-submitted.v1",
        ...     partition_key_field="session_id",
        ...     required_fields=["prompt", "session_id"],
        ...     schema_version="1.0.0",
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
    )

    event_type: str = Field(
        description="Semantic event type identifier (e.g., 'prompt.submitted')",
    )
    topic_template: str = Field(
        description="Kafka topic name template with {env} placeholder",
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

    The registry is initialized with default OmniClaude event types and can
    be extended with custom event registrations.

    Attributes:
        environment: Deployment environment name (e.g., "dev", "staging", "prod").
            Used to substitute {env} placeholder in topic templates.

    Example:
        >>> registry = EventRegistry(environment="dev")
        >>> topic = registry.resolve_topic("prompt.submitted")
        >>> print(topic)
        'dev.onex.evt.omniclaude.prompt-submitted.v1'

        >>> registry.validate_payload("prompt.submitted", {"prompt": "Hello"})
        True

        >>> enriched = registry.inject_metadata(
        ...     "prompt.submitted",
        ...     {"prompt": "Hello"},
        ... )
        >>> "correlation_id" in enriched
        True
    """

    def __init__(self, environment: str = "dev") -> None:
        """Initialize the event registry with environment prefix.

        Args:
            environment: Deployment environment name used to substitute
                {env} placeholder in topic templates. Defaults to "dev".

        Example:
            >>> registry = EventRegistry(environment="staging")
            >>> registry.resolve_topic("prompt.submitted")
            'staging.onex.evt.omniclaude.prompt-submitted.v1'
        """
        self._environment = environment
        self._registrations: dict[str, ModelEventRegistration] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default OmniClaude event types.

        Registers the standard event types emitted by OmniClaude hooks:
        - prompt.submitted: User prompt submission events
        - session.started: Session initialization events
        - session.ended: Session termination events
        - tool.executed: Tool execution events
        """
        defaults = [
            ModelEventRegistration(
                event_type="prompt.submitted",
                topic_template="{env}.onex.evt.omniclaude.prompt-submitted.v1",
                partition_key_field="session_id",
                required_fields=["prompt"],
            ),
            ModelEventRegistration(
                event_type="session.started",
                topic_template="{env}.onex.evt.omniclaude.session-started.v1",
                partition_key_field="session_id",
                required_fields=["session_id"],
            ),
            ModelEventRegistration(
                event_type="session.ended",
                topic_template="{env}.onex.evt.omniclaude.session-ended.v1",
                partition_key_field="session_id",
                required_fields=["session_id"],
            ),
            ModelEventRegistration(
                event_type="tool.executed",
                topic_template="{env}.onex.evt.omniclaude.tool-executed.v1",
                partition_key_field="session_id",
                required_fields=["tool_name"],
            ),
        ]
        for registration in defaults:
            self._registrations[registration.event_type] = registration

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
            ...         topic_template="{env}.onex.evt.custom.event.v1",
            ...     )
            ... )
            >>> registry.resolve_topic("custom.event")
            'dev.onex.evt.custom.event.v1'
        """
        self._registrations[registration.event_type] = registration

    def resolve_topic(self, event_type: str) -> str:
        """Get the Kafka topic for an event type.

        Resolves the topic template by substituting the {env} placeholder
        with the configured environment name.

        Args:
            event_type: Semantic event type identifier.

        Returns:
            Fully resolved Kafka topic name.

        Raises:
            OnexError: If the event type is not registered.

        Example:
            >>> registry = EventRegistry(environment="prod")
            >>> registry.resolve_topic("prompt.submitted")
            'prod.onex.evt.omniclaude.prompt-submitted.v1'
        """
        registration = self._registrations.get(event_type)
        if registration is None:
            registered = list(self._registrations.keys())
            raise OnexError(
                f"Unknown event type: '{event_type}'. Registered types: {registered}"
            )
        return registration.topic_template.format(env=self._environment)

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
            >>> key = registry.get_partition_key(
            ...     "prompt.submitted",
            ...     {"prompt": "Hello", "session_id": "sess-123"},
            ... )
            >>> print(key)
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
            >>> registry.validate_payload(
            ...     "prompt.submitted",
            ...     {"prompt": "Hello"},
            ... )
            True

            >>> registry.validate_payload(
            ...     "prompt.submitted",
            ...     {},
            ... )
            Traceback (most recent call last):
                ...
            OnexError: Missing required fields for 'prompt.submitted': ['prompt']
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
                back to their source. It should be populated when:

                - An event handler processes event A and emits event B as a result
                - A saga/workflow step emits a follow-up event
                - Any event is produced as a direct consequence of another event

                When None (the default), indicates this is a root event with no
                direct cause in the event stream (e.g., user-initiated actions,
                scheduled jobs, external triggers).

                Note: This is an extension point for future event chain tracing
                functionality. Current EmitDaemon usage passes None for all events
                since hook events are root events initiated by user actions.

        Returns:
            New dictionary with original payload plus injected metadata.

        Raises:
            OnexError: If the event type is not registered.

        Example:
            >>> registry = EventRegistry()
            >>> # Root event (no causation_id)
            >>> root_event = registry.inject_metadata(
            ...     "prompt.submitted",
            ...     {"prompt": "Hello"},
            ...     correlation_id="corr-123",
            ... )
            >>> root_event["causation_id"] is None
            True
            >>>
            >>> # Derived event (with causation_id linking to root)
            >>> derived_event = registry.inject_metadata(
            ...     "tool.executed",
            ...     {"tool_name": "search"},
            ...     correlation_id="corr-123",  # Same correlation for chain
            ...     causation_id=root_event["correlation_id"],  # Links to cause
            ... )
            >>> derived_event["causation_id"]
            'corr-123'
            >>> "emitted_at" in derived_event
            True
            >>> derived_event["schema_version"]
            '1.0.0'
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
            >>> reg = registry.get_registration("prompt.submitted")
            >>> reg.topic_template
            '{env}.onex.evt.omniclaude.prompt-submitted.v1'
        """
        return self._registrations.get(event_type)

    def list_event_types(self) -> list[str]:
        """List all registered event types.

        Returns:
            List of registered event type identifiers.

        Example:
            >>> registry = EventRegistry()
            >>> types = registry.list_event_types()
            >>> "prompt.submitted" in types
            True
        """
        return list(self._registrations.keys())


__all__: list[str] = [
    "EventRegistry",
    "ModelEventRegistration",
]
