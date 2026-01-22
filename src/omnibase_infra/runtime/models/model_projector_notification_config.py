# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Projector Notification Configuration Model.

Defines configuration for state transition notification publishing from projectors.
When configured, the projector will publish notifications to the event bus after
successful state transitions are committed.

This enables the Observer pattern for orchestrator coordination without tight
coupling between reducers and workflow coordinators.

Architecture Overview:
    1. ProjectorShell processes events and persists state changes
    2. Before commit, the previous state is fetched (if state tracking enabled)
    3. After successful commit, a notification is published with from_state/to_state
    4. Orchestrators subscribe to notifications and coordinate downstream workflows

Configuration Fields:
    - expected_topic: Expected event bus topic for validation/documentation (required).
        NOTE: The actual publishing topic is determined by TransitionNotificationPublisher.
        This field accepts "topic" as an alias for backwards compatibility.
    - state_column: Column name containing the FSM state (required)
    - aggregate_id_column: Column name containing the aggregate ID (required)
    - version_column: Column name containing the projection version (optional)
    - enabled: Whether notifications are enabled (default: True)

Example Usage:
    >>> from omnibase_infra.runtime.models import ModelProjectorNotificationConfig
    >>>
    >>> # Using the preferred field name
    >>> config = ModelProjectorNotificationConfig(
    ...     expected_topic="onex.fsm.state.transitions.v1",
    ...     state_column="current_state",
    ...     aggregate_id_column="entity_id",
    ...     version_column="version",
    ...     enabled=True,
    ... )
    >>>
    >>> # Using the backwards-compatible alias
    >>> config = ModelProjectorNotificationConfig(
    ...     topic="onex.fsm.state.transitions.v1",  # alias for expected_topic
    ...     state_column="current_state",
    ...     aggregate_id_column="entity_id",
    ... )

Related Tickets:
    - OMN-1139: Implement TransitionNotificationPublisher integration with ProjectorShell

Thread Safety:
    This model is immutable (frozen=True) after creation, making it thread-safe
    for concurrent read access.

.. versionadded:: 0.8.0
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelProjectorNotificationConfig(BaseModel):
    """Configuration for state transition notification publishing.

    When attached to a ProjectorShell via the notification_config parameter,
    enables automatic publishing of state transition notifications after
    successful projection commits.

    Attributes:
        expected_topic: Expected event bus topic for state transition notifications.
            This field is for documentation and validation purposes only - the actual
            publishing topic is determined by TransitionNotificationPublisher's
            configuration. ProjectorShell will log a warning if this value differs
            from the publisher's configured topic. Example topics:
            - "onex.fsm.state.transitions.v1"
            - "registration.state.transitions.v1"
            Accepts "topic" as an alias for backwards compatibility.
        state_column: Name of the column that contains the FSM state value.
            This column must exist in the projection schema and contain string
            values representing the current state.
        aggregate_id_column: Name of the column that contains the aggregate ID.
            This column must exist in the projection schema and typically contains
            a UUID identifying the aggregate instance.
        version_column: Optional name of the column that contains the projection
            version. If specified, the version value will be included in
            notifications for ordering and idempotency detection.
        enabled: Whether notification publishing is enabled. Defaults to True.
            Set to False to disable notifications without removing configuration.

    Example:
        >>> config = ModelProjectorNotificationConfig(
        ...     expected_topic="onex.fsm.state.transitions.v1",
        ...     state_column="current_state",
        ...     aggregate_id_column="entity_id",
        ...     version_column="version",
        ... )
        >>> config.expected_topic
        'onex.fsm.state.transitions.v1'
        >>> config.state_column
        'current_state'
        >>> config.enabled
        True

    Note:
        The column names specified must match columns defined in the projector's
        contract schema. The ProjectorShell will validate these column names
        against the schema at initialization time.

        Important: The expected_topic field does NOT control where notifications
        are published. The actual topic is determined by TransitionNotificationPublisher.
        This field exists so ProjectorShell can warn when there's a mismatch between
        the expected and actual topics, helping catch configuration errors early.

    See Also:
        - ProjectorShell: Uses this config for notification integration
        - TransitionNotificationPublisher: Publishes the notifications
        - ModelStateTransitionNotification: The notification payload model
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
        populate_by_name=True,  # Allow both "expected_topic" and "topic" (alias)
    )

    expected_topic: str = Field(
        ...,
        alias="topic",  # Backwards compatibility
        min_length=1,
        max_length=256,
        pattern=r"^[a-zA-Z][a-zA-Z0-9]*([._-][a-zA-Z0-9]+)*$",
        description=(
            "Expected event bus topic for state transition notifications. "
            "Must start with a letter (a-zA-Z), end with an alphanumeric character, "
            "and contain only alphanumeric characters with single dots, underscores, "
            "or hyphens as separators. Consecutive separators (e.g., '..', '--', '__', '.-') "
            "and trailing separators are not allowed. "
            "NOTE: This field is for documentation and validation purposes only. "
            "The actual publishing topic is determined by TransitionNotificationPublisher. "
            "ProjectorShell will warn if this value differs from the publisher's topic."
        ),
    )

    state_column: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Column name containing the FSM state value",
    )

    aggregate_id_column: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Column name containing the aggregate ID",
    )

    version_column: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Optional column name containing the projection version",
    )

    enabled: bool = Field(
        default=True,
        description="Whether notification publishing is enabled",
    )


__all__: list[str] = ["ModelProjectorNotificationConfig"]
