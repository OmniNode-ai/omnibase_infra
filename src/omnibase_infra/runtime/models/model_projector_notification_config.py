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
    - state_column: Column name containing the FSM state (required)
    - aggregate_id_column: Column name containing the aggregate ID (required)
    - version_column: Column name containing the projection version (optional)
    - enabled: Whether notifications are enabled (default: True)

Example Usage:
    >>> from omnibase_infra.runtime.models import ModelProjectorNotificationConfig
    >>>
    >>> config = ModelProjectorNotificationConfig(
    ...     state_column="current_state",
    ...     aggregate_id_column="entity_id",
    ...     version_column="version",
    ...     enabled=True,
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
        ...     state_column="current_state",
        ...     aggregate_id_column="entity_id",
        ...     version_column="version",
        ... )
        >>> config.state_column
        'current_state'
        >>> config.enabled
        True

    Note:
        The column names specified must match columns defined in the projector's
        contract schema. The ProjectorShell will validate these column names
        against the schema at initialization time.

    See Also:
        - ProjectorShell: Uses this config for notification integration
        - TransitionNotificationPublisher: Publishes the notifications
        - ModelStateTransitionNotification: The notification payload model
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
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
