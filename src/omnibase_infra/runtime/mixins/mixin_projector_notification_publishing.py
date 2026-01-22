# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Notification Publishing Mixin for Projector Implementations.

Provides notification publishing capability for projector shells. This mixin
extracts notification logic from ProjectorShell to keep the main class focused
on projection logic and under the method count limit.

Features:
    - Pre-projection state fetching for from_state tracking
    - Post-commit notification creation and publishing
    - Correlation ID and causation ID propagation
    - Configurable via ModelProjectorNotificationConfig

Architecture:
    This mixin works with ProjectorShell to implement the Observer pattern:

    1. Before projection: Fetch current state from database (from_state)
    2. Execute projection (handled by ProjectorShell)
    3. After successful commit: Publish notification with from_state/to_state

    ```
    Event → ProjectorShell.project() → Database Commit
                   ↓                        ↓
         _fetch_current_state()     _publish_transition_notification()
                   ↓                        ↓
              from_state              Event Bus (to Orchestrators)
    ```

See Also:
    - ProjectorShell: Main projector class that uses this mixin
    - TransitionNotificationPublisher: Publishes notifications to event bus
    - ModelProjectorNotificationConfig: Configuration for notification behavior

Related Tickets:
    - OMN-1139: Integrate TransitionNotificationPublisher with ProjectorShell

.. versionadded:: 0.8.0
    Created as part of OMN-1139 notification integration.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol
from uuid import UUID

import asyncpg

from omnibase_core.models.notifications import ModelStateTransitionNotification
from omnibase_core.protocols.notifications import (
    ProtocolTransitionNotificationPublisher,
)
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
)
from omnibase_infra.models.projectors.util_sql_identifiers import quote_identifier
from omnibase_infra.runtime.constants_notification import FROM_STATE_INITIAL
from omnibase_infra.runtime.models.model_projector_notification_config import (
    ModelProjectorNotificationConfig,
)

if TYPE_CHECKING:
    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
    from omnibase_core.models.projectors import ModelProjectorContract

logger = logging.getLogger(__name__)


class ProtocolProjectorNotificationContext(Protocol):
    """Protocol for projector context required by notification publishing mixin.

    This protocol defines the minimum interface that a projector must
    implement to use MixinProjectorNotificationPublishing.
    """

    @property
    def _contract(self) -> ModelProjectorContract:
        """The projector contract defining projection behavior."""
        ...

    @property
    def _pool(self) -> asyncpg.Pool:
        """The asyncpg connection pool for database operations."""
        ...

    @property
    def _query_timeout(self) -> float:
        """Query timeout in seconds."""
        ...

    @property
    def projector_id(self) -> str:
        """Unique identifier for the projector."""
        ...

    @property
    def aggregate_type(self) -> str:
        """Aggregate type from contract."""
        ...


class MixinProjectorNotificationPublishing:
    """Notification publishing mixin for projector implementations.

    Provides methods for fetching previous state before projection and
    publishing state transition notifications after successful commits.

    This mixin expects the implementing class to provide:
    - ``_contract``: ModelProjectorContract instance
    - ``_pool``: asyncpg.Pool for database connections
    - ``_query_timeout``: float timeout in seconds
    - ``projector_id``: str identifier for logging
    - ``aggregate_type``: str aggregate type from contract

    Example:
        >>> class ProjectorShell(MixinProjectorNotificationPublishing, ...):
        ...     def __init__(
        ...         self,
        ...         contract,
        ...         pool,
        ...         notification_publisher=None,
        ...         notification_config=None,
        ...     ):
        ...         self._contract = contract
        ...         self._pool = pool
        ...         self._notification_publisher = notification_publisher
        ...         self._notification_config = notification_config
        ...
        ...     async def project(self, event, correlation_id):
        ...         # Fetch previous state if notifications enabled
        ...         from_state = await self._fetch_current_state_for_notification(
        ...             aggregate_id, correlation_id
        ...         )
        ...
        ...         # Execute projection...
        ...         result = await self._execute_projection(...)
        ...
        ...         # Publish notification if configured
        ...         if result.rows_affected > 0:
        ...             await self._publish_transition_notification(
        ...                 event, from_state, to_state, version, correlation_id
        ...             )
        ...         return result

    Warning:
        **Race Condition in from_state Tracking**: The ``from_state`` value is fetched
        in a separate query BEFORE the projection executes. There is a race window where
        another concurrent projection could modify the state between the fetch and the
        actual projection commit. In this scenario, the ``from_state`` in the notification
        may not reflect the true previous state.

        This is acceptable for most use cases because:

        1. Notifications are best-effort (failures are logged, not raised)
        2. Consumers should be idempotent and tolerate slight inconsistencies
        3. The ``to_state`` is always accurate (extracted from projection values)

        For use cases requiring truly accurate ``from_state`` tracking, consider:

        - **Database triggers**: Use PostgreSQL triggers to capture old row values atomically
        - **RETURNING clause**: Modify the projection SQL to use ``UPDATE ... RETURNING``
          with the old value (requires schema changes)
        - **Optimistic locking**: Add version checks to detect concurrent modifications

        See ``_fetch_current_state_for_notification`` for detailed race condition analysis.
    """

    # Type hints for expected attributes from implementing class
    _contract: ModelProjectorContract
    _pool: asyncpg.Pool
    _query_timeout: float
    _notification_publisher: ProtocolTransitionNotificationPublisher | None
    _notification_config: ModelProjectorNotificationConfig | None

    @property
    def projector_id(self) -> str:
        """Unique identifier for the projector (expected from implementing class)."""
        raise NotImplementedError("projector_id must be implemented by subclass")

    @property
    def aggregate_type(self) -> str:
        """Aggregate type from contract (expected from implementing class)."""
        raise NotImplementedError("aggregate_type must be implemented by subclass")

    def _is_notification_enabled(self) -> bool:
        """Check if notification publishing is enabled.

        Returns True only if:
        - notification_publisher is configured
        - notification_config is configured
        - notification_config.enabled is True

        Returns:
            True if notifications should be published, False otherwise.
        """
        return self._get_notification_context() is not None

    def _get_notification_context(
        self,
    ) -> (
        tuple[ProtocolTransitionNotificationPublisher, ModelProjectorNotificationConfig]
        | None
    ):
        """Return (publisher, config) tuple if notifications are enabled, None otherwise.

        This helper consolidates the enabled check with access to both publisher and config,
        reducing duplication in type-narrowing methods. The 3-way check (publisher not None,
        config not None, config.enabled is True) is performed once here.

        Returns:
            Tuple of (publisher, config) if notifications are enabled, None otherwise.
        """
        if (
            self._notification_publisher is not None
            and self._notification_config is not None
            and self._notification_config.enabled
        ):
            return (self._notification_publisher, self._notification_config)
        return None

    def _get_notification_config_if_enabled(
        self,
    ) -> ModelProjectorNotificationConfig | None:
        """Return notification config if notifications are enabled, None otherwise.

        This helper provides proper type narrowing for mypy without redundant None guards.
        Delegates to ``_get_notification_context()`` for the enabled check.

        Returns:
            The notification config if enabled, None otherwise.
        """
        context = self._get_notification_context()
        return context[1] if context else None

    def _get_notification_publisher_if_enabled(
        self,
    ) -> ProtocolTransitionNotificationPublisher | None:
        """Return notification publisher if notifications are enabled, None otherwise.

        This helper provides proper type narrowing for mypy without redundant None guards.
        Delegates to ``_get_notification_context()`` for the enabled check.

        Returns:
            The notification publisher if enabled, None otherwise.
        """
        context = self._get_notification_context()
        return context[0] if context else None

    async def _fetch_current_state_for_notification(
        self,
        aggregate_id: UUID,
        correlation_id: UUID,
    ) -> str | None:
        """Fetch the current state value for notification tracking.

        Queries the projection table to retrieve the current state value
        before a projection is executed. This becomes the from_state in
        the transition notification.

        Args:
            aggregate_id: The aggregate ID to look up.
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            The current state value as a string, or None if:
            - Notifications are not enabled
            - No row exists for this aggregate (new entity)
            - State column value is NULL

        Warning:
            **Race Condition Window**: This method fetches state BEFORE the projection
            executes, not atomically with it. The sequence is::

                T1: _fetch_current_state_for_notification() -> reads state="pending"
                T2: [Another projection commits] -> state changes to "approved"
                T1: _execute_projection() -> changes state to "completed"
                T1: Notification published with from_state="pending" (STALE!)

            In this scenario, the actual transition was "approved" -> "completed",
            but the notification reports "pending" -> "completed".

            **Impact Assessment**:

            - **Low impact**: Notifications are best-effort; consumers should be idempotent
            - **to_state is accurate**: Always extracted from projection values, not fetched
            - **Ordering preserved**: projection_version ensures consumers can order correctly

            **When This Matters**:

            - Audit logging requiring exact state history
            - UI showing real-time state transitions
            - Compliance systems requiring accurate audit trails

            **Alternatives for Strict Requirements**:

            1. **Database triggers** (recommended): Create a PostgreSQL trigger that fires
               BEFORE UPDATE and captures OLD.state into a session variable or audit table::

                   CREATE OR REPLACE FUNCTION capture_old_state()
                   RETURNS TRIGGER AS $$
                   BEGIN
                       PERFORM set_config('app.old_state', OLD.state, true);
                       RETURN NEW;
                   END;
                   $$ LANGUAGE plpgsql;

            2. **RETURNING clause**: Modify projection SQL to return old value using
               a subquery or CTE (requires significant projection refactoring)::

                   WITH old AS (SELECT state FROM table WHERE id = $1)
                   UPDATE table SET state = $2 WHERE id = $1
                   RETURNING (SELECT state FROM old) AS old_state

            3. **Serializable isolation**: Use SERIALIZABLE transaction isolation
               (significant performance impact, not recommended for high-throughput)

            **Design Decision**: This implementation prioritizes simplicity and performance
            over strict consistency. The race window is small (typically <10ms) and the
            impact is limited to from_state accuracy in notifications.
        """
        config = self._get_notification_config_if_enabled()
        if config is None:
            return None

        schema = self._contract.projection_schema
        table_quoted = quote_identifier(schema.table)
        state_col_quoted = quote_identifier(config.state_column)
        pk_col_quoted = quote_identifier(config.aggregate_id_column)

        # Build SELECT query - table/column names from trusted contract
        # S608: Safe - identifiers quoted via quote_identifier(), not user input
        query = (
            f"SELECT {state_col_quoted} FROM {table_quoted} WHERE {pk_col_quoted} = $1"  # noqa: S608
        )

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    query, aggregate_id, timeout=self._query_timeout
                )

            if row is None:
                logger.debug(
                    "No existing state found for aggregate (new entity)",
                    extra={
                        "projector_id": self.projector_id,
                        "aggregate_id": str(aggregate_id),
                        "correlation_id": str(correlation_id),
                    },
                )
                return None

            state_value = row[config.state_column]
            if state_value is not None:
                return str(state_value)
            return None

        except (TimeoutError, asyncpg.PostgresError) as e:
            # Log but don't fail - notifications are best-effort
            # Narrowed to DB and timeout errors to avoid masking configuration errors
            logger.warning(
                "Failed to fetch current state for notification: %s",
                str(e),
                extra={
                    "projector_id": self.projector_id,
                    "aggregate_id": str(aggregate_id),
                    "correlation_id": str(correlation_id),
                    "error_type": type(e).__name__,
                },
            )
            return None

    def _extract_state_from_values(
        self,
        values: dict[str, object],
    ) -> str | None:
        """Extract the state value from projection values.

        Args:
            values: Column name to value mapping from value extraction.

        Returns:
            The state value as a string, or None if not found.
        """
        config = self._get_notification_config_if_enabled()
        if config is None:
            return None

        state_value = values.get(config.state_column)
        if state_value is not None:
            return str(state_value)
        return None

    def _extract_aggregate_id_from_values(
        self,
        values: dict[str, object],
    ) -> UUID | None:
        """Extract the aggregate ID from projection values.

        Args:
            values: Column name to value mapping from value extraction.

        Returns:
            The aggregate ID as a UUID, or None if not found or invalid.
        """
        config = self._get_notification_config_if_enabled()
        if config is None:
            return None

        aggregate_id_value = values.get(config.aggregate_id_column)
        if aggregate_id_value is None:
            return None

        if isinstance(aggregate_id_value, UUID):
            return aggregate_id_value

        try:
            return UUID(str(aggregate_id_value))
        except (ValueError, TypeError):
            logger.warning(
                "Invalid aggregate ID value: %s",
                aggregate_id_value,
                extra={
                    "projector_id": self.projector_id,
                    "value_type": type(aggregate_id_value).__name__,
                },
            )
            return None

    def _extract_version_from_values(
        self,
        values: dict[str, object],
    ) -> int:
        """Extract the version from projection values.

        Args:
            values: Column name to value mapping from value extraction.

        Returns:
            The version as an integer. Returns 0 if:
            - Notifications are not enabled
            - No version column is configured
            - Version column value is missing or invalid
        """
        config = self._get_notification_config_if_enabled()
        if config is None:
            return 0

        if config.version_column is None:
            return 0

        version_value = values.get(config.version_column)
        if version_value is None:
            return 0

        if isinstance(version_value, int):
            return version_value

        try:
            # Convert to string first to satisfy type checker
            return int(str(version_value))
        except (ValueError, TypeError):
            return 0

    async def _publish_transition_notification(
        self,
        event: ModelEventEnvelope[object],
        from_state: str | None,
        to_state: str,
        projection_version: int,
        aggregate_id: UUID,
        correlation_id: UUID,
    ) -> None:
        """Publish a state transition notification after successful projection.

        Creates a ModelStateTransitionNotification and publishes it via the
        configured notification publisher. This method is best-effort - errors
        are logged but not raised.

        Args:
            event: The event envelope that triggered the projection.
            from_state: The previous state value, or None for new entities.
            to_state: The new state value after projection.
            projection_version: The projection version for ordering.
            aggregate_id: The aggregate instance ID.
            correlation_id: Correlation ID for distributed tracing.
        """
        publisher = self._get_notification_publisher_if_enabled()
        if publisher is None:
            return

        # Handle new entities (no previous state)
        # from_state is required in the notification model; use FROM_STATE_INITIAL sentinel
        # for new entities to clearly distinguish from empty string state values.
        # See constants_notification.py for full documentation on this sentinel.
        effective_from_state = (
            from_state if from_state is not None else FROM_STATE_INITIAL
        )

        # Create notification
        notification = ModelStateTransitionNotification(
            aggregate_type=self.aggregate_type,
            aggregate_id=aggregate_id,
            from_state=effective_from_state,
            to_state=to_state,
            projection_version=projection_version,
            correlation_id=correlation_id,
            causation_id=event.envelope_id,
            timestamp=datetime.now(UTC),
        )

        try:
            await publisher.publish(notification)

            logger.debug(
                "Published transition notification",
                extra={
                    "projector_id": self.projector_id,
                    "aggregate_type": self.aggregate_type,
                    "aggregate_id": str(aggregate_id),
                    "from_state": effective_from_state,
                    "to_state": to_state,
                    "projection_version": projection_version,
                    "correlation_id": str(correlation_id),
                },
            )

        except (InfraConnectionError, InfraTimeoutError, InfraUnavailableError) as e:
            # Log but don't fail - notifications are best-effort (expected failures)
            logger.warning(
                "Failed to publish transition notification: %s",
                str(e),
                extra={
                    "projector_id": self.projector_id,
                    "aggregate_type": self.aggregate_type,
                    "aggregate_id": str(aggregate_id),
                    "from_state": effective_from_state,
                    "to_state": to_state,
                    "correlation_id": str(correlation_id),
                    "error_type": type(e).__name__,
                },
            )
        except Exception as e:
            # Log unexpected errors at ERROR level for visibility
            logger.exception(
                "Unexpected error publishing transition notification",
                extra={
                    "projector_id": self.projector_id,
                    "aggregate_type": self.aggregate_type,
                    "aggregate_id": str(aggregate_id),
                    "from_state": effective_from_state,
                    "to_state": to_state,
                    "correlation_id": str(correlation_id),
                    "error_type": type(e).__name__,
                },
            )


__all__: list[str] = [
    "MixinProjectorNotificationPublishing",
    "ProtocolProjectorNotificationContext",
]
