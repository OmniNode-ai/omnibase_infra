# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for topic-migration projection (OMN-12623).

Pure COMPUTE transform: maps a :class:`ModelTopicMigrationLifecycleEvent` to a
:class:`ModelTopicMigrationProjectionRow`. No I/O — deterministic and replay-safe.
The EFFECT side upserts the row keyed on ``migration_ticket`` using the
``last_applied_sequence`` guard to reject stale/duplicate events.
"""

from __future__ import annotations

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_topic_migration_executor_effect.models.model_topic_migration_lifecycle_event import (
    ModelTopicMigrationLifecycleEvent,
)
from omnibase_infra.nodes.node_topic_migration_projection.models.model_topic_migration_projection_row import (
    ModelTopicMigrationProjectionRow,
)


class HandlerTopicMigrationProjection:
    """Projects a migration lifecycle event into a projection row."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.PROJECTION_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    def project(
        self,
        event: ModelTopicMigrationLifecycleEvent,
        *,
        offset: int = 0,
        partition: str | None = None,
    ) -> ModelTopicMigrationProjectionRow:
        """Transform a lifecycle event into a materialized projection row.

        ``offset``/``partition`` carry the Kafka coordinates of the consumed
        event for the idempotency columns; ``sequence`` comes from the event.
        """
        return ModelTopicMigrationProjectionRow(
            migration_ticket=event.migration_ticket,
            old_topic=event.old_topic,
            new_topic=event.new_topic,
            old_consumer_group=event.old_consumer_group,
            new_consumer_group=event.new_consumer_group,
            current_state=event.phase,
            new_topic_provisioned=event.new_topic_provisioned,
            retirement_allowed=event.retirement_allowed,
            residual_lag=event.residual_lag,
            last_applied_event_id=event.event_id,
            last_applied_sequence=event.sequence,
            last_applied_offset=offset,
            last_applied_partition=partition,
        )


__all__ = ["HandlerTopicMigrationProjection"]
