# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the topic-migration projection handler (OMN-12623).

TDD: asserts the projection deterministically maps a lifecycle event to a
projection row and carries idempotency columns for replay-safe upserts.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_core.enums.enum_migration_phase import EnumMigrationPhase
from omnibase_infra.nodes.node_topic_migration_executor_effect.models.model_topic_migration_lifecycle_event import (
    ModelTopicMigrationLifecycleEvent,
)
from omnibase_infra.nodes.node_topic_migration_projection.handlers.handler_topic_migration_projection import (
    HandlerTopicMigrationProjection,
)

pytestmark = pytest.mark.unit


def _event(
    phase: EnumMigrationPhase, sequence: int
) -> ModelTopicMigrationLifecycleEvent:
    return ModelTopicMigrationLifecycleEvent(
        event_id=uuid4(),
        correlation_id=uuid4(),
        migration_ticket="OMN-12623",
        old_topic="onex.evt.orders.order-placed.v1",
        new_topic="onex.evt.orders.order-placed.v2",
        old_consumer_group="dev.orders.order-placed.consume.v1",
        new_consumer_group="dev.orders.order-placed.consume.v2",
        phase=phase,
        sequence=sequence,
    )


def test_projection_maps_event_to_row() -> None:
    handler = HandlerTopicMigrationProjection()
    event = _event(EnumMigrationPhase.CUTOVER, 3)
    row = handler.project(event, offset=42, partition="0")
    assert row.migration_ticket == "OMN-12623"
    assert row.current_state is EnumMigrationPhase.CUTOVER
    assert row.last_applied_event_id == event.event_id
    assert row.last_applied_sequence == 3
    assert row.last_applied_offset == 42
    assert row.last_applied_partition == "0"


def test_projection_is_deterministic() -> None:
    handler = HandlerTopicMigrationProjection()
    event = _event(EnumMigrationPhase.DUAL_WRITE, 1)
    row1 = handler.project(event, offset=1, partition="0")
    row2 = handler.project(event, offset=1, partition="0")
    assert row1 == row2
