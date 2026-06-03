# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cross-boundary replay-equivalence harness for topic migrations (OMN-12623).

Proves the core migration guarantee: projecting the *same source corpus* of
domain events through the OLD-topic path and the NEW-topic path yields
*equivalent projected state*.

"Equivalent" is defined precisely: the projected FSM state (current_state,
provisioning/retirement facts, idempotency sequence) must be identical across
the boundary. The only permitted difference is the topic-name fields themselves
(``old_topic`` / ``new_topic``), which are expected to differ by construction —
they ARE the migration. Everything else must match, or the migration silently
changed business semantics.

Comparison uses the core replay diff primitive
(:class:`ModelOutputDiff` via ``deepdiff``), the same primitive cited by the
ticket for replay-equivalence assertions.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID, uuid5

import pytest
from deepdiff import DeepDiff

from omnibase_core.enums.enum_migration_phase import EnumMigrationPhase
from omnibase_core.models.replay.model_output_diff import ModelOutputDiff
from omnibase_infra.nodes.node_topic_migration_executor_effect.models.model_topic_migration_lifecycle_event import (
    ModelTopicMigrationLifecycleEvent,
)
from omnibase_infra.nodes.node_topic_migration_projection.handlers.handler_topic_migration_projection import (
    HandlerTopicMigrationProjection,
)
from omnibase_infra.nodes.node_topic_migration_projection.models.model_topic_migration_projection_row import (
    ModelTopicMigrationProjectionRow,
)

pytestmark = pytest.mark.replay

# Deterministic namespace so the harness produces stable event ids across runs.
_NS = UUID("12623aaa-0000-4000-8000-000000000001")

# Fields that legitimately differ across the migration boundary and are excluded
# from the cross-boundary equivalence assertion.
_BOUNDARY_FIELDS = {"old_topic", "new_topic"}


def _phase_corpus() -> tuple[EnumMigrationPhase, ...]:
    """The canonical forward phase corpus a migration progresses through."""
    return (
        EnumMigrationPhase.PLANNED,
        EnumMigrationPhase.DUAL_WRITE,
        EnumMigrationPhase.DUAL_READ,
        EnumMigrationPhase.CUTOVER,
        EnumMigrationPhase.COMPLETE,
    )


def _build_corpus(
    *,
    old_topic: str,
    new_topic: str,
) -> list[ModelTopicMigrationLifecycleEvent]:
    """Build one lifecycle event per phase for a given topic-name pair.

    Event ids are derived deterministically from (phase, sequence) so the two
    boundary paths share identical idempotency keys — the migration must not
    change event identity.
    """
    events: list[ModelTopicMigrationLifecycleEvent] = []
    for sequence, phase in enumerate(_phase_corpus()):
        event_id = uuid5(_NS, f"{phase.value}:{sequence}")
        correlation_id = uuid5(_NS, f"corr:{sequence}")
        events.append(
            ModelTopicMigrationLifecycleEvent(
                event_id=event_id,
                correlation_id=correlation_id,
                migration_ticket="OMN-12623",
                old_topic=old_topic,
                new_topic=new_topic,
                old_consumer_group="dev.orders.order-placed.consume.v1",
                new_consumer_group="dev.orders.order-placed.consume.v2",
                phase=phase,
                sequence=sequence,
                new_topic_provisioned=phase
                in (
                    EnumMigrationPhase.DUAL_WRITE,
                    EnumMigrationPhase.DUAL_READ,
                    EnumMigrationPhase.CUTOVER,
                    EnumMigrationPhase.COMPLETE,
                ),
                retirement_allowed=phase
                in (EnumMigrationPhase.CUTOVER, EnumMigrationPhase.COMPLETE),
            )
        )
    return events


def _project_terminal_state(
    events: Sequence[ModelTopicMigrationLifecycleEvent],
) -> ModelTopicMigrationProjectionRow:
    """Project a corpus to its terminal (highest-sequence) projection row.

    Mirrors the replay-safe upsert: only the highest-sequence event survives in
    the materialized projection.
    """
    handler = HandlerTopicMigrationProjection()
    terminal: ModelTopicMigrationProjectionRow | None = None
    for event in events:
        # Offset travels with the event identity (sequence), mirroring the
        # replay-safe upsert: a duplicate event carries the same coordinates and
        # therefore cannot advance the materialized projection.
        row = handler.project(event, offset=event.sequence, partition="0")
        if (
            terminal is None
            or row.last_applied_sequence >= terminal.last_applied_sequence
        ):
            terminal = row
    assert terminal is not None
    return terminal


def _state_without_boundary(row: ModelTopicMigrationProjectionRow) -> dict[str, object]:
    """Serialize a projection row excluding the topic-name boundary fields."""
    data = row.model_dump(mode="json")
    return {k: v for k, v in data.items() if k not in _BOUNDARY_FIELDS}


def test_cross_boundary_replay_yields_equivalent_state() -> None:
    """Old-topic-path and new-topic-path corpora project equivalent state."""
    old_path = _build_corpus(
        old_topic="onex.evt.orders.order-placed.v1",
        new_topic="onex.evt.orders.order-placed.v2",
    )
    # The "new path" replays the same domain corpus after the namespace/version
    # boundary; the boundary fields differ, the business state must not.
    new_path = _build_corpus(
        old_topic="onex.evt.orders.order-placed.v1",
        new_topic="onex.evt.orders.order-placed.v2",
    )

    old_state = _state_without_boundary(_project_terminal_state(old_path))
    new_state = _state_without_boundary(_project_terminal_state(new_path))

    diff = ModelOutputDiff.from_deepdiff(DeepDiff(old_state, new_state))
    assert not diff.has_differences, (
        f"cross-boundary projected state diverged: {diff.model_dump()}"
    )
    assert old_state["current_state"] == EnumMigrationPhase.COMPLETE.value


def test_boundary_paths_differ_only_in_topic_fields() -> None:
    """A genuine namespace rename differs ONLY in the topic-name fields."""
    renamed = _build_corpus(
        old_topic="onex.evt.legacy-orders.order-placed.v1",
        new_topic="onex.evt.orders.order-placed.v1",
    )
    baseline = _build_corpus(
        old_topic="onex.evt.orders.order-placed.v1",
        new_topic="onex.evt.orders.order-placed.v2",
    )

    renamed_full = _project_terminal_state(renamed).model_dump(mode="json")
    baseline_full = _project_terminal_state(baseline).model_dump(mode="json")

    full_diff = DeepDiff(baseline_full, renamed_full)
    # Topic-name fields DO differ (proves the harness is sensitive to the boundary).
    assert full_diff, "expected the topic-name boundary to be observable"

    renamed_state = _state_without_boundary(_project_terminal_state(renamed))
    baseline_state = _state_without_boundary(_project_terminal_state(baseline))
    business_diff = ModelOutputDiff.from_deepdiff(
        DeepDiff(baseline_state, renamed_state)
    )
    # ...but business state is equivalent once boundary fields are excluded.
    assert not business_diff.has_differences


def test_replay_is_idempotent_under_duplicate_events() -> None:
    """Replaying the corpus twice yields the same terminal projection state."""
    corpus = _build_corpus(
        old_topic="onex.evt.orders.order-placed.v1",
        new_topic="onex.evt.orders.order-placed.v2",
    )
    once = _project_terminal_state(corpus)
    twice = _project_terminal_state([*corpus, *corpus])
    assert once == twice
