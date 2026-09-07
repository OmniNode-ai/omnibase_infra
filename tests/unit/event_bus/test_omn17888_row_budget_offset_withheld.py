# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17888: the row-budget refusal must withhold the offset, as it claimed to.

``ProjectionQueryRowBudgetError``'s docstring said, from the commit that
introduced it (omnibase_infra#3288, squash ``c4fe409a``):

    Classified as a write-path failure, not a content failure: the event is
    well-formed and still owed a row, so the offset is withheld and the record
    is redelivered once the caller is repaired.

Nothing implemented that sentence. The type was a direct SIBLING of
``ProjectionNotMaterializedError`` under ``ProjectionError``, and it was named
in no ``except`` clause anywhere in the tree. Every offset-unsafe arm the
runtime has matches ``ProjectionNotMaterializedError`` by EXACT type --
``EventBusKafka._dispatch_to_subscriber``,
``MessageDispatchEngine``'s dispatcher arm, and both ``handler_wiring``
boundary arms -- so the refusal fell through to the generic handler-failure
branch: bounded retry, then ``_route_swallowed_exception`` -> DLQ -> and the
offset ADVANCED. The docstring described the opposite of the behaviour.

That mattered on a schedule, not in principle. The .201 dev lane's busiest
``session_id`` (``9787a4a3-ec49-4819-8bdc-5044efb94550``) held 100,441 rows at
2026-09-07T15:53Z and was growing ~3,029 rows/hour, so the 125,000-row budget
was about eight hours away. Every event on that session would then have been
refused, DLQ'd, and acknowledged -- silently destroying the session's replay
rather than stalling its feed.

These tests drive the SAME harness the OMN-17379 suite drives (the real
``_consume_loop`` over a fake consumer) with the budget error substituted for
``ProjectionNotMaterializedError``. Reusing that harness verbatim is the point:
a divergence in verdict between the two files could then only come from the
exception type, which is the thing under test. ``seek_calls`` is the assertion
because under ``enable_auto_commit=True`` rewinding the fetch position is the
only action that withholds an offset.
"""

from __future__ import annotations

import inspect

import pytest
from aiokafka.structs import TopicPartition

from omnibase_infra.errors import (
    ProjectionError,
    ProjectionNotMaterializedError,
    ProjectionQueryRowBudgetError,
)
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from tests.unit.event_bus.test_omn17379_projection_offset_withheld import (
    TEST_OFFSET,
    TEST_PARTITION,
    TEST_TOPIC,
    _run_loop_with_callback_error,
    kafka_config,
    mock_producer,
)

pytestmark = pytest.mark.unit

_BUDGET_MESSAGE = (
    'projection read of "public"."session_replay_snapshots" matched more than '
    "the 125000 rows this seam will materialise (filters=['session_id']); the "
    "read is refused rather than silently truncated (OMN-17888)"
)


def test_row_budget_error_is_an_offset_unsafe_type_not_merely_a_projection_error() -> (
    None
):
    """The classification IS the mechanism; assert the hierarchy directly.

    RED on the parent: ``ProjectionQueryRowBudgetError`` subclassed
    ``ProjectionError``, so ``except ProjectionNotMaterializedError`` did not
    see it and every offset-unsafe arm in the runtime missed it at once.
    """
    assert issubclass(ProjectionQueryRowBudgetError, ProjectionNotMaterializedError), (
        "the row-budget refusal is a write-path failure -- the event is "
        "well-formed and still owed a row -- so it must be caught by the same "
        "arms that withhold the offset for ProjectionNotMaterializedError"
    )
    # And still a ProjectionError, so nothing that classified it more broadly
    # changes verdict.
    assert issubclass(ProjectionQueryRowBudgetError, ProjectionError)


def test_the_offset_unsafe_arms_match_by_type_so_the_hierarchy_is_load_bearing() -> (
    None
):
    """Prove the premise the subclass relies on: the arms are exact-type.

    If some arm named ``ProjectionQueryRowBudgetError`` separately, the
    hierarchy would be belt-and-braces rather than the mechanism. It does not:
    the type appears in exactly one non-test source file, at its raise site.
    Stated as a test so a future arm that reverts to enumerating types is
    caught here rather than in production.
    """
    from omnibase_infra.event_bus import event_bus_kafka
    from omnibase_infra.runtime import message_dispatch_engine
    from omnibase_infra.runtime.auto_wiring import handler_wiring

    for module in (event_bus_kafka, message_dispatch_engine):
        source = inspect.getsource(module)
        assert "except ProjectionNotMaterializedError" in source, (
            f"{module.__name__} no longer carries the offset-unsafe arm this "
            "classification depends on"
        )
        assert "except ProjectionQueryRowBudgetError" not in source

    handler_source = inspect.getsource(handler_wiring)
    assert handler_source.count("except ProjectionNotMaterializedError") == 2
    assert "except ProjectionQueryRowBudgetError" not in handler_source


@pytest.mark.asyncio
async def test_row_budget_refusal_rewinds_on_first_delivery(
    kafka_config: ModelKafkaEventBusConfig,  # noqa: F811
    mock_producer: object,  # noqa: F811
) -> None:
    """First delivery, full retry budget: rewind, and no dead-letter copy.

    RED on the parent: this took the "retries available" branch, returned
    ``True`` with a WARNING and no requeue behind it, and the record was gone.
    """
    consumer, dlq_calls = await _run_loop_with_callback_error(
        kafka_config,
        mock_producer,  # type: ignore[arg-type]
        ProjectionQueryRowBudgetError(
            _BUDGET_MESSAGE, projection_type="session_replay_snapshots"
        ),
        retry_count=0,
        max_retries=5,
    )

    assert consumer.seek_calls == [
        (TopicPartition(TEST_TOPIC, TEST_PARTITION), TEST_OFFSET)
    ], (
        "OMN-17888: a read the seam refused wrote no row for a non-content "
        "reason. The event is well-formed and still owed a row, so the offset "
        "must be withheld until the CALLER is repaired -- exactly what the "
        "type's own docstring already claimed."
    )
    assert dlq_calls == [], (
        "the record is preserved by withholding the offset, not by a "
        "dead-letter copy -- a DLQ write here is an ACK, and the row the event "
        "was owed is then never written"
    )


@pytest.mark.asyncio
async def test_row_budget_refusal_rewinds_with_retry_budget_exhausted(
    kafka_config: ModelKafkaEventBusConfig,  # noqa: F811
    mock_producer: object,  # noqa: F811
) -> None:
    """Offset-unsafe independently of the retry budget.

    RED on the parent: retries exhausted meant DLQ-and-advance. Re-issuing the
    same statement cannot shrink the result set, so the retry budget is not the
    question; the caller's query shape is, and that is repaired by a deploy.
    """
    consumer, dlq_calls = await _run_loop_with_callback_error(
        kafka_config,
        mock_producer,  # type: ignore[arg-type]
        ProjectionQueryRowBudgetError(
            _BUDGET_MESSAGE, projection_type="session_replay_snapshots"
        ),
        retry_count=5,
        max_retries=5,
    )

    assert consumer.seek_calls == [
        (TopicPartition(TEST_TOPIC, TEST_PARTITION), TEST_OFFSET)
    ]
    assert dlq_calls == []


@pytest.mark.asyncio
async def test_a_plain_projection_error_still_dead_letters(
    kafka_config: ModelKafkaEventBusConfig,  # noqa: F811
    mock_producer: object,  # noqa: F811
) -> None:
    """Positive control: the blast radius is exactly the reclassified type.

    A ``ProjectionError`` that is NOT one of the two offset-unsafe types keeps
    its prior semantics -- DLQ on an exhausted budget, offset advances. Without
    this control the two tests above would also pass if the arm had been
    widened to ``except ProjectionError``, which would convert every projection
    failure into a partition stall.
    """
    consumer, dlq_calls = await _run_loop_with_callback_error(
        kafka_config,
        mock_producer,  # type: ignore[arg-type]
        ProjectionError("connection pool exhausted"),
        retry_count=5,
        max_retries=5,
    )

    assert len(dlq_calls) == 1
    assert consumer.seek_calls == []
