# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A filtering projection is not a silent writer (OMN-26520).

Measured on the dev runtime 2026-11-23T10:13Z: the done-items writer consumed
work-ledger TERMINAL events and 179 of the last 200 carried an outcome other than
done, for which it writes nothing by design. A window of at least twenty such
events read as "consumed and upserted ZERO rows" and graded the runtime DEGRADED,
which the container healthcheck (``--degraded-policy fail``) turns into unhealthy.

A filtered event is not counted as consumed (as a runtime tick is not, OMN-26419).
Each HEALTHY assertion is paired with a DEGRADED one that differs in the single
fact graded: a writer that reports no filter, or that leaves the floor of events
unfiltered, still reads as a silent zero.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    EVENTS_NOT_APPLICABLE_KEY,
    _extract_count,
    _record_projection_apply,
)
from omnibase_infra.runtime.health.projection_apply_flow import (
    APPLY_DIVERGENCE_MIN_CONSUMED,
    evaluate_projection_apply_flow,
    projection_apply_divergence_status,
)
from omnibase_infra.runtime.observability.projection_apply_counters import (
    get_projection_apply_counters,
    reset_projection_apply_counters_for_test,
)

pytestmark = pytest.mark.unit

TOPIC = "onex.evt.omnimarket.work-ledger-terminal.v1"


class DoneItemsProjectionWriter:
    """Stands in for the handler: the projection identity is its class name."""


@pytest.fixture(autouse=True)
def _fresh_counters() -> Iterator[None]:
    reset_projection_apply_counters_for_test()
    yield
    reset_projection_apply_counters_for_test()


def _status_after(*, filtered: int, silent: int) -> str:
    for _ in range(filtered):
        _record_projection_apply(
            handler_instance=DoneItemsProjectionWriter(),
            topic=TOPIC,
            rows_upserted=0,
            rows_refused=0,
            not_applicable=1,
        )
    for _ in range(silent):
        _record_projection_apply(
            handler_instance=DoneItemsProjectionWriter(),
            topic=TOPIC,
            rows_upserted=0,
            rows_refused=0,
        )
    counters = get_projection_apply_counters()
    counters.close_window()
    verdict = evaluate_projection_apply_flow(
        windows=counters.retained_windows(),
        registered_projections=counters.registered_projections(),
    )
    return projection_apply_divergence_status(verdict)


def test_all_events_filtered_is_healthy() -> None:
    assert _status_after(filtered=30, silent=0) == "HEALTHY"


def test_silent_zero_writer_still_degraded() -> None:
    assert _status_after(filtered=0, silent=30) == "DEGRADED"


def test_floor_of_unfiltered_zero_events_still_degraded() -> None:
    assert (
        _status_after(filtered=40, silent=APPLY_DIVERGENCE_MIN_CONSUMED) == "DEGRADED"
    )


def test_below_floor_of_unfiltered_zero_events_is_healthy() -> None:
    assert (
        _status_after(filtered=40, silent=APPLY_DIVERGENCE_MIN_CONSUMED - 1)
        == "HEALTHY"
    )


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({EVENTS_NOT_APPLICABLE_KEY: 1, "rows_upserted": 0}, 1),
        ({"rows_upserted": 0}, 0),
        ({EVENTS_NOT_APPLICABLE_KEY: "x"}, 0),
        ({EVENTS_NOT_APPLICABLE_KEY: None}, 0),
        ({EVENTS_NOT_APPLICABLE_KEY: -3}, 0),
        (None, 0),
    ],
)
def test_extract_reads_only_a_positive_declared_count(
    result: object, expected: int
) -> None:
    assert _extract_count(result, EVENTS_NOT_APPLICABLE_KEY) == expected
