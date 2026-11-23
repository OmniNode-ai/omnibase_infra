# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A filtering projection stays healthy through the real counters and health rule (OMN-26520).

Runs the recorded dev shape (179 of 200 work-ledger TERMINAL events filtered, 21
written) through ``_record_projection_apply``, the process-wide apply counters and
``evaluate_projection_apply_flow``, then the same window with the filter field
omitted. The first grades HEALTHY, the second DEGRADED.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import _record_projection_apply
from omnibase_infra.runtime.health.projection_apply_flow import (
    evaluate_projection_apply_flow,
    projection_apply_divergence_status,
)
from omnibase_infra.runtime.observability.projection_apply_counters import (
    get_projection_apply_counters,
    reset_projection_apply_counters_for_test,
)

pytestmark = pytest.mark.integration

TOPIC = "onex.evt.omnimarket.work-ledger-terminal.v1"


class DoneItemsProjectionWriter:
    """Stands in for the handler: the projection identity is its class name."""


@pytest.fixture(autouse=True)
def _fresh_counters() -> Iterator[None]:
    reset_projection_apply_counters_for_test()
    yield
    reset_projection_apply_counters_for_test()


def _grade(*, declare_filter: bool) -> str:
    for _ in range(179):
        _record_projection_apply(
            handler_instance=DoneItemsProjectionWriter(),
            topic=TOPIC,
            rows_upserted=0,
            rows_refused=0,
            not_applicable=1 if declare_filter else 0,
        )
    counters = get_projection_apply_counters()
    counters.close_window()
    verdict = evaluate_projection_apply_flow(
        windows=counters.retained_windows(),
        registered_projections=counters.registered_projections(),
    )
    return projection_apply_divergence_status(verdict)


def test_recorded_dev_window_with_declared_filter_is_healthy() -> None:
    assert _grade(declare_filter=True) == "HEALTHY"


def test_same_window_without_declared_filter_is_degraded() -> None:
    assert _grade(declare_filter=False) == "DEGRADED"
