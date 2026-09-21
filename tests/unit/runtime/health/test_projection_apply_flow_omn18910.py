# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Falsifiers for the two projection flow-invariant dimensions (OMN-18910).

Each test below is one of the falsifiers named on the ticket. They were written
and run RED against ``origin/dev`` before the module under test existed: the
whole file failed on the import, which is the honest starting point for a gate
whose whole claim is that it refuses something today's tree does not.

The negative controls are the half that matters. A dimension that degrades on
everything is a freeze and gets switched off within a week, which is how the
fleet arrived at eight dimensions that cannot fail. So every DEGRADED assertion
here is paired with a HEALTHY one over an input that differs in exactly the
fact being graded.

Related Tickets:
    - OMN-18910: these dimensions (epic OMN-18906 AC-4)
    - OMN-18880: the total-refusal condition replayed by AC1's second falsifier
    - OMN-18992: the guard-refused zero this dimension grades the accumulation of
"""

from __future__ import annotations

import pytest

from omnibase_infra.models.observability.model_projection_apply_delta import (
    ModelProjectionApplyDelta,
)
from omnibase_infra.runtime.health.projection_apply_flow import (
    APPLY_DIVERGENCE_MIN_CONSUMED,
    DELTA_DROP_MIN_ACCUMULATION,
    DELTA_DROP_MIN_RISING_WINDOWS,
    OUTCOME_APPLY_FLOW_UNOBSERVED,
    describe_projection_apply_divergence,
    describe_projection_delta_dropped,
    evaluate_projection_apply_flow,
    projection_apply_divergence_status,
    projection_delta_dropped_status,
)

pytestmark = pytest.mark.unit


def _delta(
    projection: str,
    *,
    consumed: int = 0,
    upserted: int = 0,
    refused: int = 0,
    dropped_total: int = 0,
    topic: str = "onex.evt.some-topic.v1",
) -> ModelProjectionApplyDelta:
    """One projection's accounting for one closed window."""
    return ModelProjectionApplyDelta(
        projection=projection,
        topic=topic,
        consumed=consumed,
        upserted=upserted,
        refused_by_guard=refused,
        deltas_dropped_total=dropped_total,
    )


def _windows(
    *per_window: tuple[ModelProjectionApplyDelta, ...],
) -> tuple[tuple[ModelProjectionApplyDelta, ...], ...]:
    return tuple(per_window)


# --------------------------------------------------------------------- AC1


def test_ac1_consuming_without_writing_grades_degraded() -> None:
    """A projection whose consumed count advances while upserts stay at zero."""
    consumed_per_window = APPLY_DIVERGENCE_MIN_CONSUMED
    verdict = evaluate_projection_apply_flow(
        windows=_windows(
            (_delta("node_projection_lab_lane_health", consumed=consumed_per_window),),
            (_delta("node_projection_lab_lane_health", consumed=consumed_per_window),),
        ),
        registered_projections=("node_projection_lab_lane_health",),
    )

    assert projection_apply_divergence_status(verdict) == "DEGRADED"
    assert verdict.diverging_projections == ("node_projection_lab_lane_health",)
    assert "node_projection_lab_lane_health" in describe_projection_apply_divergence(
        verdict
    )


def test_ac1_omn18880_total_refusal_condition_grades_degraded() -> None:
    """Replay of OMN-18880: every event refused, so nothing is ever written.

    The runner-fleet projection rejected its own emitter's payload on a missing
    observation field for about nine hours. It consumed the whole time, wrote
    nothing the whole time, and lag was zero the whole time.
    """
    verdict = evaluate_projection_apply_flow(
        windows=_windows(
            *(
                (
                    _delta(
                        "node_projection_runner_fleet",
                        consumed=APPLY_DIVERGENCE_MIN_CONSUMED,
                        upserted=0,
                        topic="onex.evt.runner-fleet.v1",
                    ),
                )
                for _ in range(10)
            )
        ),
        registered_projections=("node_projection_runner_fleet",),
    )

    assert projection_apply_divergence_status(verdict) == "DEGRADED"
    assert "node_projection_runner_fleet" in verdict.diverging_projections


def test_ac1_negative_control_a_writing_projection_is_healthy() -> None:
    """The same consumed volume, with rows landing, must not degrade."""
    verdict = evaluate_projection_apply_flow(
        windows=_windows(
            (
                _delta(
                    "node_projection_consumer_flow",
                    consumed=APPLY_DIVERGENCE_MIN_CONSUMED * 4,
                    upserted=1,
                ),
            ),
        ),
        registered_projections=("node_projection_consumer_flow",),
    )

    assert projection_apply_divergence_status(verdict) == "HEALTHY"
    assert verdict.diverging_projections == ()


def test_ac1_below_the_floor_is_not_a_verdict() -> None:
    """A trickle that wrote nothing yet is a sample too small to grade."""
    verdict = evaluate_projection_apply_flow(
        windows=_windows(
            (
                _delta(
                    "node_projection_quiet", consumed=APPLY_DIVERGENCE_MIN_CONSUMED - 1
                ),
            ),
        ),
        registered_projections=("node_projection_quiet",),
    )

    assert projection_apply_divergence_status(verdict) == "HEALTHY"


# --------------------------------------------------------------------- AC2


def test_ac2_a_high_but_flat_drop_count_is_healthy() -> None:
    """Flat-state alarming is the noise that gets a dimension muted."""
    frozen = 6_210_195
    verdict = evaluate_projection_apply_flow(
        windows=_windows(
            *(
                (
                    _delta(
                        "node_projection_consumer_flow",
                        consumed=5,
                        upserted=5,
                        dropped_total=frozen,
                    ),
                )
                for _ in range(6)
            )
        ),
        registered_projections=("node_projection_consumer_flow",),
    )

    assert projection_delta_dropped_status(verdict) == "HEALTHY"
    assert verdict.drop_accumulating_projections == ()


def test_ac2_an_accumulating_drop_count_is_degraded() -> None:
    """A cache discarding delta after delta is the OMN-18905 shape."""
    rising = tuple(
        (
            _delta(
                "node_projection_consumer_flow",
                consumed=50,
                upserted=50,
                dropped_total=DELTA_DROP_MIN_ACCUMULATION * (n + 1),
            ),
        )
        for n in range(DELTA_DROP_MIN_RISING_WINDOWS + 1)
    )
    verdict = evaluate_projection_apply_flow(
        windows=_windows(*rising),
        registered_projections=("node_projection_consumer_flow",),
    )

    assert projection_delta_dropped_status(verdict) == "DEGRADED"
    assert verdict.drop_accumulating_projections == ("node_projection_consumer_flow",)
    assert "node_projection_consumer_flow" in describe_projection_delta_dropped(verdict)


def test_ac2_a_single_rise_is_idempotence_not_a_defect() -> None:
    """One drop is legitimate idempotence and must not alarm."""
    verdict = evaluate_projection_apply_flow(
        windows=_windows(
            (
                _delta(
                    "node_projection_work_events",
                    consumed=9,
                    upserted=9,
                    dropped_total=0,
                ),
            ),
            (
                _delta(
                    "node_projection_work_events",
                    consumed=9,
                    upserted=9,
                    dropped_total=1,
                ),
            ),
            (
                _delta(
                    "node_projection_work_events",
                    consumed=9,
                    upserted=9,
                    dropped_total=1,
                ),
            ),
        ),
        registered_projections=("node_projection_work_events",),
    )

    assert projection_delta_dropped_status(verdict) == "HEALTHY"


def test_ac2_a_declared_immutable_grain_exposure_is_excluded() -> None:
    """Session replay and work events drop by design; they are exemptions.

    Both key on a content-addressed, immutable grain, so the cache only ever
    compares a key against a delta derived from the same source event. The drop
    is the intended idempotence. Enumerated and excluded, not thresholded over.
    """
    rising = tuple(
        (
            _delta(
                "node_projection_session_replay",
                consumed=50,
                upserted=50,
                dropped_total=DELTA_DROP_MIN_ACCUMULATION * (n + 1),
            ),
        )
        for n in range(DELTA_DROP_MIN_RISING_WINDOWS + 1)
    )
    verdict = evaluate_projection_apply_flow(
        windows=_windows(*rising),
        registered_projections=("node_projection_session_replay",),
        immutable_grain_projections=("node_projection_session_replay",),
    )

    assert projection_delta_dropped_status(verdict) == "HEALTHY"
    assert verdict.excluded_immutable_grain == ("node_projection_session_replay",)
    # The exclusion is rendered, because unreported is not the same as ungated.
    assert "node_projection_session_replay" in describe_projection_delta_dropped(
        verdict
    )


# --------------------------------------------------------------------- AC3


def test_ac3_a_registered_projection_with_no_closed_window_fails_closed() -> None:
    """Absent input is UNKNOWN, and UNKNOWN is never rendered as HEALTHY."""
    verdict = evaluate_projection_apply_flow(
        windows=(),
        registered_projections=("node_projection_runner_fleet",),
    )

    assert verdict.apply_flow_evaluated is False
    assert projection_apply_divergence_status(verdict) == "DEGRADED"
    assert projection_delta_dropped_status(verdict) == "DEGRADED"
    assert OUTCOME_APPLY_FLOW_UNOBSERVED in describe_projection_apply_divergence(
        verdict
    )
    assert OUTCOME_APPLY_FLOW_UNOBSERVED in describe_projection_delta_dropped(verdict)


def test_ac3_a_process_wiring_no_projection_is_not_degraded() -> None:
    """A measured zero is a different fact from an unreadable one.

    A process that dispatches no projection has nothing to diverge. Degrading
    it would put every non-projection runtime permanently DEGRADED on a
    dimension it cannot act on, and whether a projection SHOULD be attached
    here is already owned by ``projection_attachment``.
    """
    verdict = evaluate_projection_apply_flow(
        windows=(),
        registered_projections=(),
    )

    assert projection_apply_divergence_status(verdict) == "HEALTHY"
    assert projection_delta_dropped_status(verdict) == "HEALTHY"


def test_ac3_a_dropped_total_that_goes_backwards_fails_closed() -> None:
    """A counter that decreases was not read from one continuous process.

    Either the writer restarted mid-window or the reading is not what it says
    it is. Both are indeterminate, and an indeterminate accumulation reading
    must not be rendered as a measured flat one.
    """
    verdict = evaluate_projection_apply_flow(
        windows=_windows(
            (
                _delta(
                    "node_projection_runner_fleet",
                    consumed=5,
                    upserted=5,
                    dropped_total=900,
                ),
            ),
            (
                _delta(
                    "node_projection_runner_fleet",
                    consumed=5,
                    upserted=5,
                    dropped_total=3,
                ),
            ),
        ),
        registered_projections=("node_projection_runner_fleet",),
    )

    assert projection_delta_dropped_status(verdict) == "DEGRADED"
    assert verdict.indeterminate_drop_projections == ("node_projection_runner_fleet",)


# --------------------------------------------------------------------- AC4


def test_ac4_a_healthy_lane_raises_neither_and_records_that_it_evaluated() -> None:
    """A silent evaluation and one that never ran are otherwise the same."""
    verdict = evaluate_projection_apply_flow(
        windows=_windows(
            (
                _delta("node_projection_consumer_flow", consumed=40, upserted=40),
                _delta("node_projection_runner_fleet", consumed=12, upserted=12),
            ),
            (
                _delta("node_projection_consumer_flow", consumed=38, upserted=38),
                _delta("node_projection_runner_fleet", consumed=9, upserted=9),
            ),
        ),
        registered_projections=(
            "node_projection_consumer_flow",
            "node_projection_runner_fleet",
        ),
    )

    assert projection_apply_divergence_status(verdict) == "HEALTHY"
    assert projection_delta_dropped_status(verdict) == "HEALTHY"
    assert verdict.apply_flow_evaluated is True
    assert verdict.observed_window_count == 2
    assert verdict.projection_count == 2
    # The evaluation states its own scope, so a reader can tell a green run
    # from a run that enumerated nothing.
    divergence_detail = describe_projection_apply_divergence(verdict)
    assert "2" in divergence_detail
    assert "window" in divergence_detail
