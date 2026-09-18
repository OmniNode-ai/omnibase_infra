# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18748 — the rotating slice skips half its pool when the period drifts.

THE MEASURED DEFECT. A collaborator observed that OMN-18172 was never a
candidate in any scheduled evidence-autoclose sweep between 2026-09-17T22:15Z
and 2026-09-18T16:16Z, although verified work sat on it. Its newest merged OCC
companion (onex_change_control#9974) landed 2026-09-17T12:01:32Z, so the 6-hour
forward window let go of it the same evening and everything after that was the
backfill arm's job.

The backfill arm did not refuse it. The arm never looked.

``_rotation_tick`` divides the wall clock by ``backfill_rotation_minutes``,
whose default is 30 and whose own field description says why: *"Set it to the
workflow's real cron interval; the default matches the sweep's \\*/30
schedule."* The cron moved to ``0 */2 * * *`` on 2026-09-15 and the default did
not follow, so consecutive scheduled runs advance the tick by FOUR while the
slice is five wide. The start index jumps twenty positions per run and five are
read, and with ``gcd(20, pool)`` dividing the pool into blocks the same offsets
come back forever.

Measured against the live pool size from run 35299192253's own receipt
(``backfill_pool_size: 150``, ``backfill_candidates_selected: 5``): 75 of 150
positions reachable, 75 unreachable, indefinitely. A ticket in the starved half
is never refused, never commented and never appears in any receipt, which reads
exactly like a ticket nobody has evidence for.

Every test below carries its negative control at the OLD period, because a
coverage assertion that passes for the wrong reason is indistinguishable from
one that passes for the right one.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    _rotating_slice,
    _rotation_tick,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_request import (
    ModelEvidenceAutocloseSweepRequest,
)

pytestmark = pytest.mark.unit

#: The sweep's real cadence since 2026-09-15 (`cron: '0 */2 * * *'`).
_CRON_INTERVAL_MINUTES = 120

#: The period the default carried while the cron said something else.
_DRIFTED_PERIOD_MINUTES = 30

#: Read off run 35299192253's receipt rather than invented: that run reported
#: `backfill_pool_size: 150` and `backfill_candidates_selected: 5`.
_POOL_SIZE = 150
_SLICE_WIDTH = 5

_FIRST_TICK_AT = datetime(2026, 9, 18, 0, 24, 0, tzinfo=UTC)


def _positions_covered(*, rotation_minutes: int, ticks: int) -> set[int]:
    """Pool indices the slice reads over ``ticks`` consecutive SCHEDULED runs.

    The runs are spaced at the real cron interval regardless of
    ``rotation_minutes`` — that divergence is the whole subject. Indices are
    used rather than elements so the result is comparable against the pool's
    full index set without depending on what the pool holds.
    """
    pool = list(range(_POOL_SIZE))
    covered: set[int] = set()
    for step in range(ticks):
        now = _FIRST_TICK_AT + timedelta(minutes=_CRON_INTERVAL_MINUTES * step)
        covered |= set(
            _rotating_slice(pool, _SLICE_WIDTH, _rotation_tick(now, rotation_minutes))
        )
    return covered


def test_consecutive_scheduled_runs_advance_the_slice_by_exactly_one() -> None:
    """One cron interval must be one rotation period.

    Asserted on the tick DIFFERENCE rather than on an absolute value: the tick
    is counted from the epoch, so any absolute assertion would be a restatement
    of the timestamp the test chose.
    """
    default_period = ModelEvidenceAutocloseSweepRequest.model_fields[
        "backfill_rotation_minutes"
    ].default
    later = _FIRST_TICK_AT + timedelta(minutes=_CRON_INTERVAL_MINUTES)

    advance = _rotation_tick(later, default_period) - _rotation_tick(
        _FIRST_TICK_AT, default_period
    )
    assert advance == 1, (
        "consecutive scheduled runs must advance the rotating slice by exactly "
        f"one; at rotation_minutes={default_period} against a "
        f"{_CRON_INTERVAL_MINUTES}-minute cron they advance by {advance}, so "
        f"{advance - 1} slice(s) of {_SLICE_WIDTH} are skipped every run"
    )


def test_the_drifted_period_is_what_advances_by_four() -> None:
    """The negative control, and the diagnosis.

    Pinning the defect's own arithmetic keeps this suite honest about WHAT it
    fixed: a later change that made the coverage test pass by some other route
    would leave this one failing.
    """
    later = _FIRST_TICK_AT + timedelta(minutes=_CRON_INTERVAL_MINUTES)
    advance = _rotation_tick(later, _DRIFTED_PERIOD_MINUTES) - _rotation_tick(
        _FIRST_TICK_AT, _DRIFTED_PERIOD_MINUTES
    )
    assert advance == 4


def test_the_whole_pool_is_covered_within_one_full_rotation() -> None:
    """`_rotating_slice`'s own documented property, at the real cadence.

    Its docstring states ``ceil(len(pool) / width)`` consecutive ticks cover
    the whole pool. That holds only when consecutive ticks differ by one, which
    is what the default now guarantees.
    """
    default_period = ModelEvidenceAutocloseSweepRequest.model_fields[
        "backfill_rotation_minutes"
    ].default
    ticks = math.ceil(_POOL_SIZE / _SLICE_WIDTH)

    covered = _positions_covered(rotation_minutes=default_period, ticks=ticks)

    assert covered == set(range(_POOL_SIZE))


def test_the_drifted_period_starves_half_the_pool_forever() -> None:
    """THE POSITIVE CONTROL for the claim above, and the measurement.

    Thirty days of 2-hourly runs — 360 ticks against a pool of 150 and a slice
    of 5, thirty times more than a full rotation needs — and half the pool is
    still unread. That is starvation rather than slow drain, and it is the
    reason OMN-18172 produced no outcome row in any run of the observed window.
    """
    covered = _positions_covered(rotation_minutes=_DRIFTED_PERIOD_MINUTES, ticks=360)

    assert len(covered) == 75
    starved = set(range(_POOL_SIZE)) - covered
    assert len(starved) == 75
    # The starved set is structural, not incidental: it is the same offset
    # within every block of ten, which is `gcd(slice_advance, pool)` at work.
    assert {index % 10 for index in starved} == {5, 6, 7, 8, 9}


def test_a_pool_no_larger_than_the_slice_is_returned_whole() -> None:
    """Unchanged behaviour, pinned because the coverage claim leans on it.

    At that size every tick sees everything, so the coverage property holds
    trivially and must keep holding for a pool that shrinks below the slice
    width rather than depending on the rotation at all.
    """
    pool = [0, 1, 2]
    for tick in range(10):
        assert _rotating_slice(pool, _SLICE_WIDTH, tick) == pool
