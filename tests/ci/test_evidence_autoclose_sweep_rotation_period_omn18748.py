# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18748 — the cron and the rotation period are one fact in two files.

``backfill_rotation_minutes`` exists to make the backfill arm's rotating
slice advance by exactly one slice per scheduled run. That is a claim about
the SCHEDULE, and the schedule lives in a different file, in a different
language, owned by a different change. Nothing joined them, so when the cron
moved from ``*/30 * * * *`` to ``0 */2 * * *`` on 2026-09-15 the default kept
its old value and the slice began advancing four positions per run while
reading five — half the pool unreachable, silently, for three days.

The unit suite proves the coverage property at the correct period. This file
is the part that keeps it true: it reads the cron out of the workflow and
asserts the request model's default equals it. A future cadence change is
then a red test naming both files, rather than a starvation nobody can see
from either one of them.

CLAUDE.md rule 5: the pairing ships as a gate, not as a comment asking the
next editor to remember.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    _rotating_slice,
    _rotation_tick,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_request import (
    ModelEvidenceAutocloseSweepRequest,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "evidence-autoclose-sweep.yml"


def _cron_expressions() -> list[str]:
    """Every ``cron:`` the sweep workflow declares.

    ``on:`` is parsed by YAML 1.1 as the boolean ``True`` rather than the
    string, so the key is looked up under both spellings instead of assumed —
    the same care the sibling suite takes for the same reason.
    """
    workflow: dict[Any, Any] = yaml.safe_load(
        SWEEP_WORKFLOW.read_text(encoding="utf-8")
    )
    assert isinstance(workflow, dict), f"{SWEEP_WORKFLOW} did not parse to a mapping"
    triggers: dict[str, Any] | None = None
    for key in (True, "on"):
        candidate = workflow.get(key)
        if isinstance(candidate, dict):
            triggers = candidate
            break
    assert triggers is not None, f"{SWEEP_WORKFLOW} declares no trigger block"
    schedule = triggers.get("schedule")
    assert isinstance(schedule, list) and schedule, (
        f"{SWEEP_WORKFLOW} declares no schedule; the rotation period this file "
        "pins is meaningless without one"
    )
    return [str(entry["cron"]) for entry in schedule]


def _interval_minutes(cron: str) -> int:
    """Minutes between consecutive firings of a fixed-interval cron.

    Deliberately NARROW. It understands exactly the two shapes this workflow
    has ever used — ``*/N * * * *`` and ``M */H * * *`` — and raises on
    anything else rather than guessing. A cron this cannot read is a cadence
    whose rotation period nobody has thought about, and the correct outcome is
    a loud failure here, not a silently skipped assertion.
    """
    fields = cron.split()
    assert len(fields) == 5, f"unrecognised cron expression {cron!r}"
    minute, hour = fields[0], fields[1]
    if minute.startswith("*/") and hour == "*":
        return int(minute[2:])
    if minute.isdigit() and hour.startswith("*/"):
        return int(hour[2:]) * 60
    raise AssertionError(
        f"cron {cron!r} is not a fixed-interval schedule this gate can read. "
        "Either express the cadence as '*/N * * * *' or 'M */H * * *', or "
        "extend this function deliberately — do not delete the assertion."
    )


def test_the_rotation_period_equals_the_sweep_cron_interval() -> None:
    """The join OMN-18748 was opened for.

    Asserted against the model DEFAULT rather than against a value the
    workflow passes, because the workflow passes nothing: the contract
    declares the field and the default is what every scheduled run gets.
    """
    crons = _cron_expressions()
    assert len(crons) == 1, (
        "this gate assumes one schedule; several cadences would need a "
        "rotation period each, which the node does not have"
    )
    interval = _interval_minutes(crons[0])
    default = ModelEvidenceAutocloseSweepRequest.model_fields[
        "backfill_rotation_minutes"
    ].default

    assert default == interval, (
        f"{SWEEP_WORKFLOW.name} fires every {interval} minutes but "
        f"`backfill_rotation_minutes` defaults to {default}. The rotating "
        "slice advances by one period per run, so consecutive runs would skip "
        f"{max(interval // default, 1) - 1} slice(s) and part of the backfill "
        "pool would never be adjudicated. Change both together."
    )


def test_the_workflow_passes_no_rotation_override() -> None:
    """The default is the live value, which is what makes the test above bind.

    If the workflow ever started passing ``--backfill-rotation-minutes``, the
    assertion above would be pinning a value nothing reads — green, and about
    nothing. This is the precondition, stated rather than assumed.
    """
    body = SWEEP_WORKFLOW.read_text(encoding="utf-8")
    assert "--backfill-rotation-minutes" not in body, (
        "the workflow now overrides the rotation period; the cron/default "
        "pairing above no longer describes what a scheduled run uses, and this "
        "gate must be repointed at the override rather than left green"
    )


# OMN-21722 AC2: the sized width (40) drains a 133 pool in ceil(133/40) runs.
_BACKFILL_POOL_SIZE = 133
_SIZED_DRAIN_RUNS = math.ceil(_BACKFILL_POOL_SIZE / 40)


def test_the_default_width_on_the_cron_cadence_does_not_partition_the_pool() -> None:
    """OMN-21722 AC2: the wider slice, advanced by the real clock, covers the pool.

    Fires consecutive scheduled runs exactly one cron interval apart, maps each
    to its rotation tick the way the handler does (``_rotation_tick`` with the
    model's default period), and slices the pool at the model's default width.
    Consecutive runs must land on consecutive ticks, and ``ceil(133/40)`` runs
    must cover every candidate: a rotation period shorter than the cron would
    skip slices and partition the pool, and a narrower width would leave most
    of it unexamined after those runs.
    """
    interval = _interval_minutes(_cron_expressions()[0])
    fields = ModelEvidenceAutocloseSweepRequest.model_fields
    rotation = fields["backfill_rotation_minutes"].default
    width = fields["backfill_max_candidates"].default
    assert isinstance(rotation, int)
    assert isinstance(width, int)

    pool = list(range(_BACKFILL_POOL_SIZE))
    first_run = datetime(2026, 10, 30, 20, 10, tzinfo=UTC)
    ticks = [
        _rotation_tick(first_run + timedelta(minutes=interval * run), rotation)
        for run in range(_SIZED_DRAIN_RUNS)
    ]
    assert ticks == list(range(ticks[0], ticks[0] + _SIZED_DRAIN_RUNS)), (
        f"runs {interval} minutes apart landed on ticks {ticks} with a "
        f"{rotation}-minute rotation period; consecutive runs must advance "
        "exactly one slice"
    )
    covered: set[int] = set()
    for tick in ticks:
        covered |= set(_rotating_slice(pool, width, tick))
    missing = len(pool) - len(covered)
    assert missing == 0, (
        f"{_SIZED_DRAIN_RUNS} scheduled runs at backfill_max_candidates={width} "
        f"left {missing} of {len(pool)} backfill candidates unexamined"
    )
