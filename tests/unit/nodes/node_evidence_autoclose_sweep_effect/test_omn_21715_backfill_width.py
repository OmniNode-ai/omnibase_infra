# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21722 AC1: the backfill width is sized from measured run time.

The closer offered ``backfill_max_candidates`` = 5 per run against a pool of 133
and flipped 0 tickets in 8 scheduled runs. The width is a run-budget knob, so it
is held against the budget it spends: the workflow's own ``timeout-minutes``.

The measurements are the closer's own receipts for the 8 scheduled runs of
2026-10-30T20:10Z to 2026-10-31T10:09Z (``duration_ms`` of the node result and
the count of outcomes that reached the verifier). They are constants here with
their source, so a change to the width, the timeout or the cadence is a red test
that names the arithmetic, not a silent starvation or a killed job.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import yaml

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    _rotating_slice,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_request import (
    ModelEvidenceAutocloseSweepRequest,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SWEEP_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "evidence-autoclose-sweep.yml"

# Receipts of the 8 scheduled runs (run ids 41121136343 .. 41191613667).
_WORST_MEASURED_SWEEP_SECONDS = 285.0  # run 41175878375, 9 verifier-class calls
_WORST_MEASURED_VERIFIER_CALLS = 9
_WIDTH_BEFORE = 5  # every one of the 8 runs selected exactly 5 backfill candidates
_POOL_SIZE = 133  # backfill_pool_size in every one of the 8 receipts
# The sweep may spend at most this share of the job timeout; the rest is the
# job's own setup (checkouts, contract roots, venv purity) and headroom.
_SWEEP_SHARE_OF_TIMEOUT = 0.5


def _job_timeout_seconds() -> float:
    workflow: dict[Any, Any] = yaml.safe_load(
        SWEEP_WORKFLOW.read_text(encoding="utf-8")
    )
    jobs = workflow["jobs"]
    timeouts = [
        float(job["timeout-minutes"]) * 60
        for job in jobs.values()
        if "timeout-minutes" in job
    ]
    assert timeouts, f"{SWEEP_WORKFLOW} declares no job timeout"
    return min(timeouts)


def _default_width() -> int:
    default = ModelEvidenceAutocloseSweepRequest.model_fields[
        "backfill_max_candidates"
    ].default
    assert isinstance(default, int)
    return default


def test_the_default_width_is_the_sized_value() -> None:
    assert _default_width() == 40
    assert (
        ModelEvidenceAutocloseSweepRequest(
            correlation_id=uuid4()
        ).backfill_max_candidates
        == _default_width()
    )


def test_the_worst_measured_run_with_the_wider_slice_fits_inside_the_timeout() -> None:
    per_call = _WORST_MEASURED_SWEEP_SECONDS / _WORST_MEASURED_VERIFIER_CALLS
    added = _default_width() - _WIDTH_BEFORE
    projected = _WORST_MEASURED_SWEEP_SECONDS + added * per_call
    budget = _job_timeout_seconds() * _SWEEP_SHARE_OF_TIMEOUT
    assert projected <= budget, (
        f"backfill_max_candidates={_default_width()} projects a {projected:.0f}s "
        f"sweep at the worst measured {per_call:.1f}s per call, over the "
        f"{budget:.0f}s share ({_SWEEP_SHARE_OF_TIMEOUT:.0%}) of the workflow's "
        f"{_job_timeout_seconds():.0f}s job timeout"
    )


def test_the_wider_slice_covers_the_pool_in_ceil_pool_over_width_ticks() -> None:
    width = _default_width()
    pool = list(range(_POOL_SIZE))
    ticks = math.ceil(_POOL_SIZE / width)
    covered: set[int] = set()
    for tick in range(ticks):
        chosen = _rotating_slice(pool, width, tick)
        assert len(chosen) == width
        assert len(set(chosen)) == width, "a slice must not hand an item back twice"
        covered |= set(chosen)
    assert covered == set(pool)


def test_the_width_stays_inside_the_declared_ceiling() -> None:
    field = ModelEvidenceAutocloseSweepRequest.model_fields["backfill_max_candidates"]
    ceiling = next(m.le for m in field.metadata if hasattr(m, "le"))
    assert _default_width() <= ceiling
