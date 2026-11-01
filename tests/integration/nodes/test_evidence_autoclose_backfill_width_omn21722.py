# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21722: the backfill width drains the whole pool through the real request model.

Drives the request model default and the handler's rotating slice together, the
way consecutive scheduled ticks do, and holds the width to the workflow's own
cadence and timeout read from disk.
"""

from __future__ import annotations

import math
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    _rotating_slice,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_request import (
    ModelEvidenceAutocloseSweepRequest,
)

pytestmark = pytest.mark.integration

_WORKFLOW = (
    Path(__file__).resolve().parents[3]
    / ".github"
    / "workflows"
    / "evidence-autoclose-sweep.yml"
)


# The drain promise of OMN-21722: a 133 pool in ceil(133/40) = 4 scheduled ticks
# (8 hours at the 2-hour cadence), not the 27 ticks (54 hours) of the old width.
_POOL_SIZE = 133
_DRAIN_TICKS = math.ceil(_POOL_SIZE / 40)


def test_consecutive_ticks_at_the_default_width_drain_the_pool() -> None:
    request = ModelEvidenceAutocloseSweepRequest(correlation_id=uuid4())
    width = request.backfill_max_candidates
    pool = list(range(_POOL_SIZE))
    ticks = math.ceil(len(pool) / width)
    assert ticks <= _DRAIN_TICKS, (
        f"backfill_max_candidates={width} needs {ticks} ticks to sweep a "
        f"{_POOL_SIZE} pool once; the sized width drains it in {_DRAIN_TICKS}"
    )
    seen: set[int] = set()
    for tick in range(ticks):
        seen |= set(_rotating_slice(pool, width, tick))
    assert seen == set(pool)


def test_the_workflow_timeout_leaves_room_for_the_default_width() -> None:
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    timeouts = [
        float(job["timeout-minutes"]) * 60
        for job in workflow["jobs"].values()
        if "timeout-minutes" in job
    ]
    assert timeouts
    width = ModelEvidenceAutocloseSweepRequest(
        correlation_id=uuid4()
    ).backfill_max_candidates
    projected = 285.0 + (width - 5) * (285.0 / 9)
    assert projected <= min(timeouts) * 0.5
