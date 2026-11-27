# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26106: every sweep step that starts hosted dod_verify declares its budget.

Hosted ``dod_verify`` refuses to start without ``DOD_VERIFY_START_BUDGET_S``
(omnimarket OMN-25856) and answers ``unresolved`` with ``total_checks=0``; the
sweep step carried no such value, so every ticket read ``0/0 ACs verified``.
The value must be above 0 and below the 600 s dispatch deadline; the dev
runtime overlay declares 540.

Falsifier: delete the variable from either step's ``env:`` and the matching
test names the step.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/evidence-autoclose-sweep.yml"
_VAR = "DOD_VERIFY_START_BUDGET_S"
_DISPATCH_DEADLINE_S = 600.0


def _steps_running(needle: str) -> list[dict[str, object]]:
    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = doc["jobs"]["evidence-autoclose-sweep"]["steps"]
    return [s for s in steps if needle in str(s.get("run", ""))]


@pytest.mark.parametrize(
    "needle", ["skill evidence_autoclose_sweep", 'skill dod_verify "$']
)
def test_step_declares_a_usable_start_budget(needle: str) -> None:
    steps = _steps_running(needle)
    assert len(steps) == 1, f"expected one step running {needle!r}, found {len(steps)}"
    env = steps[0].get("env") or {}
    assert _VAR in env, f"step {steps[0].get('name')!r} does not declare {_VAR}"
    assert 0 < float(env[_VAR]) < _DISPATCH_DEADLINE_S
