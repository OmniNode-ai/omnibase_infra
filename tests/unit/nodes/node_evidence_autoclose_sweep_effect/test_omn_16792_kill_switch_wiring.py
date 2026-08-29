# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16792 AC3 — the kill switch must be reachable from where it is set.

`node_evidence_autoclose_sweep_effect` honours ``ONEX_AUTOCLOSE_DISABLED``
unconditionally and does zero I/O when it is set. The node side was never in
doubt; the wiring was. The node reads the switch from its PROCESS environment,
and a GitHub repository variable is not exported into a step's environment — it
is reachable only through the ``vars.*`` expression context. Before this the
sweep step's ``env:`` block named ``LINEAR_API_KEY``, ``GH_TOKEN``,
``UV_NO_SYNC`` and ``ONEX_CC_REPO_PATH`` and nothing else, so
``gh variable set ONEX_AUTOCLOSE_DISABLED --body 1`` set a value the sweep
process could never see. The documented halt procedure would have done nothing,
and would have done nothing *silently*, at the moment an operator reached for
it.

This is a wiring assertion, not a behaviour assertion: the failure it guards is
"the code honours a switch nothing can turn", which no handler-level test can
observe. It reads the committed workflow because the workflow IS the wiring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_KILL_SWITCH = "ONEX_AUTOCLOSE_DISABLED"
_WORKFLOW = (
    Path(__file__).resolve().parents[4]
    / ".github"
    / "workflows"
    / "evidence-autoclose-sweep.yml"
)


def _sweep_step() -> dict[str, Any]:
    """The step that actually runs the sweep, found by its invocation.

    Located by the command it runs rather than by name or index, so renaming
    the step or inserting one ahead of it cannot silently retarget this test at
    a step that never runs the sweep.

    Matched on ``skill evidence_autoclose_sweep`` alone: OMN-16846 moved the
    invocation off ``uv run onex`` and onto the dispatch venv's own binary
    (``"${DISPATCH_VENV}/bin/onex" skill evidence_autoclose_sweep``), so
    requiring ``onex skill ...`` as one contiguous substring matched nothing
    and this module's assertions collapsed rather than failing on their
    subject.
    """
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["evidence-autoclose-sweep"]["steps"]
    matches = [
        step
        for step in steps
        if "skill evidence_autoclose_sweep" in str(step.get("run", ""))
    ]
    assert len(matches) == 1, (
        "expected exactly one step invoking `skill evidence_autoclose_sweep`, "
        f"found {len(matches)}"
    )
    return matches[0]


def test_the_sweep_step_exports_the_kill_switch() -> None:
    step = _sweep_step()
    env = step.get("env") or {}
    assert _KILL_SWITCH in env, (
        f"the sweep step does not export {_KILL_SWITCH}, so the node cannot read "
        "it and `gh variable set` is a no-op — the kill switch is documented but "
        "unwired"
    )


def test_the_kill_switch_is_sourced_from_the_repo_variable() -> None:
    """It must come from `vars.*`, not an input and not a literal.

    An input is a per-dispatch choice and would leave every scheduled run
    unhaltable — the runs that matter most. A literal would be a value only a
    commit can change, which is not a switch at all.
    """
    value = str((_sweep_step().get("env") or {}).get(_KILL_SWITCH, ""))
    assert f"vars.{_KILL_SWITCH}" in value, (
        f"{_KILL_SWITCH} must be sourced from the repo variable "
        f"(`${{{{ vars.{_KILL_SWITCH} }}}}`), got: {value!r}"
    )
    assert "github.event.inputs" not in value, (
        "a per-dispatch input cannot halt the scheduled runs, which are the ones "
        "an operator needs to stop"
    )


def test_the_kill_switch_is_not_a_workflow_input() -> None:
    """The workflow's own header says it must not be surfaced as an input."""
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    # PyYAML parses the bare `on:` key as the boolean True.
    triggers = workflow.get("on") or workflow.get(True) or {}
    inputs = (triggers.get("workflow_dispatch") or {}).get("inputs") or {}
    assert _KILL_SWITCH.lower() not in {name.lower() for name in inputs}


def test_apply_never_rides_in_on_the_invocation_line() -> None:
    """Guarded here because this file is where the sweep step is parsed.

    This assertion used to be ``"--apply" not in run`` — "OMN-16106 is DRY-RUN
    until arming is a deliberate, separate change". OMN-16106 IS that change,
    so the absolute form is retired: it would now fail on the very edit it was
    written to wait for, which is not the same thing as guarding against an
    accident.

    What it guards instead is the property that survives arming, and it is the
    stricter half of the original: the flag must never appear on the
    ``onex skill`` invocation line, because a guarded flag and an unconditional
    one read identically in a diff. ``--apply`` may only enter through the
    array the ``SWEEP_APPLY`` branch populates. The full apply contract — the
    input's ``default: false``, the two-conjunct guard, the schedule path's
    inability to reach it, and the mode labelling — is pinned in
    ``tests/ci/test_evidence_autoclose_sweep_apply_input.py``.
    """
    run = str(_sweep_step().get("run", ""))
    invocation_lines = [
        line for line in run.splitlines() if "skill evidence_autoclose_sweep" in line
    ]
    assert invocation_lines, "could not find the sweep invocation line"
    for line in invocation_lines:
        assert "--apply" not in line, (
            "--apply must never be literal on the invocation line; it may only "
            f"be appended by the guarded branch. Offending line: {line!r}"
        )
