# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16846 — the dispatch venv and the gate venv must be separate.

## The collision this closes

``onex skill dod_verify`` is provided by ``node_dod_verify``, which ships in
**omnimarket**. omnimarket sits ABOVE omnibase_infra in the layer graph, so it
is never a lock dependency — the sweep job co-installs it as a runtime
provider (OMN-16736). That co-install is correct and required.

The behaviour checks dod_verify then runs are ``uv run pytest ...`` with
``cwd: "${OMNI_HOME}/omnibase_infra"`` (see the OCC contracts for OMN-16759 /
16790 / 16784 / 16785). ``tests/conftest.py::pytest_configure`` calls
``assert_venv_purity()`` (OMN-15620), which refuses any venv carrying an
``onex.nodes`` provider that ``uv.lock`` does not declare — i.e. exactly the
co-installed omnimarket.

Before this fix both roles landed on ONE venv: the sweep co-installed
omnimarket into ``${GITHUB_WORKSPACE}/.venv``, and every behaviour check's
``uv run pytest`` resolved that same project venv. Run 33194402437 recorded
all three OMN-16759 behaviour checks as::

    [failed] dod-16759-ac1-ac2-supersede-assert: FAILED (8592ms):
      Exit: OMN-15620 venv-purity gate: Canonical venv is IMPURE ...
    status=failed verified=1 failed=3 behavior_proving=0

The check never executed. Neither side is wrong on its own — the purity gate
is correct and the co-install is required — so the defect is the collapse of
two environments into one.

## What is fixed, and why the handler is part of it

Separating the venvs in the workflow is necessary but NOT sufficient, because
the sweep hard-coded its verifier dispatch as ``["uv", "run", "onex", "skill",
"dod_verify", ticket]``. ``uv run`` re-resolves the project at the process's
cwd, which is the product clone — i.e. the GATE venv. So with the venvs split
the verifier would resolve the pure venv and die on ``Unknown node
'node_dod_verify'``, which is the exact failure OMN-16736 fixed.

The sweep must dispatch the verifier from the environment it was ITSELF
composed in (``sys.executable``'s own ``onex``), not re-resolve one from a
directory. That also removes the implicit coupling "the sweep's cwd decides
which venv the verifier runs in", which is what made the two roles
inseparable in the first place.

Proven by execution on 2026-08-28, same ticket, same contract, only the
environment shape differing:

* collapsed (impure venv is both dispatch and gate):
  ``status=failed verified=1 failed=3 behavior_proving=0``
* separated (impure dispatch venv, pure gate venv at ``${OMNI_HOME}``):
  ``status=verified verified=5 failed=0 behavior_proving=3``

with no ``ONEX_ALLOW_VENV_IMPURITY`` set anywhere and no gate weakened.

The workflow half is asserted by parsing the committed workflow, for the same
reason ``test_omn_16845_omni_home_wiring.py`` and
``test_omn_16792_kill_switch_wiring.py`` do it: a job that composes the wrong
venv is invisible to every handler-level test, and the wiring IS the fix.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    HandlerEvidenceAutocloseSweep,
    _dod_verify_argv,
)

pytestmark = pytest.mark.unit

_WORKFLOW = (
    Path(__file__).resolve().parents[4]
    / ".github"
    / "workflows"
    / "evidence-autoclose-sweep.yml"
)


def _steps() -> list[dict[str, Any]]:
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    return list(workflow["jobs"]["evidence-autoclose-sweep"]["steps"])


def _index_of_step_running(*needles: str) -> int | None:
    """First step whose ``run`` body contains every needle."""
    for i, step in enumerate(_steps()):
        run = str(step.get("run", ""))
        if all(needle in run for needle in needles):
            return i
    return None


def _dispatch_indices(steps: list[dict[str, Any]]) -> list[int]:
    """Steps that dispatch the verifier or the sweep, matched on the skill
    name so a change in how the binary is spelled cannot un-anchor this."""
    return [
        i
        for i, step in enumerate(steps)
        if "skill dod_verify" in str(step.get("run", ""))
        or "skill evidence_autoclose_sweep" in str(step.get("run", ""))
    ]


# --------------------------------------------------------------------------
# Handler: the verifier is dispatched from the sweep's OWN environment
# --------------------------------------------------------------------------


def test_dod_verify_argv_uses_the_sweep_interpreters_own_onex() -> None:
    argv = _dod_verify_argv("OMN-16759")

    assert argv[0] == str(Path(sys.executable).parent / "onex"), (
        "the verifier must be dispatched from the interpreter the sweep is "
        "itself running in, so the sweep's dispatch venv (which carries the "
        "co-installed omnimarket) is decided by composition rather than by "
        "whatever project the process cwd happens to resolve to"
    )
    assert argv[1:] == ["skill", "dod_verify", "OMN-16759"]


def test_dod_verify_argv_does_not_shell_out_through_uv_run() -> None:
    argv = _dod_verify_argv("OMN-16759")

    assert "uv" not in argv, (
        "`uv run onex skill dod_verify` re-resolves the project venv at the "
        "process cwd — the product clone, i.e. the GATE venv. That coupling "
        "is what forces the omnimarket co-install into the same venv the "
        "behaviour checks run pytest in, where the OMN-15620 purity gate "
        "correctly refuses it (run 33194402437: failed=3, "
        "behavior_proving=0)"
    )


async def test_real_runner_execs_the_resolved_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wiring, not just the helper: ``_run_dod_verify_command_real`` must
    exec what ``_dod_verify_argv`` resolves."""
    captured: list[tuple[str, ...]] = []

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"{}", b""

    async def _fake_exec(*args: str, **kwargs: Any) -> _FakeProc:
        captured.append(args)
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    handler = HandlerEvidenceAutocloseSweep(linear_client=None)
    await handler._run_dod_verify_command_real("OMN-16759", cwd="", timeout=5.0)

    assert captured, "the real runner never spawned a subprocess"
    assert list(captured[0]) == _dod_verify_argv("OMN-16759")


# --------------------------------------------------------------------------
# Workflow: two venvs, composed separately, both asserted
# --------------------------------------------------------------------------


def test_a_separate_dispatch_venv_is_composed() -> None:
    index = _index_of_step_running("UV_PROJECT_ENVIRONMENT", "uv sync")

    assert index is not None, (
        "no step composes a dispatch venv separate from the project's own "
        "`.venv`. Without one, the omnimarket co-install lands in the same "
        "venv the behaviour checks run `uv run pytest` in, and the OMN-15620 "
        "purity gate refuses every one of them before it executes"
    )


def test_the_omnimarket_coinstall_does_not_target_the_gate_venv() -> None:
    steps = _steps()
    coinstall = [
        i
        for i, step in enumerate(steps)
        if "uv pip install" in str(step.get("run", ""))
        and ".omnimarket-src" in str(step.get("run", ""))
    ]

    assert coinstall, "no omnimarket co-install step found"
    for i in coinstall:
        run = str(steps[i].get("run", ""))
        assert ".venv/bin/python" not in run, (
            f"step {i} ({steps[i].get('name')!r}) co-installs omnimarket into "
            "the project's own `.venv` — that is the venv every behaviour "
            "check's `uv run pytest` resolves, and an undeclared "
            "`onex.nodes` provider there is exactly what the OMN-15620 gate "
            "refuses. Install it into the dispatch venv instead"
        )
        assert "DISPATCH_VENV" in run, (
            f"step {i} ({steps[i].get('name')!r}) does not name the dispatch "
            "venv as its install target"
        )


def test_every_dispatch_runs_through_the_dispatch_venv_not_uv_run() -> None:
    steps = _steps()
    indices = _dispatch_indices(steps)

    assert indices, (
        "expected at least one step dispatching dod_verify or the sweep — "
        "found none, so this test is not anchored to the job"
    )
    for i in indices:
        run = str(steps[i].get("run", ""))
        assert "uv run onex" not in run, (
            f"step {i} ({steps[i].get('name')!r}) dispatches through "
            "`uv run onex`, which resolves the project venv at the step's "
            "cwd — the GATE venv, which by design does not carry omnimarket. "
            "Every dispatch would fail closed on `Unknown node "
            "'node_dod_verify'` (the OMN-16736 regression)"
        )
        assert "DISPATCH_VENV" in run, (
            f"step {i} ({steps[i].get('name')!r}) does not dispatch through "
            "the composed dispatch venv"
        )


def test_gate_venv_purity_is_asserted_before_the_sweep_runs() -> None:
    """CLAUDE.md rule 5: the separation is enforced, not merely intended.

    Without this the failure mode is silent and mislabelled — a re-polluted
    gate venv produces N per-ticket FAILED checks that read like N unproven
    tickets rather than one environment fault.
    """
    steps = _steps()
    purity_index = _index_of_step_running("find_undeclared_onex_providers")

    assert purity_index is not None, (
        "no step asserts that the gate venv is free of undeclared "
        "`onex.nodes` providers. The separation this PR makes would then be "
        "unenforced: any later step that pip-installs a sibling into `.venv` "
        "silently reintroduces the collision"
    )
    run = str(steps[purity_index].get("run", ""))
    assert "exit 1" in run and "::error::" in run, (
        "the gate-venv purity assertion does not fail fast — a check that "
        "only ever logs is advisory, and advisory checks get ignored"
    )
    for i in _dispatch_indices(steps):
        assert purity_index < i, (
            f"step {i} ({steps[i].get('name')!r}) dispatches before the "
            f"gate-venv purity assertion at step {purity_index} — the "
            "assertion must run first or it cannot prevent anything"
        )
