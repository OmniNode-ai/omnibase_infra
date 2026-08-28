# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16845 — the sweep job must export a usable ``OMNI_HOME``.

``EvidenceCollector._resolve_cwd`` (omnimarket) substitutes ``${OMNI_HOME}``
from the process environment and requires the rendered path to exist. This
job never exported it, so every check pinned ``cwd: "${OMNI_HOME}/<repo>"``
rendered to ``/<repo>`` and was recorded FAILED — proven by execution:
``env -u OMNI_HOME`` against this job's env shape reproduces
``failed: 3, behavior_proving_count: 0``; the identical run with a real
``OMNI_HOME`` reproduces ``verified, failed: 0, behavior_proving_count: 3``
(see OMN-16845).

This is a wiring assertion, not a behaviour assertion, for the same reason
``test_omn_16792_kill_switch_wiring.py`` is: the failure it guards ("the
checker reads an env var this job never sets") is invisible to any
handler-level test and only shows up by parsing the committed workflow,
which IS the wiring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

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


def _dod_verify_invoking_indices(steps: list[dict[str, Any]]) -> list[int]:
    """Steps that spawn ``uv run onex skill dod_verify`` (directly or via the
    sweep, which shells out to it per ticket) — found by content, not name,
    so an insertion or rename cannot silently retarget this test.
    """
    indices = []
    for i, step in enumerate(steps):
        run = str(step.get("run", ""))
        if (
            "onex skill dod_verify" in run
            or "onex skill evidence_autoclose_sweep" in run
        ):
            indices.append(i)
    return indices


def _omni_home_export_index(steps: list[dict[str, Any]]) -> int | None:
    """Index of the step that writes ``OMNI_HOME=`` into ``$GITHUB_ENV``.

    Must be an actual env-file export (persists to every later step),
    not a step-local ``env:`` value — a step-local value would not reach
    the per-ticket ``uv run onex skill dod_verify`` subprocess the sweep
    step spawns internally.
    """
    for i, step in enumerate(steps):
        run = str(step.get("run", ""))
        if "OMNI_HOME=" in run and "GITHUB_ENV" in run:
            return i
    return None


def test_a_step_exports_omni_home_to_github_env() -> None:
    steps = _steps()
    index = _omni_home_export_index(steps)
    assert index is not None, (
        "no step in evidence-autoclose-sweep.yml writes OMNI_HOME to "
        "$GITHUB_ENV — EvidenceCollector._resolve_cwd will render "
        "'${OMNI_HOME}/<repo>' to '/<repo>', which never exists, and every "
        "cwd-anchored behavior check will be recorded FAILED (OMN-16845)"
    )


def test_omni_home_is_exported_before_every_dod_verify_invocation() -> None:
    steps = _steps()
    export_index = _omni_home_export_index(steps)
    assert export_index is not None
    invoking_indices = _dod_verify_invoking_indices(steps)
    assert invoking_indices, (
        "expected at least one step invoking dod_verify or the sweep — "
        "found none, so this test is not actually anchored to the job"
    )
    for i in invoking_indices:
        assert export_index < i, (
            f"step {i} ({steps[i].get('name')!r}) runs dod_verify/the sweep "
            f"before OMNI_HOME is exported at step {export_index} "
            f"({steps[export_index].get('name')!r}) — GITHUB_ENV writes only "
            "take effect for steps that run AFTER the write"
        )


def test_omni_home_export_is_fail_fast_not_a_silent_default() -> None:
    """The export step must assert its own precondition rather than writing
    a best-guess path — a silent default here would just move the bug from
    'never set' to 'set wrong and never checked' (CLAUDE.md rule 8)."""
    steps = _steps()
    index = _omni_home_export_index(steps)
    assert index is not None
    run = str(steps[index].get("run", ""))
    assert "exit 1" in run and ("::error::" in run), (
        "the OMNI_HOME export step has no fail-fast assertion — it must "
        "verify the derived path actually places the expected repo "
        "checkout under it before exporting, and error loudly (exit 1) if "
        "not, rather than exporting an unverified guess"
    )
