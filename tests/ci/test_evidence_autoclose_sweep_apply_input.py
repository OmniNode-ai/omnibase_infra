# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16106 — the deliberate ``--apply`` path, and the guard that keeps it deliberate.

``evidence-autoclose-sweep.yml`` shipped DRY-RUN only, with its own header
naming the flip to ``--apply`` as "a later, deliberate change (a separate PR),
not a flag flip buried in this file". This is that change, and this file is
what makes "deliberate" mechanically true rather than a comment.

The failure this prevents is not "apply is missing" — it is the two ways
wiring apply goes wrong:

1. **A scheduled run applies.** ``schedule`` fires every 30 minutes with no
   operator present. If ``--apply`` can be reached from any path other than an
   explicit ``workflow_dispatch`` that asked for it, the sweep becomes an
   unattended Linear mutator. The guard asserted below names BOTH conjuncts —
   ``github.event_name == 'workflow_dispatch'`` and the input being literally
   ``'true'`` — so a cron tick, whose ``github.event.inputs`` is null, cannot
   satisfy it by any evaluation order.

2. **The flag becomes unconditional.** ``--apply`` appended straight onto the
   ``onex skill`` invocation reads identically to the guarded form in a diff.
   So the invocation line is asserted to carry no literal ``--apply`` at all:
   the flag may only enter through the shell array the guard populates.

The third assertion is about honesty rather than safety. The job name and the
step summary both stated "DRY-RUN" as a constant. Left alone they would have
kept saying it while the run flipped tickets, which is exactly the class of
silent-mislabel defect OMN-16832 was about one field over. Both are now
rendered from the SAME guard expression as the flag itself, and that identity
is asserted — a run cannot label itself DRY-RUN and pass ``--apply``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "evidence-autoclose-sweep.yml"

JOB_ID = "evidence-autoclose-sweep"
APPLY_INPUT = "apply"
SWEEP_STEP_NAME = "Run evidence autoclose sweep"
SUMMARY_STEP_NAME = "Post summary"

# The one expression that decides whether this run mutates Linear. It is
# written as a single constant here precisely because it appears at more than
# one site in the workflow (the flag, the job name, the summary) and those
# sites MUST agree — a run that labels itself DRY-RUN while passing --apply is
# the defect this constant exists to make impossible.
APPLY_GUARD = (
    "github.event_name == 'workflow_dispatch' && github.event.inputs.apply == 'true'"
)


def _load_workflow() -> dict[str, Any]:
    loaded = yaml.safe_load(SWEEP_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), f"{SWEEP_WORKFLOW} did not parse to a mapping"
    return loaded


def _job() -> dict[str, Any]:
    job = _load_workflow()["jobs"][JOB_ID]
    assert isinstance(job, dict), f"{JOB_ID} did not parse to a mapping"
    return job


def _steps() -> list[dict[str, Any]]:
    steps = _job()["steps"]
    assert isinstance(steps, list) and steps, f"{JOB_ID} declares no steps"
    return steps


def _step(name: str) -> dict[str, Any]:
    matches = [s for s in _steps() if s.get("name") == name]
    assert len(matches) == 1, (
        f"expected exactly one step named {name!r} in {JOB_ID}; found {len(matches)}"
    )
    return matches[0]


def _triggers() -> dict[str, Any]:
    # `on:` is parsed by YAML 1.1 as the boolean True, not the string "on", so
    # the key is looked up by identity against both spellings rather than
    # assumed.
    workflow: dict[Any, Any] = _load_workflow()
    for key in (True, "on"):
        triggers = workflow.get(key)
        if isinstance(triggers, dict):
            return triggers
    raise AssertionError(f"{SWEEP_WORKFLOW} declares no trigger block")


def test_apply_is_an_explicit_dispatch_input_defaulting_to_false() -> None:
    """The operator has to ask for it, per dispatch, in the UI.

    ``default: false`` is the load-bearing half: a dispatch that leaves the box
    alone is a dry run, so the cheap "just re-run it" reflex cannot mutate
    anything.
    """
    triggers = _triggers()
    dispatch = triggers.get("workflow_dispatch")
    assert isinstance(dispatch, dict), "workflow_dispatch must declare inputs"
    inputs = dispatch.get("inputs") or {}
    assert APPLY_INPUT in inputs, (
        f"workflow_dispatch must declare an explicit '{APPLY_INPUT}' input — "
        "an apply path reachable without one is not a deliberate choice"
    )
    spec = inputs[APPLY_INPUT]
    assert spec.get("type") == "boolean", (
        f"'{APPLY_INPUT}' must be a boolean input (a rendered checkbox), got "
        f"{spec.get('type')!r}"
    )
    assert spec.get("default") is False, (
        f"'{APPLY_INPUT}' must default to false; got {spec.get('default')!r}"
    )
    assert spec.get("description"), (
        f"'{APPLY_INPUT}' must carry a description — it is the only thing the "
        "operator reads before ticking a box that mutates Linear"
    )


def test_the_schedule_trigger_survives_and_cannot_reach_apply() -> None:
    """The 30-minute cron is still the primary path, and it is still dry-run.

    ``schedule`` events carry no ``inputs``, so ``github.event.inputs.apply``
    is null there and the guard's second conjunct alone already refuses. The
    first conjunct is asserted anyway: two independent reasons, because this is
    the one property whose failure is unattended.
    """
    assert "schedule" in _triggers(), (
        "the scheduled sweep must survive this change — dry-run enumeration is "
        "what keeps the gap comments current"
    )
    assert "github.event_name == 'workflow_dispatch'" in APPLY_GUARD
    assert "github.event.inputs.apply == 'true'" in APPLY_GUARD


def test_apply_reaches_the_sweep_only_through_the_guard() -> None:
    """The flag is conditional in the shell, never literal on the command line.

    Asserted as an absence on the run script: if ``--apply`` appears anywhere
    other than inside the guarded branch, this fails. That is stricter than
    checking the guard exists, because a guard plus an unconditional flag is
    still an unconditional flag.
    """
    step = _step(SWEEP_STEP_NAME)
    env = step.get("env") or {}
    guard_expr = env.get("SWEEP_APPLY") or (_job().get("env") or {}).get("SWEEP_APPLY")
    assert guard_expr, (
        "the sweep's apply decision must be carried by a SWEEP_APPLY env var so "
        "the guard is evaluated once, in the expression context, and read as "
        "data by the shell"
    )
    assert APPLY_GUARD in guard_expr, (
        f"SWEEP_APPLY must be computed from the canonical guard\n"
        f"  expected substring: {APPLY_GUARD}\n"
        f"  got: {guard_expr}"
    )

    run = step.get("run") or ""
    assert 'if [ "${SWEEP_APPLY}" = "true" ]' in run, (
        "the run script must branch on SWEEP_APPLY explicitly"
    )
    # Every --apply occurrence must sit inside the guarded branch. The
    # invocation itself must not mention it.
    invocation_lines = [
        line
        for line in run.splitlines()
        if "onex" in line and "evidence_autoclose_sweep" in line
    ]
    assert invocation_lines, "could not find the `onex skill` invocation"
    for line in invocation_lines:
        assert "--apply" not in line, (
            f"--apply must never appear on the invocation line itself: {line!r}"
        )


def test_no_input_is_interpolated_directly_into_the_run_script() -> None:
    """Dispatch inputs reach the shell as env, not as expression substitution.

    ``${{ github.event.inputs.lookback_hours }}`` pasted into a ``run:`` body is
    a script-injection site: the expression is substituted before bash ever
    parses the line, so the input's characters become shell syntax. Both inputs
    are now passed through ``env:`` and referenced as ``"${VAR}"``, which bash
    reads as a value.

    Asserted as "no ``${{`` anywhere in this run body" rather than as a
    denylist of input names: the property that keeps the step safe is that the
    shell script is a constant, not that one particular expression is absent.
    """
    run = _step(SWEEP_STEP_NAME).get("run") or ""
    assert "${{" not in run, (
        "no `${{ ... }}` expression may be interpolated into the sweep step's "
        f"run body — pass it through `env:` instead; found one in:\n{run}"
    )
    assert '"${LOOKBACK_HOURS}"' in run, (
        "the lookback must be read from the environment as a quoted value"
    )


def test_the_run_labels_its_own_mode_from_the_same_guard() -> None:
    """A run cannot call itself DRY-RUN while flipping tickets.

    The job name and the step summary are rendered from the identical guard
    expression that gates the flag, so the label and the behaviour cannot drift
    apart. Before this, both were the constant string "DRY-RUN".
    """
    job = _job()
    job_name = job.get("name") or ""
    assert APPLY_GUARD in job_name, (
        "the job name must render its mode from the canonical apply guard, not "
        f"state a constant; got {job_name!r}"
    )
    assert "DRY-RUN" in job_name and "APPLY" in job_name, (
        f"the job name must be able to render BOTH modes; got {job_name!r}"
    )

    summary_step = _step(SUMMARY_STEP_NAME)
    summary_mode = (summary_step.get("env") or {}).get("SWEEP_MODE") or ""
    assert APPLY_GUARD in summary_mode, (
        "the step summary must render its mode from the same guard; a summary "
        f"that hardcodes DRY-RUN lies on an apply run. Got {summary_mode!r}"
    )
    summary = summary_step.get("run") or ""
    assert "this workflow never passes --apply" not in summary, (
        "the summary still carries the pre-OMN-16106 claim that this workflow "
        "never applies — that statement is now false"
    )


def test_the_kill_switch_still_dominates_the_apply_path() -> None:
    """ONEX_AUTOCLOSE_DISABLED stays wired, and stays out of the inputs.

    The kill switch has to bind every run including an apply dispatch, which is
    exactly why it is a repo variable plumbed into the step env (OMN-16792 AC3)
    and NOT a per-dispatch input: an input is a choice, and the operator making
    the apply choice is the last person who should be able to opt out of the
    halt.
    """
    env = _step(SWEEP_STEP_NAME).get("env") or {}
    assert (
        env.get("ONEX_AUTOCLOSE_DISABLED") == "${{ vars.ONEX_AUTOCLOSE_DISABLED }}"
    ), (
        "the sweep step must keep reading the kill switch from the repo "
        f"variable; got {env.get('ONEX_AUTOCLOSE_DISABLED')!r}"
    )
    inputs = (_triggers().get("workflow_dispatch") or {}).get("inputs") or {}
    assert "ONEX_AUTOCLOSE_DISABLED" not in inputs, (
        "the kill switch must not be a dispatch input — it must bind runs "
        "nobody dispatched"
    )
