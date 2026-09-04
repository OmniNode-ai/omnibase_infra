# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16106 — the ``--apply`` path, and the guards that keep it honest.

``evidence-autoclose-sweep.yml`` shipped DRY-RUN only; a later slice made
``--apply`` reachable from an explicit ``workflow_dispatch``. This revision
makes the **schedule** the applying path, on the operator ruling of
2026-09-04 (~23:05Z, firm): the evidence closer closes tickets on its own —
when a ticket's definition of done is met by bound evidence it is Done, with
no human approval step, because that is the reason the closer exists. A
closer that only previews on its schedule and mutates only when a human
happens to tick a box is a manual triage sweep wearing a cron trigger.

So the property this file pins has INVERTED, deliberately, and the inversion
is the whole point of reading it:

1. **The 30-minute schedule applies.** ``schedule`` is the unattended path and
   it is now the one that writes. What bounds it is not the trigger — it is
   the flip predicate (untouched by this change and not this file's to set),
   the per-candidate exclusion fence (OMN-17891), and the
   ``ONEX_AUTOCLOSE_DISABLED`` kill switch. All three are asserted below to
   still bind a scheduled run.

2. **A dispatch can still force a dry run.** The diagnostic path has to
   survive: ``workflow_dispatch`` with ``apply`` left unticked reaches every
   decision and writes none of them, which is what makes a rehearsal of a
   fence list possible. ``apply`` therefore keeps ``default: false`` and keeps
   meaning "this dispatch writes", while a cron tick — which carries no inputs
   at all — is decided by its event name alone.

3. **The flag stays conditional.** ``--apply`` appended straight onto the
   ``onex skill`` invocation reads identically to the guarded form in a diff.
   So the invocation line is asserted to carry no literal ``--apply`` at all:
   the flag may only enter through the shell array the guard populates.

4. **The fence is reachable without a dispatcher.** A per-dispatch input
   cannot fence an unattended run — the same argument OMN-16792 made about the
   kill switch, which is a repo variable for exactly this reason. With the
   schedule applying, an exclusion list that only a human can type is a fence
   that is absent precisely when nobody is watching. So the standing fence
   arrives from ``vars.ONEX_AUTOCLOSE_EXCLUDE`` and is UNIONED with any
   dispatch-supplied list: a dispatch may add to the standing fence, never
   shrink it.

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
EXCLUDE_INPUT = "exclude"
SWEEP_STEP_NAME = "Run evidence autoclose sweep"
SUMMARY_STEP_NAME = "Post summary"
# The shell variable carrying the UNION of the standing repo-variable fence
# and any dispatch-supplied exclusion. It is what `--exclude` is given.
FENCE_VAR = "FENCE_TICKETS"

# The one expression that decides whether this run mutates Linear. It is
# written as a single constant here precisely because it appears at more than
# one site in the workflow (the flag, the job name, the summary) and those
# sites MUST agree — a run that labels itself DRY-RUN while passing --apply is
# the defect this constant exists to make impossible.
APPLY_GUARD = (
    "github.event_name == 'schedule' || "
    "(github.event_name == 'workflow_dispatch' && github.event.inputs.apply == 'true')"
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
    """A dispatch that leaves the box alone is the DIAGNOSTIC dry run.

    ``default: false`` is still load-bearing, but for the opposite reason it
    was before: now that the schedule applies, this input is what preserves a
    read-only path at all. Someone rehearsing a fence list, or reading what the
    sweep currently decides without writing it, dispatches and leaves the box
    alone. Flipping this default to ``true`` would delete the dry run.
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


def test_the_schedule_still_fires_every_thirty_minutes() -> None:
    """The unattended trigger is the closer. It has to exist to be one."""
    schedule = _triggers().get("schedule")
    assert isinstance(schedule, list) and schedule, (
        "the scheduled sweep is the closer's only unattended path — without it "
        "nothing closes on its own, which is the whole point (operator ruling, "
        "2026-09-04)"
    )
    crons = [entry.get("cron") for entry in schedule]
    assert "*/30 * * * *" in crons, (
        f"the 30-minute cadence must survive this change; got {crons!r}"
    )


def test_the_scheduled_run_applies() -> None:
    """A cron tick reaches ``--apply``, by its event name alone.

    This is the inversion. ``schedule`` carries no ``github.event.inputs`` at
    all, so an input could never enable it — the guard has to name the event.
    Asserted both ways: the schedule disjunct is present, AND the guard is not
    still the dispatch-only form that made every unattended run read-only.
    """
    assert "github.event_name == 'schedule'" in APPLY_GUARD, (
        "the apply guard must name the schedule event — a closer whose "
        "scheduled run only previews is not closing anything"
    )
    guard = (_job().get("env") or {}).get("SWEEP_APPLY") or ""
    assert APPLY_GUARD in guard, (
        f"SWEEP_APPLY must carry the canonical guard\n  expected: {APPLY_GUARD}\n"
        f"  got: {guard}"
    )
    # The pre-ruling form, spelled out so a revert to it fails loudly rather
    # than silently returning the closer to preview-only.
    dispatch_only = (
        "${{ (github.event_name == 'workflow_dispatch' && "
        "github.event.inputs.apply == 'true') && 'true' || 'false' }}"
    )
    assert guard != dispatch_only, (
        "SWEEP_APPLY is back to the dispatch-only form: scheduled runs would "
        "preview forever and nothing would ever close unattended"
    )


def test_a_dispatch_can_still_force_a_dry_run() -> None:
    """The diagnostic path survives: dispatch, leave ``apply`` unticked.

    The guard's dispatch branch requires the input to be literally ``'true'``,
    so a dispatch that leaves the default alone evaluates the whole expression
    to false and the run writes nothing. That is asserted structurally — the
    dispatch branch is a CONJUNCTION on the input, not a bare event-name test
    that would make every dispatch apply too and leave no read-only path at
    all.
    """
    assert (
        "(github.event_name == 'workflow_dispatch' && "
        "github.event.inputs.apply == 'true')"
    ) in APPLY_GUARD, (
        "the dispatch branch must stay conjoined with the apply input, or a "
        "dispatch could no longer request a dry run and the rehearsal path "
        f"would be gone; got {APPLY_GUARD!r}"
    )
    inputs = (_triggers().get("workflow_dispatch") or {}).get("inputs") or {}
    assert inputs.get(APPLY_INPUT, {}).get("default") is False, (
        "with the schedule applying, this default is the only thing that keeps "
        "a read-only dispatch reachable"
    )


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


def test_the_exclusion_fence_is_a_dispatch_input_that_defaults_to_empty() -> None:
    """OMN-17891 — the per-candidate fence the apply path had no way to express.

    ``default: ''`` is the load-bearing half in the opposite direction from
    ``apply``'s: a run that names nothing excludes nothing, so the field cannot
    quietly shrink a scheduled sweep's coverage.
    """
    inputs = (_triggers().get("workflow_dispatch") or {}).get("inputs") or {}
    assert EXCLUDE_INPUT in inputs, (
        f"workflow_dispatch must declare an '{EXCLUDE_INPUT}' input — without "
        "one, an apply dispatch can only choose a companion-merge WINDOW, "
        "never decline a named candidate inside it"
    )
    spec = inputs[EXCLUDE_INPUT]
    assert spec.get("default") == "", (
        f"'{EXCLUDE_INPUT}' must default to the empty string; got "
        f"{spec.get('default')!r}"
    )
    assert spec.get("description"), (
        f"'{EXCLUDE_INPUT}' must carry a description — it is a caller "
        "ASSERTION the sweep cannot verify, so the dispatcher has to be told "
        "they own its accuracy"
    )


def test_the_exclusion_reaches_the_sweep_only_through_a_guarded_array() -> None:
    """``--exclude`` is conditional in the shell, never literal on the line.

    Same property as ``--apply``: a guard plus an unconditional flag reads
    identically to a guarded one in a diff, so the invocation is asserted to
    carry no literal ``--exclude`` at all. The value itself arrives via ``env``
    (asserted by ``test_no_input_is_interpolated_directly_into_the_run_script``,
    which forbids every ``${{ }}`` in this run body).
    """
    step = _step(SWEEP_STEP_NAME)
    env = step.get("env") or {}
    assert env.get("EXCLUDE_TICKETS") == "${{ github.event.inputs.exclude || '' }}", (
        "the exclusion list must reach the shell as an env var, not as "
        f"expression substitution; got {env.get('EXCLUDE_TICKETS')!r}"
    )

    run = step.get("run") or ""
    assert f'if [ -n "${{{FENCE_VAR}}}" ]' in run, (
        f"the run script must branch on {FENCE_VAR} being non-empty"
    )
    invocation_lines = [
        line
        for line in run.splitlines()
        if "onex" in line and "evidence_autoclose_sweep" in line
    ]
    assert invocation_lines, "could not find the `onex skill` invocation"
    for line in invocation_lines:
        assert "--exclude" not in line, (
            f"--exclude must never appear on the invocation line itself: {line!r}"
        )


def test_the_standing_fence_reaches_an_unattended_run() -> None:
    """OMN-16106 — a fence only a dispatcher can type is absent on the schedule.

    This is the same argument OMN-16792 made about the kill switch and for the
    same reason: ``github.event.inputs`` is null on a cron tick, so the
    OMN-17891 exclusion input — the only per-candidate refusal that exists —
    can never bind the runs nobody dispatched. Once the schedule applies, that
    is a fence that is missing exactly when no operator is watching.

    A repo variable is reachable from every event, so the standing list arrives
    that way. It is deliberately a variable and not an input, for the same
    reason the kill switch is.
    """
    env = _step(SWEEP_STEP_NAME).get("env") or {}
    assert env.get("STANDING_EXCLUDE_TICKETS") == (
        "${{ vars.ONEX_AUTOCLOSE_EXCLUDE }}"
    ), (
        "the standing fence must be read from a repo variable so it binds "
        "scheduled runs; got "
        f"{env.get('STANDING_EXCLUDE_TICKETS')!r}"
    )
    inputs = (_triggers().get("workflow_dispatch") or {}).get("inputs") or {}
    assert "ONEX_AUTOCLOSE_EXCLUDE" not in inputs, (
        "the standing fence must not be a dispatch input — it exists to bind "
        "runs nobody dispatched"
    )


def test_a_dispatch_can_add_to_the_standing_fence_but_never_shrink_it() -> None:
    """The two lists are UNIONED, and the union is what reaches ``--exclude``.

    If the dispatch input replaced the variable, a dispatch naming one ticket
    would silently drop every standing exclusion — a fence that gets smaller
    the more precisely you aim it. Asserted structurally: the fence value is
    built from BOTH names, and the ``--exclude`` argument carries that built
    value rather than either input on its own.
    """
    run = _step(SWEEP_STEP_NAME).get("run") or ""
    build_lines = [line for line in run.splitlines() if f"{FENCE_VAR}=" in line]
    assert build_lines, f"could not find where {FENCE_VAR} is built"
    built_from = (
        "\n".join(build_lines)
        + "\n"
        + "\n".join(
            line
            for line in run.splitlines()
            if "STANDING_EXCLUDE_TICKETS" in line or "EXCLUDE_TICKETS}" in line
        )
    )
    assert "STANDING_EXCLUDE_TICKETS" in built_from, (
        "the fence value must be built from the standing repo variable"
    )
    assert "EXCLUDE_TICKETS" in built_from, (
        "the fence value must also be built from the dispatch input, so an "
        "operator can add a candidate to a scheduled run's refusals"
    )
    assert f'exclude_args+=(--exclude "${{{FENCE_VAR}}}")' in run, (
        f"--exclude must carry the UNIONED {FENCE_VAR}, not either list alone"
    )
    assert 'exclude_args+=(--exclude "${EXCLUDE_TICKETS}")' not in run, (
        "--exclude must not carry the dispatch input alone: a dispatch would "
        "then shrink the standing fence rather than add to it"
    )


def test_the_fence_binds_the_applying_run_at_the_same_depth_as_apply() -> None:
    """The fence is not nested inside — nor gated on — the apply branch.

    Now that the schedule applies, this property carries more weight than it
    did as a rehearsal argument: a fence evaluated inside the apply branch
    would still work, but a fence evaluated inside a DISPATCH-only branch
    would leave every scheduled write unfenced. Asserted as sibling nesting,
    the same structural check as before.
    """
    run = _step(SWEEP_STEP_NAME).get("run") or ""
    fence_guard = f'if [ -n "${{{FENCE_VAR}}}" ]'
    apply_guard = 'if [ "${SWEEP_APPLY}" = "true" ]'
    assert fence_guard in run and apply_guard in run
    fence_indent = next(
        len(line) - len(line.lstrip())
        for line in run.splitlines()
        if fence_guard in line
    )
    apply_indent = next(
        len(line) - len(line.lstrip())
        for line in run.splitlines()
        if apply_guard in line
    )
    assert fence_indent == apply_indent, (
        "the fence guard must sit at the same nesting depth as the apply guard "
        f"(sibling, not nested): {fence_indent} vs {apply_indent}"
    )


def test_the_kill_switch_still_dominates_the_apply_path() -> None:
    """ONEX_AUTOCLOSE_DISABLED stays wired, and stays out of the inputs.

    The kill switch has to bind every run, which now means every SCHEDULED run
    — it is the only way to stop an applying closer without editing this file,
    and a cron tick carries no inputs to opt out with. That is exactly why it
    is a repo variable plumbed into the step env (OMN-16792 AC3) and not a
    per-dispatch input.
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
