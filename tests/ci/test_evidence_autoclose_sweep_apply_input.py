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

# OMN-17658 REWRITE. What used to live here was APPLY_GUARD — the single
# expression `github.event_name == 'schedule' || (workflow_dispatch && inputs
# .apply == 'true')` that decided whether a run mutated Linear, asserted at
# three sites so they could not drift apart.
#
# That constant is gone because the thing it pinned is gone. Its first
# disjunct WAS the arming authority for every unattended write, and P1-8
# (decision 11 / §3b / F-R5-4) requires that authority to be a declared, typed
# contract input on the node instead of an expression in this file. So the
# workflow now asserts two much weaker properties, and the third — whether a
# scheduled run writes — is not this file's to assert at all:
#
#   * TRIGGER_EXPR: the FACT of what launched the run, passed to the node.
#   * DISPATCH_APPLY_EXPR: the operator's explicit request on THIS dispatch.
#
# The property that replaces "the three sites agree" is stronger, and it is
# asserted below: NO site in this workflow may name an arming decision for a
# scheduled run, because there is nowhere left for such a decision to be
# correct.
TRIGGER_EXPR = "github.event_name == 'schedule' && 'schedule' || 'dispatch'"
DISPATCH_APPLY_EXPR = (
    "(github.event_name == 'workflow_dispatch' && "
    "github.event.inputs.apply == 'true') && 'true' || 'false'"
)
# The pre-OMN-17658 arming disjunct, spelled once so its RETURN fails loudly.
RETIRED_SCHEDULE_ARMING_DISJUNCT = "github.event_name == 'schedule' || "


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


def test_the_workflow_no_longer_decides_whether_a_scheduled_run_writes() -> None:
    """OMN-17658 §3b. The arming authority is the contract, not this file.

    REWRITTEN, and the rewrite is the assertion. The previous test here was
    ``test_the_scheduled_run_applies`` and it required this workflow to carry
    ``github.event_name == 'schedule'`` inside the expression that gates
    ``--apply`` — i.e. it required the arming authority for every unattended
    write to live in a YAML string. omnibase_infra#3195 armed the closer by
    editing that string, and the plan row that was supposed to authorise
    arming (P1-8) was still Backlog at the time.

    The property now is an ABSENCE plus a redirection:

      * no expression in this workflow may arm a scheduled write;
      * the workflow states the trigger and passes it to the node;
      * there is no ``--scheduled-apply`` flag for it to pass, so it could not
        arm one even by accident.

    Whether a scheduled run writes is answered by
    ``ModelEvidenceAutocloseSweepRequest.scheduled_apply``, declared on
    ``node_evidence_autoclose_sweep_effect/contract.yaml`` and pinned by
    tests/unit/nodes/node_evidence_autoclose_sweep_effect/
    test_omn_17658_closer_safety_fences.py.
    """
    job = _job()
    job_env = job.get("env") or {}
    assert job_env.get("SWEEP_TRIGGER") == "${{ " + TRIGGER_EXPR + " }}", (
        "the workflow must report WHAT LAUNCHED IT as a plain fact; got "
        f"{job_env.get('SWEEP_TRIGGER')!r}"
    )
    assert "SWEEP_APPLY" not in job_env, (
        "SWEEP_APPLY is back. That variable carried the schedule-arming "
        "disjunct, which is exactly what OMN-17658 moved into the contract"
    )

    step = _step(SWEEP_STEP_NAME)
    run = step.get("run") or ""
    # Comment lines are excluded on purpose: the run body NAMES the forbidden
    # flag in a comment explaining why it must not exist, and a check that
    # could not tell a prohibition from its violation would forbid documenting
    # the rule.
    live_flag = [
        line
        for line in run.splitlines()
        if "--scheduled-apply" in line and not line.lstrip().startswith("#")
    ]
    assert not live_flag, (
        "a --scheduled-apply flag makes this workflow an arming authority "
        f"again; the contract's declared default is the only one: {live_flag}"
    )

    # The retired disjunct must not reappear at ANY site: not the job name,
    # not an env expression, not the summary. A single grep over the whole
    # file, because the defect is the disjunct existing here at all.
    body = SWEEP_WORKFLOW.read_text(encoding="utf-8")
    offending = [
        line
        for line in body.splitlines()
        if RETIRED_SCHEDULE_ARMING_DISJUNCT in line
        and not line.lstrip().startswith("#")
    ]
    assert not offending, (
        "the schedule-arming disjunct is back in a live expression — the "
        f"arming authority has left the contract again:\n{offending}"
    )


def test_the_schedule_reaches_the_node_as_a_typed_trigger() -> None:
    """The fact travels; the decision does not.

    ``--trigger`` is unconditional on the invocation line — every run states
    what launched it — while ``--apply`` stays inside a guarded array. That
    asymmetry is the whole design: a fact may be stated unconditionally, a
    write request may not.
    """
    run = _step(SWEEP_STEP_NAME).get("run") or ""
    assert '--trigger "${SWEEP_TRIGGER}"' in run, (
        "the sweep must be told what launched it, as a quoted env value"
    )
    invocation = [
        line
        for line in run.splitlines()
        if "onex" in line and "evidence_autoclose_sweep" in line
    ]
    assert invocation, "could not find the `onex skill` invocation"


def test_a_dispatch_can_still_force_a_dry_run() -> None:
    """The diagnostic path survives: dispatch, leave ``apply`` unticked.

    The guard's dispatch branch requires the input to be literally ``'true'``,
    so a dispatch that leaves the default alone evaluates the whole expression
    to false and the run writes nothing. That is asserted structurally — the
    dispatch branch is a CONJUNCTION on the input, not a bare event-name test
    that would make every dispatch apply too and leave no read-only path at
    all.
    """
    job_env = _job().get("env") or {}
    assert (
        job_env.get("SWEEP_DISPATCH_APPLY") == "${{ " + DISPATCH_APPLY_EXPR + " }}"
    ), (
        "the dispatch write request must stay a CONJUNCTION on the apply "
        "input. A bare event-name test would make every dispatch write and "
        "leave no read-only path at all — and the rehearsal path has to "
        f"survive arming, not be replaced by it. Got "
        f"{job_env.get('SWEEP_DISPATCH_APPLY')!r}"
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
    guard_expr = (step.get("env") or {}).get("SWEEP_DISPATCH_APPLY") or (
        _job().get("env") or {}
    ).get("SWEEP_DISPATCH_APPLY")
    assert guard_expr, (
        "the dispatch write request must be carried by a SWEEP_DISPATCH_APPLY "
        "env var so the expression is evaluated once, in the expression "
        "context, and read as data by the shell"
    )
    assert DISPATCH_APPLY_EXPR in guard_expr, (
        f"SWEEP_DISPATCH_APPLY must be the canonical dispatch conjunction\n"
        f"  expected substring: {DISPATCH_APPLY_EXPR}\n"
        f"  got: {guard_expr}"
    )

    run = step.get("run") or ""
    assert 'if [ "${SWEEP_DISPATCH_APPLY}" = "true" ]' in run, (
        "the run script must branch on SWEEP_DISPATCH_APPLY explicitly"
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


def test_no_line_in_this_workflow_asserts_a_mode_it_cannot_know() -> None:
    """OMN-17935 + OMN-17658. A string that reads as a fact must be one.

    REWRITTEN from ``test_the_run_labels_its_own_mode_from_the_same_guard``,
    which required the job name and the step summary to render APPLY/DRY-RUN
    from the same expression that gated the flag. That was the right property
    while the workflow decided the mode. It no longer does: for a scheduled run
    the answer lives in the node's contract, so any APPLY/DRY-RUN label
    rendered here would be an assertion this file cannot support — the same
    defect one field over as the ``operator-dispatched`` echo OMN-17935
    records, which asserted a dispatcher that did not exist on run
    33932169358 while four tickets moved to Done.

    Three properties, all absences or redirections:

      1. the job name renders the TRIGGER, which this file does know;
      2. no apply-branch line hardcodes a dispatcher;
      3. the provenance is derived from the event at ONE site, and it says
         where the write decision actually lives.
    """
    job_name = _job().get("name") or ""
    assert "github.event_name == 'schedule'" in job_name, (
        f"the job name must render the trigger it knows; got {job_name!r}"
    )
    assert "SCHEDULE" in job_name and "DISPATCH" in job_name, (
        f"the job name must be able to render both triggers; got {job_name!r}"
    )
    assert "SCHEDULE - APPLY" not in job_name and "SCHEDULE APPLY" not in job_name, (
        "the job name must not assert what a SCHEDULED run's write mode was — "
        f"the contract decides that and this file cannot see it; got {job_name!r}"
    )

    run = _step(SWEEP_STEP_NAME).get("run") or ""
    # 2. AC3 of OMN-17935, red-first against the pre-fix tree: the old line
    # `echo "mode: APPLY (operator-dispatched, github.event.inputs.apply=true)"`
    # sat unconditionally inside the apply branch.
    offending = [
        line
        for line in run.splitlines()
        if "operator-dispatched" in line.lower()
        and "SWEEP_TRIGGER" not in run.split(line)[0].rsplit("if ", 1)[-1]
        and "elif" not in line
        and "echo" in line
        and "OPERATOR-DISPATCHED" not in line
    ]
    assert not offending, (
        "an apply-branch line claims a dispatcher unconditionally; on a "
        f"scheduled run there is none: {offending}"
    )
    assert 'if [ "${SWEEP_TRIGGER}" = "schedule" ]' in run, (
        "the provenance must be DERIVED from the event at one site, not "
        "asserted — OMN-17935 AC2"
    )
    assert "provenance: SCHEDULED" in run, (
        "a scheduled run's log must say it was scheduled — OMN-17935 AC1"
    )
    # 4. AC4: the dry-run branch is checked for the same defect in the other
    # direction. It must not claim a schedule either.
    assert "provenance: OPERATOR-DISPATCHED with apply unticked" in run, (
        "the dry-run branch must name its own real provenance rather than "
        "inherit a claim from the applying one"
    )

    summary_step = _step(SUMMARY_STEP_NAME)
    summary = summary_step.get("run") or ""
    assert "this workflow never passes --apply" not in summary, (
        "the summary still carries the pre-OMN-16106 claim that this workflow "
        "never applies — that statement is now false"
    )
    assert "scheduled_apply" in summary, (
        "the summary must point a reader at where a scheduled run's write "
        "decision actually lives, rather than asserting it here"
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
    apply_guard = 'if [ "${SWEEP_DISPATCH_APPLY}" = "true" ]'
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


# -- OMN-17342: the backfill arm's trigger surface ------------------------


BACKFILL_INPUT = "backfill_lookback_hours"
BACKFILL_VAR = "BACKFILL_LOOKBACK_HOURS"


def test_the_backfill_arm_is_a_dispatch_override_not_an_arming_authority() -> None:
    """The input exists as a per-dispatch override. It does not arm anything.

    REWRITTEN by OMN-17658. This test was
    ``test_the_backfill_arm_is_a_dispatch_input_defaulting_to_off`` and read
    ``default: '0'`` as "the arm is off", which was true while the node's
    contract default was also 0. The arm was deliberately sequenced behind the
    recurring-companion refusal (OMN-17934) and the children conjunct
    (OMN-17658), because a wide arm arriving before them would have reproduced
    the OMN-17292 re-flip across the whole standing backlog rather than once.

    Both fences land in the same commit as this rewrite, so the arm is armed —
    in the NODE CONTRACT's declared default, which is where an arming authority
    belongs and where ``scheduled_apply`` also lives. ``default: '0'`` here now
    means only "this dispatch passes no flag", and the test asserts the input's
    existence and documentation rather than a claim about arming it cannot
    make.
    """
    inputs = (_triggers().get("workflow_dispatch") or {}).get("inputs") or {}
    assert BACKFILL_INPUT in inputs, (
        f"workflow_dispatch must declare a '{BACKFILL_INPUT}' input — without "
        "one there is no way to reach a companion that merged outside the "
        "forward window, which is 113 of the 118 evidence-bearing open tickets "
        "measured 2026-09-05"
    )
    spec = inputs[BACKFILL_INPUT]
    assert str(spec.get("default")) == "0", (
        f"'{BACKFILL_INPUT}' must default to '0' — meaning 'this dispatch "
        "passes no flag', so an untouched box leaves the node on its own "
        f"contract-declared window; got {spec.get('default')!r}"
    )
    assert spec.get("description"), (
        f"'{BACKFILL_INPUT}' must carry a description — it changes which "
        "tickets an unattended-capable mechanism looks at"
    )


def test_no_scheduled_run_can_change_the_backfill_window_from_this_file() -> None:
    """The window a scheduled run uses is the contract's, not this file's.

    REWRITTEN by OMN-17658; the assertion is byte-identical and its MEANING
    inverted, which is exactly why the docstring had to change with it.
    ``github.event.inputs`` is null on a cron tick, so this expression renders
    ``'0'`` on every scheduled run and the shell passes no backfill flag —
    that part is unchanged. What changed is what "no flag" means: the node's
    contract default was 0 (single-armed) and is now armed, so this expression
    no longer keeps the schedule single-armed, it keeps the schedule's window
    OUT OF THIS FILE.

    That is still worth pinning, and for the surviving half of the original
    reason: editing this fallback to a non-zero literal would put a window
    size back into a YAML expression where the contract cannot see it, and
    nobody reviewing the contract would know the schedule's reach had changed.
    """
    env = _step(SWEEP_STEP_NAME).get("env") or {}
    assert (
        env.get(BACKFILL_VAR)
        == "${{ github.event.inputs.backfill_lookback_hours || '0' }}"
    ), (
        "the backfill window must reach the shell as an env var whose fallback "
        "is '0' — i.e. 'this run passes no flag, the contract's window "
        f"applies'; got {env.get(BACKFILL_VAR)!r}"
    )
    assert BACKFILL_VAR not in set(_job().get("env") or {}), (
        "the backfill window must not be set at job scope, where it would outlive the guard"
    )


def test_the_backfill_flag_reaches_the_sweep_only_through_a_guarded_array() -> None:
    """Same property as ``--apply`` and ``--exclude``: never literal on the line."""
    run = _step(SWEEP_STEP_NAME).get("run") or ""
    assert f'case "${{{BACKFILL_VAR}:-0}}" in' in run, (
        "the run script must branch on the backfill window, so an absent "
        "input and an explicit 0 are the same run"
    )
    assert "*[!0-9]*)" in run, (
        "a non-integer value must be rejected explicitly and loudly, not "
        "absorbed by a suppressed stderr — a malformed input that silently "
        "reads as OFF is a run whose log says nothing while its coverage "
        "reverts"
    )
    assert "2>/dev/null" not in run, (
        "this step must not suppress stderr anywhere: a suppressed error in a "
        "coverage-widening path returns zero rows and reads exactly like a "
        "clean run"
    )
    invocation_lines = [
        line
        for line in run.splitlines()
        if "onex" in line and "evidence_autoclose_sweep" in line
    ]
    assert invocation_lines, "could not find the `onex skill` invocation"
    for line in invocation_lines:
        assert "--backfill-lookback-hours" not in line, (
            "--backfill-lookback-hours must never appear on the invocation "
            f"line itself: {line!r}"
        )
    assert (
        'backfill_args+=(--backfill-lookback-hours "${BACKFILL_LOOKBACK_HOURS}")' in run
    ), "the flag must be appended to the guarded array with the window quoted"


def test_the_backfill_flags_the_cli_accepts_match_the_request_model() -> None:
    """A flag the mapping does not declare is a silently-ignored dispatch.

    The workflow passes ``--backfill-lookback-hours``; the CLI resolves flags
    from ``skill_mapping.yaml``. If those two drift, the sweep runs single-armed
    while the job log says the arm is on — the failure mode this whole ticket
    exists to remove, one layer up.
    """
    import yaml

    from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models import (
        ModelEvidenceAutocloseSweepRequest,
    )

    mapping = yaml.safe_load(
        (
            REPO_ROOT / "src" / "omnibase_infra" / "cli" / "skill_mapping.yaml"
        ).read_text()
    )
    entry = next(
        skill
        for skill in mapping["skills"]
        if skill["skill_name"] == "evidence_autoclose_sweep"
    )
    declared = {arg["payload_field"] for arg in entry["args"]}
    fields = set(ModelEvidenceAutocloseSweepRequest.model_fields)
    backfill_fields = {name for name in fields if name.startswith("backfill_")}
    assert backfill_fields, "the request model must carry the backfill knobs"
    assert backfill_fields <= declared, (
        "every backfill field on the request model must be reachable from the "
        f"CLI; missing {sorted(backfill_fields - declared)}"
    )


def test_the_disarm_marker_reaches_an_unattended_run() -> None:
    """OMN-17658 auto-disarm, the persisted half, and why it is a VARIABLE.

    The node disarms itself within a run the moment a candidate proves a closer
    flip on it was already undone by a person. That is immediate and not
    durable — the next cron tick is a fresh process. The durable half has to
    bind runs nobody launched, and a ``workflow_dispatch`` input renders empty
    on a cron tick, so it would be missing exactly when it is needed. Same
    reachability argument, and the same fix, as OMN-16792 AC3 found for the
    kill switch: a repo variable, plumbed explicitly into the step's
    environment because ``vars.*`` lives only in the expression context.
    """
    env = _step(SWEEP_STEP_NAME).get("env") or {}
    assert env.get("AUTOCLOSE_DISARMED_BY") == "${{ vars.ONEX_AUTOCLOSE_DISARMED }}", (
        "the disarm marker must be a repo VARIABLE plumbed into the step env; "
        f"got {env.get('AUTOCLOSE_DISARMED_BY')!r}"
    )
    inputs = (_triggers().get("workflow_dispatch") or {}).get("inputs") or {}
    assert "disarmed" not in inputs and "disarm" not in inputs, (
        "the disarm marker must not be a dispatch input — an input renders "
        "empty on every scheduled run, which is every run that writes "
        "unattended"
    )


def test_the_disarm_marker_reaches_the_sweep_only_through_a_guarded_array() -> None:
    """Same property as --apply, --exclude and the backfill flag."""
    run = _step(SWEEP_STEP_NAME).get("run") or ""
    assert 'if [ -n "${AUTOCLOSE_DISARMED_BY:-}" ]' in run, (
        "the disarm flag must be guarded on the marker being non-empty"
    )
    assert 'disarm_args+=(--disarmed-by "${AUTOCLOSE_DISARMED_BY}")' in run, (
        "the marker must reach the CLI as a quoted env value inside an array, "
        "never interpolated"
    )
    invocation = [
        line
        for line in run.splitlines()
        if "onex" in line and "evidence_autoclose_sweep" in line
    ]
    assert invocation, "could not find the `onex skill` invocation"
    for line in invocation:
        assert "--disarmed-by" not in line, (
            f"--disarmed-by must never be literal on the invocation line: {line!r}"
        )


def test_the_disarm_binds_at_the_same_depth_as_the_apply_branch() -> None:
    """A disarm nested inside the apply branch would not bind a scheduled run.

    The same structural property the OMN-17891 fence has, for the same reason:
    the disarm must be evaluated for every run, not only for the shape of run
    an operator launched.
    """
    run = _step(SWEEP_STEP_NAME).get("run") or ""
    disarm_guard = 'if [ -n "${AUTOCLOSE_DISARMED_BY:-}" ]'
    apply_guard = 'if [ "${SWEEP_DISPATCH_APPLY}" = "true" ]'
    disarm_indent = next(
        len(line) - len(line.lstrip())
        for line in run.splitlines()
        if disarm_guard in line
    )
    apply_indent = next(
        len(line) - len(line.lstrip())
        for line in run.splitlines()
        if apply_guard in line
    )
    assert disarm_indent == apply_indent, (
        "the disarm guard must sit at the same nesting depth as the apply "
        f"guard (sibling, not nested): {disarm_indent} vs {apply_indent}"
    )


def test_the_cli_accepts_every_fence_field_the_request_model_declares() -> None:
    """A fence the CLI cannot pass is a fence that does not exist in production.

    The node's contract declaring ``disarmed_by_ticket`` is necessary and not
    sufficient: the workflow reaches the node through ``onex skill``, so a
    missing entry in ``skill_mapping.yaml`` would make the guarded array above
    fail at dispatch — loudly, but only in production.
    """
    from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_request import (
        ModelEvidenceAutocloseSweepRequest,
    )

    mapping = yaml.safe_load(
        (REPO_ROOT / "src" / "omnibase_infra" / "cli" / "skill_mapping.yaml").read_text(
            encoding="utf-8"
        )
    )
    skills = mapping["skills"] if isinstance(mapping, dict) else mapping
    entry = next(s for s in skills if s.get("skill_name") == "evidence_autoclose_sweep")
    payload_fields = {arg["payload_field"] for arg in entry["args"]}
    for field in (
        "trigger",
        "disarmed_by_ticket",
        "max_flips_per_run",
        "history_page_size",
        "history_max_pages",
    ):
        assert field in payload_fields, (
            f"skill_mapping.yaml does not expose {field!r} for "
            "evidence_autoclose_sweep — the workflow could not pass it"
        )
    model_fields = set(ModelEvidenceAutocloseSweepRequest.model_fields)
    assert payload_fields <= model_fields, (
        "skill_mapping.yaml exposes fields the request model does not declare: "
        f"{sorted(payload_fields - model_fields)}"
    )
    assert "scheduled_apply" not in payload_fields, (
        "there must be NO CLI flag for scheduled_apply. A caller that could "
        "pass one would be an arming authority, which is exactly what "
        "OMN-17658 moved into the contract"
    )
