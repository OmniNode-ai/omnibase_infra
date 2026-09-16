# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pins the per-run routing WIRING and its enforcement (OMN-18031).

The decision logic is covered by ``test_runner_route_decision.py``. This module
covers the half that logic cannot defend on its own: that the route job is
actually present, actually hosted, actually registered as a gate, and that the
audit pass that guards the consumer shape is actually wired in.

THE FAILURE MODE THIS EXISTS FOR IS SILENT. Routing is deliberately INERT while
the trusted seam reads ``["ubuntu-latest"]``, so deleting the route job changes
no job placement at all -- the only observable difference between "routing works
and chose hosted" and "routing is gone" is a decision artifact nobody is
required to read. A test is therefore the only thing standing between a
refactor and the quiet disappearance of the whole mechanism.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
POLICY = REPO_ROOT / "config" / "runner_routing_policy.yaml"

ROUTE_GATE_NAME = "Runner Route (OMN-18031) / route"


def _workflow(name: str) -> dict[str, Any]:
    loaded = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    """Return the ``on:`` block.

    YAML 1.1 parses the bare key ``on`` as the BOOLEAN ``True``, so
    ``workflow["on"]`` raises KeyError on every GitHub workflow file. Resolved
    once here rather than rediscovered at each call site.
    """
    block = workflow.get(True, workflow.get("on"))
    assert isinstance(block, dict)
    return block


def _load(module_name: str, relative: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# --- the route job exists, is hosted, and is unconditional ------------------


def test_ci_yml_carries_the_route_job_calling_the_reusable_workflow() -> None:
    route = _workflow("ci.yml")["jobs"]["route"]
    assert route["uses"] == "./.github/workflows/runner-route-reusable.yml"
    assert route["name"] == "Runner Route (OMN-18031)"


def test_the_route_job_is_unconditional_so_a_skip_is_always_anomalous() -> None:
    """A gate registered in STRICT_GATE_JOBS must never legitimately skip: a
    conditional job that skips publishes a `skipped` check run, which branch
    protection counts as passing.
    """
    route = _workflow("ci.yml")["jobs"]["route"]
    assert "if" not in route
    assert "needs" not in route


def test_the_route_decision_job_is_hosted_and_says_why() -> None:
    """A router must not queue behind the fleet it routes onto."""
    reusable = _workflow("runner-route-reusable.yml")
    assert reusable["jobs"]["route"]["runs-on"] == "ubuntu-latest"
    text = (WORKFLOWS / "runner-route-reusable.yml").read_text(encoding="utf-8")
    assert "MUST NOT share fate" in text


def test_the_route_job_is_time_bounded_so_it_cannot_stall_every_consumer() -> None:
    """It sits on the critical path ahead of every heavy job in the run.

    The budget covers a checkout and a cached environment sync as well as the
    decision itself, which takes milliseconds. Widening it is how a routing
    layer starts costing every pipeline in the repository a few minutes that
    nobody attributes to it, so a sync that no longer fits fails here instead.
    """
    assert (
        _workflow("runner-route-reusable.yml")["jobs"]["route"]["timeout-minutes"] <= 5
    )


def test_both_route_call_sites_set_up_the_environment_before_deciding() -> None:
    """THE SILENT DEGRADATION, pinned.

    The route job imports the routing node, and importing anything in this
    package executes its __init__, which pulls the database and broker
    surfaces. A route job without the project environment therefore raises
    ModuleNotFoundError, falls through to its own fail-closed floor, and
    reports a GREEN job whose decision reads
    `probe_error:module_unavailable` -- measured on run 35039101488, where the
    mechanism was completely inert and no check was red.

    Nothing else can see that: the floor exists precisely so a broken router
    still hands the run a usable runs-on. The only durable guard is that the
    setup step is present and precedes the decision.
    """
    for workflow_name, job in ROUTE_CALL_SITES:
        steps = _workflow(workflow_name)["jobs"][job]["steps"]
        setup_index = next(
            (
                i
                for i, step in enumerate(steps)
                if "setup-python-uv" in str(step.get("uses", ""))
            ),
            None,
        )
        assert setup_index is not None, (
            f"{workflow_name}:{job} invokes the routing node without setting up "
            f"the environment; it will fall through to the fail-closed floor and "
            f"report a green, inert route job"
        )
        decide_index = next(
            i for i, step in enumerate(steps) if step.get("id") == "decide"
        )
        assert setup_index < decide_index, workflow_name
        body = steps[decide_index]["run"]
        assert "uv run --frozen" in body, (
            f"{workflow_name}:{job} runs the module outside the synced environment"
        )


def test_no_workflow_call_declaration_carries_an_expression() -> None:
    """A STARTUP FAILURE, and the most expensive shape of one.

    Actions evaluates `${{ }}` inside a workflow_call inputs/outputs block,
    where the `needs` context does not exist. An illustrative snippet in an
    output DESCRIPTION therefore fails every CALLING workflow before a single
    job starts -- measured on run 35037012235, where ci.yml and the routing
    probe each concluded failure with zero jobs and the run title fell back to
    the file path, which is the only visible symptom. No check reports on a run
    that never started, so nothing else in this suite can see it.
    """
    for name in ("runner-route-reusable.yml", "runner-route-probe.yml"):
        block = _triggers(_workflow(name)).get("workflow_call")
        if not isinstance(block, dict):
            continue
        for section in ("inputs", "outputs"):
            for field, spec in (block.get(section) or {}).items():
                description = str((spec or {}).get("description", ""))
                assert "${{" not in description, (
                    f"{name}: workflow_call.{section}.{field} description carries "
                    f"an expression; that is a startup failure in every caller"
                )


def test_the_reusable_workflow_exposes_the_outputs_consumers_read() -> None:
    """A consumer reads the label set and can audit the verdict.

    `runs_on` is the name a call site reads it by -- `runs-on:
    fromJSON(needs.route.outputs.runs_on)` says what it is at the point of use
    -- and `labels` is the name the pilot shipped and the saturation monitor
    already consumes. Both are emitted rather than one renamed: renaming the
    one a live consumer reads would break it silently, since an unknown output
    resolves to the empty string and an empty `runs-on` fails to schedule
    rather than failing loudly.
    """
    outputs = _workflow("runner-route-reusable.yml")[True]["workflow_call"]["outputs"]
    assert set(outputs) == {"labels", "runs_on", "decision", "reason"}
    assert "jobs.route.outputs.runs_on" in outputs["runs_on"]["value"]
    assert "jobs.route.outputs.labels" in outputs["labels"]["value"]


# --- enforcement: registration is half the mechanism -----------------------


def test_the_route_job_is_registered_as_a_strict_gate() -> None:
    """Without this entry an absent or skipped route job yields CI Summary
    SUCCESS, silently retiring per-run routing on a fully green run.
    """
    gate = _load("ci_summary_gate", "scripts/ci/ci_summary_gate.py")
    assert ROUTE_GATE_NAME in gate.STRICT_GATE_JOBS


def test_the_registered_gate_name_matches_the_workflow_it_names() -> None:
    """The '<caller display name> / <inner job name>' shape is fragile by
    construction: renaming either half silently unregisters the gate. This
    derives both halves from the workflows rather than restating the string.
    """
    caller = _workflow("ci.yml")["jobs"]["route"]["name"]
    inner = _workflow("runner-route-reusable.yml")["jobs"]["route"]["name"]
    assert f"{caller} / {inner}" == ROUTE_GATE_NAME


def test_the_route_workflows_are_pinned_hosted_in_the_policy_allowlist() -> None:
    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    paths = {entry["path"] for entry in policy["hosted_runner_allowlist"]}
    assert ".github/workflows/runner-route-reusable.yml" in paths
    assert ".github/workflows/runner-route-probe.yml" in paths


def test_the_canary_allowlist_reason_was_not_widened_at_all() -> None:
    """The canary entry must still cover exactly what it always covered.

    The first build added the two monitor jobs INTO runner-fleet-canary.yml and
    extended this reason text to match. That broke a codified invariant --
    tests/ci/test_dev_lane_liveness.py asserts the canary stays single-job -- so
    the jobs moved to dev-lane-liveness.yml and this entry went back to its
    original OMN-13915 justification. An entry that still mentioned jobs it no
    longer covers is an allowlist describing a file that does not exist.
    """
    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in policy["hosted_runner_allowlist"]
        if item["path"] == ".github/workflows/runner-fleet-canary.yml"
    )
    assert "OMN-18031" not in entry["reason"]
    assert "saturation-record" not in entry["reason"]
    assert "lab-load-probe" not in entry["reason"]


def test_the_new_hosted_monitor_job_carries_its_own_allowlist_reason() -> None:
    """dev-lane-liveness.yml gained one bare-hosted job, so it needs an entry
    naming that job and why it is hosted -- otherwise the audit's bare
    ubuntu-latest rule would fire, or an unexplained entry would suppress it.
    """
    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in policy["hosted_runner_allowlist"]
        if item["path"] == ".github/workflows/dev-lane-liveness.yml"
    )
    assert "OMN-18031" in entry["reason"]
    assert "saturation-record" in entry["reason"]


def test_the_audit_route_wiring_pass_is_wired_into_local_workflows() -> None:
    """Rule 5: a check that is not wired as a gate is advisory and gets ignored."""
    source = (REPO_ROOT / "scripts" / "audit-runner-routing.py").read_text(
        encoding="utf-8"
    )
    assert "def audit_route_wiring(" in source
    assert "findings.extend(audit_route_wiring(policy, args.repo_root))" in source


def test_the_audit_route_wiring_pass_actually_fires_on_a_broken_consumer(
    tmp_path: Path,
) -> None:
    """POSITIVE CONTROL. The pass returns zero findings against the real tree,
    and a checker that always returns zero is indistinguishable from a correct
    one. This proves it is capable of a finding at all.
    """
    audit = _load("audit_runner_routing", "scripts/audit-runner-routing.py")
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "broken.yml").write_text(
        "name: broken\non: [push]\njobs:\n"
        "  consume:\n"
        "    runs-on: ${{ fromJSON(needs.route.outputs.labels) }}\n"
        "    steps: [{run: 'true'}]\n",
        encoding="utf-8",
    )
    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    findings = audit.audit_route_wiring(policy, tmp_path)
    messages = " ".join(f.message for f in findings)
    assert findings, "the route-wiring pass produced no finding on a broken consumer"
    assert "does not declare needs" in messages


def test_the_audit_route_wiring_pass_is_clean_against_the_real_tree() -> None:
    audit = _load("audit_runner_routing", "scripts/audit-runner-routing.py")
    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    findings = audit.audit_route_wiring(policy, REPO_ROOT)
    assert findings == [], [f"{f.scope}: {f.message}" for f in findings]


# --- the P1 wiring proof itself --------------------------------------------


@pytest.mark.parametrize(
    ("consumer", "producer"),
    [("consume-inline", "route-inline"), ("consume-reusable", "route-reusable")],
)
def test_the_probe_proves_both_wiring_shapes(consumer: str, producer: str) -> None:
    """P1 must prove the NORMAL-job shape and the REUSABLE-WORKFLOW shape
    together. They are not the same claim, and the four-repo fan-out depends on
    the second one specifically.
    """
    jobs = _workflow("runner-route-probe.yml")["jobs"]
    job = jobs[consumer]
    assert job["runs-on"] == f"${{{{ fromJSON(needs.{producer}.outputs.labels) }}}}"
    assert job["needs"] == producer


def test_the_probe_gate_treats_a_skipped_consumer_as_a_failure() -> None:
    """The failure P1 hunts is a job that never SCHEDULED because its runs-on
    resolved to nothing. That presents as `skipped`, not as a red test, so the
    gate must assert on the result explicitly.
    """
    text = (WORKFLOWS / "runner-route-probe.yml").read_text(encoding="utf-8")
    assert 'inline}" = "success"' in text or '[ "${inline}" = "success" ]' in text
    assert '[ "${reusable}" = "success" ]' in text


# --- the saturation monitor rides the EXISTING scheduler -------------------


def test_the_saturation_monitor_rides_an_existing_scheduler_not_a_third_one() -> None:
    """It extends dev-lane-liveness.yml, which already runs on a schedule.

    NOT runner-fleet-canary.yml, which is where it first went:
    tests/ci/test_dev_lane_liveness.py::test_fleet_canary_is_not_borrowed_for_the_lane_probe
    asserts the canary stays single-job and GitHub-hosted, and its own failure
    message names this workflow as the home for lane/host-scoped probes. That
    invariant exists because a canary sharing fate with the fleet it watches
    proves nothing.
    """
    lane = _workflow("dev-lane-liveness.yml")
    assert _triggers(lane)["schedule"], "dev-lane-liveness must already be scheduled"
    assert {"dev-lane-liveness", "lab-load-probe", "saturation-record"} <= set(
        lane["jobs"]
    )


def test_the_canary_is_left_exactly_as_it_was() -> None:
    """Single-job, GitHub-hosted, unchanged cadence -- the OMN-13915 boundary."""
    canary = _workflow("runner-fleet-canary.yml")
    assert list(canary["jobs"]) == ["fleet-status"]
    assert canary["jobs"]["fleet-status"]["runs-on"] == "ubuntu-latest"
    assert _triggers(canary)["schedule"] == [{"cron": "*/15 * * * *"}]


def test_no_third_scheduled_routing_workflow_was_added() -> None:
    """The consent row puts a third scheduler out of scope. This asserts the
    saturation monitor did not quietly become one.
    """
    scheduled = []
    for path in WORKFLOWS.glob("*.y*ml"):
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            continue
        triggers = loaded.get(True, loaded.get("on"))
        if isinstance(triggers, dict) and "schedule" in triggers:
            scheduled.append(path.name)
    assert "runner-route-probe.yml" not in scheduled
    assert "runner-route-reusable.yml" not in scheduled


def test_the_lab_probe_is_pinned_self_hosted_by_a_literal_not_the_seam() -> None:
    """It must run ON the lab: a hosted runner has no route to the lab's
    tailnet/RFC1918 addresses.

    The runs-on is a LITERAL and not OMNI_RUNNER_SELECTOR_V1 on purpose: routed
    through the seam, a flip could relocate this probe onto GitHub-hosted
    compute, where it would measure the wrong machine and publish the reading as
    the lab's. The pre-existing dev-lane-liveness job pins its own runner for
    the same reason.

    OMN-18253: this used to also assert ``continue-on-error is True`` here, on
    the reasoning that the job's failure is a data point rather than an outage.
    That reasoning is correct and is NOT dropped -- it moved into the probe,
    which exits 0 for a measurement whether or not the lab is saturated and
    non-zero only when it cannot measure at all. The blanket was what also hid
    the ModuleNotFoundError that killed 10 of this job's first 12 runs, so the
    property is now asserted where it is true rather than where it was
    convenient. tests/ci/test_lab_load_probe_omn18253.py owns that half,
    including the saturated-lab control; this asserts the blanket is gone.
    """
    job = _workflow("dev-lane-liveness.yml")["jobs"]["lab-load-probe"]
    assert "continue-on-error" not in job
    assert job["timeout-minutes"] <= 3
    assert job["runs-on"] == ["self-hosted", "omnibase-ci"]
    assert "OMNI_TRUSTED_CI_RUNS_ON_JSON" not in str(job["runs-on"])


def test_the_saturation_record_runs_even_when_its_input_failed() -> None:
    """An absent lab probe is the state the monitor most needs to record -- it
    is alert condition 4, not a reason to skip the sample.
    """
    job = _workflow("dev-lane-liveness.yml")["jobs"]["saturation-record"]
    assert job["if"] == "always()"
    assert job["runs-on"] == "ubuntu-latest"
    assert set(job["needs"]) == {"lab-load-probe"}


def _step(job: dict[str, Any], name: str) -> dict[str, Any]:
    for step in job["steps"]:
        if step.get("name") == name:
            return step
    raise AssertionError(f"no step named {name!r} in job {job.get('name')!r}")


def test_the_lab_probe_step_fails_loudly_and_does_not_read_the_container() -> None:
    """OMN-18031 follow-up (2026-09-12): the probe step used to run
    ``set -uo pipefail`` (no ``-e``) around a bare ``python3 -`` call to
    ``probe_local_lab_load()``, which measures THIS CONTAINER's own
    ``/proc/loadavg`` and ``os.cpu_count()`` -- not the ``.201`` host
    ``max_lab_load_ratio`` is calibrated against. It must now fail the step on
    a crash (``-e``) and call the org-busy/idle-based function instead.

    OMN-18253 moved the call out of the inline heredoc into
    ``scripts/ci/probe_lab_load.py``, because a heredoc that could not fail
    honestly -- the probe never raises, so an unreachable API exited 0 -- and
    could not be driven by a test was half the reason the job stayed dead. The
    requirement is unchanged and is asserted through the module: the workflow
    invokes it, and the module sources the ORG-BUSY function rather than the
    container-scoped one.
    """
    job = _workflow("dev-lane-liveness.yml")["jobs"]["lab-load-probe"]
    step = _step(
        job, "Probe the fleet's org busy/idle counts as the lab saturation signal"
    )
    script = step["run"]
    assert "set -euo pipefail" in script
    assert "scripts/ci/probe_lab_load.py" in script
    assert "probe_local_lab_load" not in script

    module = (
        Path(__file__).resolve().parents[2] / "scripts" / "ci" / "probe_lab_load.py"
    ).read_text(encoding="utf-8")
    assert "probe_lab_saturation_from_fleet" in module
    assert "probe_local_lab_load" not in module


def test_route_reason_is_no_longer_a_hardcoded_literal() -> None:
    """OMN-18031 follow-up (2026-09-12): ``ROUTE_REASON: unknown`` had no
    input, so condition 3 (``route_fallback_sustained`` -- the alert the
    operator actually asked for) could never fire. It must now be sourced
    from a step output, with ``unknown`` only as its documented fallback.
    """
    job = _workflow("dev-lane-liveness.yml")["jobs"]["saturation-record"]
    step = _step(job, "Build the record and evaluate sustained saturation")
    route_reason_env = str(step["env"]["ROUTE_REASON"])
    assert route_reason_env != "unknown"
    assert "steps.route_reason.outputs.reason" in route_reason_env

    reason_step = _step(job, "Fetch the most recent route decision's reason")
    assert reason_step["id"] == "route_reason"
    assert reason_step["continue-on-error"] is True


# --- the seam is read, never written --------------------------------------


def test_nothing_added_here_writes_an_actions_variable() -> None:
    """CLAUDE.md rule 14: the routing variables are single-owner. This lane
    reads them as a ceiling and must contain no mutation path at all -- a
    read-modify-write from a second lane clobbers with no error and no audit.
    """
    surfaces = [
        REPO_ROOT / "scripts" / "ci" / "runner_route_decision.py",
        REPO_ROOT / "scripts" / "ci" / "runner_saturation_record.py",
        WORKFLOWS / "runner-route-reusable.yml",
        WORKFLOWS / "runner-route-probe.yml",
    ]
    forbidden = (
        "gh variable set",
        "gh api -X PATCH",
        "actions/variables",
        "gh variable delete",
    )
    for path in surfaces:
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, (
                f"{path.name} contains a variable-mutation path: {needle}"
            )


# --- the seam reaches the module as JSON, not as shell-mangled argv --------


def _decide_step(workflow_name: str, job: str) -> dict[str, Any]:
    wf = _workflow(workflow_name)
    for step in wf["jobs"][job]["steps"]:
        if isinstance(step, dict) and step.get("id") == "decide":
            return step
    raise AssertionError(f"{workflow_name}:{job} has no step with id 'decide'")


def test_the_decide_step_body_contains_no_actions_interpolation() -> None:
    """MEASURED LIVE, on the merged pilot's own CI run.

    Run 34131479134, artifact `runner-route-decision-34131479134-route`, the
    first real decision this mechanism ever published:

        {"decision": "hosted", "reason": "probe_error:seam_unparseable", ...}

    The step interpolated the runner variable straight into the shell command:

        --seam-json "${{ vars.OMNI_TRUSTED_CI_RUNS_ON_JSON }}"

    Actions substitutes the RAW value before the shell parses the line, so
    `["self-hosted","omnibase-ci"]` renders as
    `--seam-json "["self-hosted","omnibase-ci"]"`. The shell strips the inner
    quotes and the module receives `[self-hosted,omnibase-ci]`, which is not
    JSON. Reproduced exactly in a local bash one-liner.

    The failure is fail-CLOSED, and that is precisely why it survived review:
    every decision was `hosted`, which is what an inert mechanism looks like.
    It was invisible under the old hosted seam too -- `["ubuntu-latest"]`
    renders as `[ubuntu-latest]`, equally unparseable, equally hosted. The
    reason string was the only thing that ever differed, and nothing read it.
    A green route job is not evidence the module saw its inputs.

    Pinning "no interpolation anywhere in the run body" rather than "quote the
    seam properly" is deliberate: it is one rule a reader can check by looking,
    it cannot be satisfied by a cleverer quoting, and it closes the Actions
    template-injection class in the same stroke.
    """
    for workflow_name, job in (
        ("runner-route-reusable.yml", "route"),
        ("runner-route-probe.yml", "route-inline"),
    ):
        body = _decide_step(workflow_name, job)["run"]
        assert "${{" not in body, (
            f"{workflow_name}:{job} interpolates an Actions expression into the "
            f"shell body; pass it through env: instead -- a JSON array renders "
            f"as unquoted argv and reaches the module as non-JSON"
        )


def test_the_runner_variables_are_supplied_through_env() -> None:
    """The counterpart: removing the interpolation must not remove the input.

    A step with an interpolation-free body that no longer receives the seam at
    all would satisfy the test above and route hosted forever -- the same
    outcome, one layer down.
    """
    for workflow_name, job in (
        ("runner-route-reusable.yml", "route"),
        ("runner-route-probe.yml", "route-inline"),
    ):
        env = _decide_step(workflow_name, job).get("env") or {}
        joined = " ".join(str(v) for v in env.values())
        assert "OMNI_TRUSTED_CI_RUNS_ON_JSON" in joined, workflow_name
        assert "OMNI_PUBLIC_PR_RUNS_ON_JSON" in joined, workflow_name
        body = _decide_step(workflow_name, job)["run"]
        assert "--seam-json" in body and "--public-json" in body, workflow_name


# --- OMN-18412: the private-repo rule reaches the workflow, not just the module


ROUTE_CALL_SITES = (
    ("runner-route-reusable.yml", "route"),
    ("runner-route-probe.yml", "route-inline"),
)


def test_both_route_call_sites_give_the_module_a_token_to_read_visibility() -> None:
    """Without a token the probe returns `unknown` on every run, and the
    private-repo rule degrades to its weaker half in silence -- a green route
    job, a plausible record, and no refusal that should have happened.

    The JOB token is what is wired, deliberately: repository metadata is
    readable under the `metadata: read` scope every Actions token already
    carries, so the rule costs no new credential and no new scope.
    """
    for workflow_name, job in ROUTE_CALL_SITES:
        env = _decide_step(workflow_name, job).get("env") or {}
        assert "GITHUB_TOKEN" in env, (
            f"{workflow_name}:{job} gives the module no token, so "
            f"probe_repo_visibility returns 'unknown' on every run"
        )
        assert "github.token" in str(env["GITHUB_TOKEN"]), workflow_name


def test_both_route_call_sites_fail_the_run_on_a_refusal() -> None:
    """A refusal must STOP the run. It is not a crash and not a capacity
    fallback: it means the only placement this repository's policy allows is
    one the 2026-09-14 operator ruling forbids.

    Three things are pinned together because each fails on its own. `rc` must
    be initialised, or `set -u` makes the check itself the error. The exit
    status must be CAPTURED rather than swallowed. And the run must exit
    non-zero, or the refusal is a log line that places the jobs anyway.
    """
    for workflow_name, job in ROUTE_CALL_SITES:
        body = _decide_step(workflow_name, job)["run"]
        assert "rc=0" in body, f"{workflow_name}:{job} leaves rc unset under set -u"
        assert "|| rc=$?" in body, (
            f"{workflow_name}:{job} discards the module's exit status, so a "
            f"refusal is indistinguishable from a successful decision"
        )
        assert '[ "${rc}" = "3" ]' in body, workflow_name
        assert "exit 1" in body, (
            f"{workflow_name}:{job} reports a refusal without failing the run"
        )


def test_the_refusal_check_runs_after_the_fail_closed_floor() -> None:
    """ORDER IS THE MECHANISM. The floor writes hosted labels whenever no
    `labels=` line exists, so that a crashed router still hands the run a
    usable runs-on. If the refusal check ran first, a refusal would exit before
    the floor and the ordering would not matter; if the floor ran and then
    OVERWROTE a refusal, a forbidden hosted placement would be emitted by the
    very step that exists to make failures safe. The module writes its own
    `labels=` line before returning 3, which makes the floor a no-op on this
    path -- and this test pins the arrangement that keeps it one.
    """
    for workflow_name, job in ROUTE_CALL_SITES:
        body = _decide_step(workflow_name, job)["run"]
        floor = body.index("probe_error:module_unavailable")
        refusal = body.index('[ "${rc}" = "3" ]')
        assert floor < refusal, (
            f"{workflow_name}:{job} checks the refusal before the fail-closed "
            f"floor, so the floor can overwrite a refusal with hosted labels"
        )


def test_the_refusal_exit_code_matches_the_module() -> None:
    """The workflow tests a literal 3 and the module returns a named constant.
    A rename or renumber on either side silently disarms the check, so the two
    are compared rather than assumed equal.
    """
    route = _load("runner_route_decision", "scripts/ci/runner_route_decision.py")
    assert route.REFUSED_EXIT_CODE == 3
    for workflow_name, job in ROUTE_CALL_SITES:
        body = _decide_step(workflow_name, job)["run"]
        assert f'[ "${{rc}}" = "{route.REFUSED_EXIT_CODE}" ]' in body, workflow_name


def test_visibility_is_read_by_the_module_and_not_interpolated() -> None:
    """The counterpart to the no-interpolation rule: the fact arrives because
    the module reads it, so no `--repo-visibility` argument is spelled at
    either call site. An interpolated `github.event.repository.private` would
    also be absent from some event payloads, which is the second reason the
    probe owns this rather than the template.
    """
    for workflow_name, job in ROUTE_CALL_SITES:
        body = _decide_step(workflow_name, job)["run"]
        assert "--repo-visibility" not in body, workflow_name
        assert "${{" not in body, workflow_name


# --- G6: the first heavy-job consumer of the route output -------------------
#
# `lint` is the ONE job whose placement now resolves from the route decision.
# The tests below pin three separate things, because they fail independently:
# that the consumer is wired at all, that its blast radius is still one job,
# and that moving it did not relocate fork isolation to a call site.

LINT_RUNS_ON = "${{ fromJSON(needs.route.outputs.labels) }}"
SELECTOR_V2_MARKER = "OMNI_RUNNER_SELECTOR_V2"


def test_lint_resolves_its_placement_from_the_route_output() -> None:
    lint = _workflow("ci.yml")["jobs"]["lint"]
    assert lint["runs-on"].strip() == LINT_RUNS_ON
    assert lint["name"] == "Lint"


def test_lint_declares_the_needs_edge_its_expression_requires() -> None:
    """Without `needs: route` the expression resolves to nothing and the job
    fails to SCHEDULE -- which does not surface as a test failure anywhere.
    """
    lint = _workflow("ci.yml")["jobs"]["lint"]
    assert "route" in lint["needs"]
    assert "occ-preflight" in lint["needs"], "the pre-existing OCC edge was dropped"


def test_lint_carries_the_v2_selector_marker() -> None:
    """The marker is how runner placement is FOUND. Every survey to date was a
    grep for the V1 string; an unmarked consumer is invisible to the next one.
    """
    text = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    lines = text.splitlines()
    index = next(i for i, line in enumerate(lines) if LINT_RUNS_ON in line)
    assert SELECTOR_V2_MARKER in "\n".join(lines[max(0, index - 6) : index])


def test_lint_is_the_only_route_consumer_in_ci_yml() -> None:
    """Blast radius. One cheap always-running job proves the mechanism; the
    other heavy jobs move only on separate evidence, so a routing defect cannot
    take the whole fan-out with it.
    """
    jobs = _workflow("ci.yml")["jobs"]
    consumers = sorted(
        name
        for name, job in jobs.items()
        if isinstance(job, dict)
        and isinstance(job.get("runs-on"), str)
        and "outputs.labels" in job["runs-on"]
    )
    assert consumers == ["lint"]


def test_the_other_heavy_jobs_still_read_the_seam_directly() -> None:
    """The corollary of the test above, asserted positively: V1 was not removed
    from the jobs that were left alone. A migration that quietly blanked their
    selectors would also satisfy the "only lint consumes route" assertion.
    """
    text = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    assert text.count("OMNI_RUNNER_SELECTOR_V1") >= 3


def test_lints_runs_on_names_no_runner_variable_at_all() -> None:
    """Fork isolation must NOT be restated at the call site. It lives once,
    inside the decision module (step S1). A consumer that re-implements the fork
    branch re-opens the hole the single implementation exists to close.
    """
    runs_on = _workflow("ci.yml")["jobs"]["lint"]["runs-on"]
    for variable in (
        "OMNI_PUBLIC_PR_RUNS_ON_JSON",
        "OMNI_TRUSTED_CI_RUNS_ON_JSON",
        "OMNI_REQUIRED_CI_RUNS_ON_JSON",
    ):
        assert variable not in runs_on


def _decide_via_node(**kwargs: Any) -> Any:
    """Drive the shipped decision -- the routing node's handler.

    The wiring tests assert on PLACEMENT, so they go through the same typed
    request the route job builds rather than re-deriving one; a wiring test
    that exercised a copy of the decision would pass while the shipped one
    changed underneath it.
    """
    route_module = _load("runner_route_decision", "scripts/ci/runner_route_decision.py")
    from omnibase_infra.nodes.node_ci_runner_route_compute.handlers.handler_ci_runner_route import (
        HandlerCIRunnerRoute,
    )

    policy = kwargs.pop("policy")
    request = route_module.build_request(
        event_name=kwargs["event_name"],
        head_repo=kwargs.get("head_repo", ""),
        repository=kwargs["repository"],
        workflow_path=kwargs["workflow_path"],
        seam_json=kwargs["seam_json"],
        public_json=kwargs["public_json"],
        visibility=kwargs.get("visibility", "public"),
        fleet=kwargs["fleet"],
        lab=kwargs["lab"],
        hosted_workflows=tuple(kwargs.get("allowlist") or ()),
        force="auto",
        policy=policy,
        fleet_expected_count=88,
    )
    decision = HandlerCIRunnerRoute().handle(request)
    return SimpleNamespace(
        labels=list(decision.runs_on),
        decision=decision.decision.value,
        reason=decision.reason_wire,
    )


def test_fork_pr_isolation_still_holds_for_the_lint_workflow_path() -> None:
    """The behavioural half of the test above. For a fork PR the route output
    IS the public variable's labels, so lint's placement on a fork is decided by
    the same knob the V1 expression read -- before any capacity signal.
    """
    route = _load("runner_route_decision", "scripts/ci/runner_route_decision.py")
    # The SHIPPED thresholds, from the contract that declares them. Read from
    # the contract rather than the config file precisely because the config
    # file no longer carries them: a test that kept reading the old home would
    # pass on a policy nothing applies.
    route_module = _load("runner_route_decision", "scripts/ci/runner_route_decision.py")
    policy = route_module.load_contract_policy(
        REPO_ROOT
        / "src/omnibase_infra/nodes/node_ci_runner_route_compute/contract.yaml"
    )
    result = _decide_via_node(
        event_name="pull_request",
        head_repo="a-fork/omnibase_infra",
        repository="OmniNode-ai/omnibase_infra",
        # omnibase_infra is public, so the OMN-18412 rule is a no-op here and
        # this stays a statement about fork isolation alone.
        visibility="public",
        workflow_path=".github/workflows/ci.yml",
        # A seam that DOES permit the lab, so the assertion is about fork
        # isolation and not about today's inert hosted ceiling.
        seam_json='["self-hosted","omnibase-ci"]',
        public_json='["ubuntu-latest"]',
        fleet={"ok": True, "online": 88, "busy": 0, "total": 88},
        lab={
            "ok": True,
            "age_seconds": 5,
            "hosts": [{"label": "h201", "ratio": 0.1, "free_mem_mib": 40000}],
        },
        policy=policy,
        allowlist=[],
    )
    assert result.reason == "fork_isolation"
    assert result.labels == ["ubuntu-latest"]
    assert "self-hosted" not in result.labels


def test_a_misconfigured_public_variable_cannot_widen_a_fork_onto_the_fleet() -> None:
    """Positive control on the guard above: even if the public variable itself
    named the fleet, a fork PR still lands hosted.
    """
    route = _load("runner_route_decision", "scripts/ci/runner_route_decision.py")
    # The SHIPPED thresholds, from the contract that declares them. Read from
    # the contract rather than the config file precisely because the config
    # file no longer carries them: a test that kept reading the old home would
    # pass on a policy nothing applies.
    route_module = _load("runner_route_decision", "scripts/ci/runner_route_decision.py")
    policy = route_module.load_contract_policy(
        REPO_ROOT
        / "src/omnibase_infra/nodes/node_ci_runner_route_compute/contract.yaml"
    )
    result = _decide_via_node(
        event_name="pull_request",
        head_repo="a-fork/omnibase_infra",
        repository="OmniNode-ai/omnibase_infra",
        # omnibase_infra is public, so the OMN-18412 rule is a no-op here and
        # this stays a statement about fork isolation alone.
        visibility="public",
        workflow_path=".github/workflows/ci.yml",
        seam_json='["self-hosted","omnibase-ci"]',
        public_json='["self-hosted","omnibase-ci"]',
        fleet={"ok": True, "online": 88, "busy": 0, "total": 88},
        lab={
            "ok": True,
            "age_seconds": 5,
            "hosts": [{"label": "h201", "ratio": 0.1, "free_mem_mib": 40000}],
        },
        policy=policy,
        allowlist=[],
    )
    assert result.reason == "fork_isolation"
    assert "self-hosted" not in result.labels


def test_the_lint_gate_name_is_unchanged_so_ci_summary_still_matches() -> None:
    """CI Summary is the single required context here. A renamed job is an
    absent gate, and an unregistered absent gate yields SUCCESS.
    """
    gate = _load("ci_summary_gate", "scripts/ci/ci_summary_gate.py")
    assert "Lint" in gate.STRICT_GATE_JOBS


# --- the V2 marker audit pass ----------------------------------------------


def test_the_selector_marker_pass_is_wired_into_local_workflows() -> None:
    """Rule 5: an unwired check is advisory and gets ignored."""
    source = (REPO_ROOT / "scripts" / "audit-runner-routing.py").read_text(
        encoding="utf-8"
    )
    assert "def audit_selector_markers(" in source
    assert "findings.extend(audit_selector_markers(args.repo_root))" in source


def _marker_findings(tmp_path: Path, body: str) -> list[str]:
    audit = _load("audit_runner_routing", "scripts/audit-runner-routing.py")
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True, exist_ok=True)
    (workflows / "fixture.yml").write_text(body, encoding="utf-8")
    return [f.message for f in audit.audit_selector_markers(tmp_path)]


def test_the_marker_pass_fires_on_an_unmarked_route_consumer(tmp_path: Path) -> None:
    messages = " ".join(
        _marker_findings(
            tmp_path,
            "name: f\non: [push]\njobs:\n"
            "  route:\n    runs-on: ubuntu-latest\n    outputs:\n"
            "      labels: x\n    steps: [{run: 'true'}]\n"
            "  consume:\n    needs: route\n"
            "    runs-on: ${{ fromJSON(needs.route.outputs.labels) }}\n"
            "    steps: [{run: 'true'}]\n",
        )
    )
    assert "carries no OMNI_RUNNER_SELECTOR_V2 marker" in messages


def test_the_marker_pass_fires_on_a_marker_left_behind_by_a_revert(
    tmp_path: Path,
) -> None:
    """A job reverted to the seam expression keeps a marker that now lies --
    worse than no marker, because a survey counts it as migrated.
    """
    messages = " ".join(
        _marker_findings(
            tmp_path,
            "name: f\non: [push]\njobs:\n"
            "  consume:\n"
            "    # OMNI_RUNNER_SELECTOR_V2 — placement comes from the route job.\n"
            "    runs-on: ubuntu-latest\n"
            "    steps: [{run: 'true'}]\n",
        )
    )
    assert "no runs-on below it resolves from a route output" in messages


def test_the_marker_pass_fires_on_a_half_finished_migration(tmp_path: Path) -> None:
    """V1 comment over a V2 expression: the comment contradicts the code."""
    messages = " ".join(
        _marker_findings(
            tmp_path,
            "name: f\non: [push]\njobs:\n"
            "  route:\n    runs-on: ubuntu-latest\n    outputs:\n"
            "      labels: x\n    steps: [{run: 'true'}]\n"
            "  consume:\n    needs: route\n"
            "    # OMNI_RUNNER_SELECTOR_V1 — trusted CI defaults to self-hosted\n"
            "    # OMNI_RUNNER_SELECTOR_V2 — placement comes from the route job.\n"
            "    runs-on: ${{ fromJSON(needs.route.outputs.labels) }}\n"
            "    steps: [{run: 'true'}]\n",
        )
    )
    assert "still carries the OMNI_RUNNER_SELECTOR_V1 marker" in messages


def test_the_marker_pass_accepts_a_correctly_marked_consumer(tmp_path: Path) -> None:
    """The negative control for the three positives above: a checker that
    always finds something is as useless as one that never does.
    """
    assert (
        _marker_findings(
            tmp_path,
            "name: f\non: [push]\njobs:\n"
            "  route:\n    runs-on: ubuntu-latest\n    outputs:\n"
            "      labels: x\n    steps: [{run: 'true'}]\n"
            "  consume:\n    needs: route\n"
            "    # OMNI_RUNNER_SELECTOR_V2 — placement comes from the route job.\n"
            "    runs-on: ${{ fromJSON(needs.route.outputs.labels) }}\n"
            "    steps: [{run: 'true'}]\n",
        )
        == []
    )


def test_the_marker_pass_is_clean_against_the_real_tree() -> None:
    audit = _load("audit_runner_routing", "scripts/audit-runner-routing.py")
    findings = audit.audit_selector_markers(REPO_ROOT)
    assert findings == [], [f"{f.scope}: {f.message}" for f in findings]
