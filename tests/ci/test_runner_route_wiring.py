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
    """It sits on the critical path ahead of every heavy job in the run."""
    assert (
        _workflow("runner-route-reusable.yml")["jobs"]["route"]["timeout-minutes"] <= 5
    )


def test_the_reusable_workflow_exposes_the_three_outputs_consumers_read() -> None:
    outputs = _triggers(_workflow("runner-route-reusable.yml"))["workflow_call"][
        "outputs"
    ]
    assert set(outputs) == {"labels", "decision", "reason"}


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
    """It must run ON the lab (a hosted runner has no route to the lab's
    tailnet/RFC1918 addresses) and its failure is a data point, not an outage.

    The runs-on is a LITERAL and not OMNI_RUNNER_SELECTOR_V1 on purpose: routed
    through the seam, a flip could relocate this probe onto GitHub-hosted
    compute, where it would measure the wrong machine and publish the reading as
    the lab's. The pre-existing dev-lane-liveness job pins its own runner for
    the same reason.
    """
    job = _workflow("dev-lane-liveness.yml")["jobs"]["lab-load-probe"]
    assert job["continue-on-error"] is True
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
