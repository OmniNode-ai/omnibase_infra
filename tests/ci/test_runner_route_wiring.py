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


def test_the_canary_allowlist_reason_was_widened_with_its_new_jobs() -> None:
    """Adding jobs to an allowlisted file silently widens what the entry covers.
    The reason text must have grown to match, or the allowlist has outgrown its
    own justification.
    """
    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in policy["hosted_runner_allowlist"]
        if item["path"] == ".github/workflows/runner-fleet-canary.yml"
    )
    assert "OMN-18031" in entry["reason"]
    assert "saturation-record" in entry["reason"]
    assert "lab-load-probe" in entry["reason"]


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


def test_the_saturation_monitor_extends_the_existing_canary_not_a_third_scheduler() -> (
    None
):
    canary = _workflow("runner-fleet-canary.yml")
    schedule = _triggers(canary)["schedule"]
    assert schedule == [{"cron": "*/5 * * * *"}]
    assert {"fleet-status", "lab-load-probe", "saturation-record"} <= set(
        canary["jobs"]
    )


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


def test_the_fleet_status_job_stays_hosted_byte_for_byte() -> None:
    """The canary's isolation from the fleet it monitors is unchanged."""
    assert (
        _workflow("runner-fleet-canary.yml")["jobs"]["fleet-status"]["runs-on"]
        == "ubuntu-latest"
    )


def test_the_lab_probe_is_self_hosted_and_cannot_block_the_canary() -> None:
    """It must run ON the lab (a hosted runner has no route to the lab's
    tailnet/RFC1918 addresses) and its failure is a data point, not an outage.
    """
    job = _workflow("runner-fleet-canary.yml")["jobs"]["lab-load-probe"]
    assert job["continue-on-error"] is True
    assert job["timeout-minutes"] <= 3
    assert "OMNI_TRUSTED_CI_RUNS_ON_JSON" in job["runs-on"]


def test_the_saturation_record_runs_even_when_its_inputs_failed() -> None:
    """A red fleet-status and an absent lab probe are the two states the
    monitor most needs to record.
    """
    job = _workflow("runner-fleet-canary.yml")["jobs"]["saturation-record"]
    assert job["if"] == "always()"
    assert job["runs-on"] == "ubuntu-latest"
    assert set(job["needs"]) == {"fleet-status", "lab-load-probe"}


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
