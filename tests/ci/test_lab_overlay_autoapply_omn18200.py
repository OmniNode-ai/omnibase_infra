# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18200 AC7 -- the rebuild trigger actually contains the lab-overlay caller.

AC7 is written the way it is because the defect being closed is *a rule that only
a runbook implements*. Rule 24(a) said the k3s ``onex-lab`` overlay is re-applied
from every runtime-affecting merge; ``k8s/onex-lab/apply_lab_lane.sh`` had zero
callers in the org, so the sentence was true of nothing. A test that asserted the
applier module's behaviour alone would have passed just as happily with the module
wired to nothing at all.

So this file asserts the GRAPH: that the workflow has a job which reads the
agent's record and emits the receipt, that the artifact name it uploads is the one
the gate queries, and that the emitter cannot be skipped. Each assertion names the
specific way the 2026-09-11 state stayed invisible.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "runtime-rebuild-trigger.yml"
READER = REPO_ROOT / "scripts" / "ci" / "fetch_lab_overlay_record.py"
JOB = "verify-lab-overlay-converged"
LANE = "onex-lab-k3s"
SHA = "a" * 40


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def job(workflow: dict[str, Any]) -> dict[str, Any]:
    jobs = workflow["jobs"]
    assert JOB in jobs, (
        f"{WORKFLOW.name} carries no {JOB!r} job. Rule 24(a)'s onex-lab half is "
        "then documentation again: apply_lab_lane.sh had zero callers on "
        "2026-09-11 and the lane sat three days stale under a green trigger."
    )
    return jobs[JOB]


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return list(job["steps"])


def _step_text(job: dict[str, Any]) -> str:
    return json.dumps(_steps(job))


# --------------------------------------------------------------------------- #
# AC7 -- the caller exists in the graph                                       #
# --------------------------------------------------------------------------- #
def test_the_job_invokes_the_record_reader(job: dict[str, Any]) -> None:
    """The job must call the reader by path. Without this the receipt would be
    emitted from checks nothing produced."""
    assert "scripts/ci/fetch_lab_overlay_record.py" in _step_text(job)


def test_the_job_emits_the_receipt_on_the_k3s_lane(job: dict[str, Any]) -> None:
    text = _step_text(job)
    assert "lab_pass_receipt.py emit" in text
    assert f"--lane {LANE}" in text


def test_the_apply_itself_is_wired_in_the_agent(job: dict[str, Any]) -> None:
    """The host-side half. The record this job reads is written by the deploy
    agent, and the agent must actually call the overlay's own apply script --
    never a retyped kubectl sequence, which is the drift k8s/onex-lab exists to
    refuse."""
    applier = (
        REPO_ROOT / "scripts" / "deploy-agent" / "deploy_agent" / "lab_overlay.py"
    ).read_text(encoding="utf-8")
    assert "apply_lab_lane.sh" in applier

    agent = (
        REPO_ROOT / "scripts" / "deploy-agent" / "deploy_agent" / "agent.py"
    ).read_text(encoding="utf-8")
    assert "_apply_lab_overlay" in agent, (
        "the applier exists but nothing in the agent's deploy path calls it, "
        "which is the same zero-callers shape AC7 names"
    )
    assert agent.count("self._apply_lab_overlay(cmd)") == 1


def test_the_uploaded_artifact_name_matches_what_the_gate_queries(
    job: dict[str, Any],
) -> None:
    """The name IS the key. A receipt uploaded under any other name is a receipt
    the gate cannot find, which reads exactly like one that was never emitted."""
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "ci"))
    try:
        from lab_pass_receipt import EnumLabLane, artifact_name
    finally:
        sys.path.pop(0)

    expected = artifact_name(EnumLabLane.ONEX_LAB_K3S, SHA)
    assert expected == f"lab-pass-receipt-{LANE}-{SHA}"

    uploads = [
        step for step in _steps(job) if "upload-artifact" in str(step.get("uses", ""))
    ]
    assert len(uploads) == 1
    name = uploads[0]["with"]["name"]
    assert name.startswith(f"lab-pass-receipt-{LANE}-")
    assert "source_sha" in name, "the artifact must be keyed by the merged sha"
    assert uploads[0]["with"]["if-no-files-found"] == "error"


# --------------------------------------------------------------------------- #
# AC2's shape on this lane -- the emitter cannot be skipped                    #
# --------------------------------------------------------------------------- #
def test_the_emit_and_upload_steps_run_on_both_outcomes(job: dict[str, Any]) -> None:
    """The regressed condition in run 34657547387 was a SKIPPED verify job under a
    green run. A skipped emitter produces "nobody ran it" while meaning "it
    failed", and those are the two states the receipt exists to tell apart."""
    for step in _steps(job):
        name = str(step.get("name", ""))
        if "Emit" in name or "Upload" in name or "Read the deploy agent" in name:
            assert step.get("if") == "always()", name


def test_the_job_runs_even_when_compose_convergence_failed(job: dict[str, Any]) -> None:
    condition = " ".join(str(job["if"]).split())
    assert condition.startswith("always()"), (
        "without always() a failed compose convergence guard skips this job, and "
        "the lab lane's state goes unrecorded in exactly the window it matters"
    )
    # Still scoped: a no-op trigger has nothing to wait for, and a merge to main
    # targets stability-test, which is not a lab surface.
    assert "published == 'true'" in condition
    assert "runtime_lane == 'dev'" in condition


def test_the_job_waits_for_the_compose_convergence_guard(job: dict[str, Any]) -> None:
    """Ordering is load-bearing. The agent applies the overlay only after its own
    verify, so polling before the convergence guard returns would poll a record
    that cannot exist yet."""
    assert job["needs"] == ["trigger-rebuild", "verify-lane-converged"]


def test_the_job_runs_on_the_lab_host_fleet(job: dict[str, Any]) -> None:
    """The agent's HTTP surface is on the lab host's LAN. Hosted compute cannot
    see it, and this label is the runner carrying the host-gateway alias."""
    assert job["runs-on"] == ["self-hosted", "omnibase-deploy"]
    assert "host.docker.internal" in _step_text(job), (
        "localhost inside the runner container reaches the runner, not the host; "
        "every compose-dev receipt emitted before that was understood carried "
        "three identical connection-refused failures"
    )


def test_the_checkout_is_pinned_to_the_trusted_base_ref(job: dict[str, Any]) -> None:
    """Never the merge ref: a merged fork PR would put fork-authored code on the
    self-hosted fleet."""
    checkouts = [
        step for step in _steps(job) if "actions/checkout" in str(step.get("uses", ""))
    ]
    assert len(checkouts) == 1
    assert checkouts[0]["with"]["ref"] == "${{ github.event.pull_request.base.ref }}"
    assert checkouts[0]["with"]["persist-credentials"] is False


# --------------------------------------------------------------------------- #
# the lane value is distinct from the boot gate's                             #
# --------------------------------------------------------------------------- #
def test_the_k3s_lane_does_not_share_the_boot_gates_artifact_name() -> None:
    """``onex-lab`` is already the ephemeral kind cluster's lane, emitted for the
    same sha on this same repository by deliver-dev-candidate-to-staging.yml.
    ``evaluate_gate`` reads the NEWEST artifact per name, so two emitters on one
    name would silently discard one verdict."""
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "ci"))
    try:
        from lab_pass_receipt import EnumLabLane, artifact_name
    finally:
        sys.path.pop(0)

    assert artifact_name(EnumLabLane.ONEX_LAB, SHA) != artifact_name(
        EnumLabLane.ONEX_LAB_K3S, SHA
    )

    delivery = (
        REPO_ROOT / ".github" / "workflows" / "deliver-dev-candidate-to-staging.yml"
    ).read_text(encoding="utf-8")
    assert "lab-pass-receipt-onex-lab-${{ github.sha }}" in delivery, (
        "the premise of the separate lane value is that the boot gate already "
        "owns the onex-lab name; if that upload moved, re-derive the split"
    )
    trigger = WORKFLOW.read_text(encoding="utf-8")
    assert "lab-pass-receipt-onex-lab-${{" not in trigger


# --------------------------------------------------------------------------- #
# the reader always produces checks                                           #
# --------------------------------------------------------------------------- #
def _run_reader(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    out = tmp_path / "checks.json"
    return subprocess.run(
        [
            sys.executable,
            str(READER),
            "--sha",
            SHA,
            "--agent-url",
            "http://127.0.0.1:1",
            "--out",
            str(out),
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_an_unreachable_agent_still_writes_a_failing_check(tmp_path: Path) -> None:
    """The case that must never become a silent pass: no record, no lab apply, and
    the receipt has to say so naming the sha."""
    completed = _run_reader(
        tmp_path, "--wait-seconds", "0", "--poll-interval-seconds", "1"
    )

    assert completed.returncode == 0, completed.stderr
    checks = json.loads((tmp_path / "checks.json").read_text())
    assert len(checks) == 1
    assert checks[0]["name"] == "lab_overlay_record"
    assert checks[0]["ok"] is False
    assert checks[0]["evidence"], "a failing check with no evidence proves nothing"


def test_the_reader_refuses_an_abbreviated_sha(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(READER),
            "--sha",
            "abc1234",
            "--agent-url",
            "http://127.0.0.1:1",
            "--out",
            str(tmp_path / "checks.json"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 1
    assert "40-character lowercase commit sha" in completed.stderr


def test_the_reader_refuses_a_record_whose_payload_names_another_sha() -> None:
    """A name and a payload that disagree is a finding, and the same one
    evaluate_gate makes on an artifact."""
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "ci"))
    try:
        from fetch_lab_overlay_record import checks_from_record
    finally:
        sys.path.pop(0)

    checks = checks_from_record(
        {"sha": "b" * 40, "checks": [{"name": "x", "ok": True, "evidence": "y"}]},
        sha=SHA,
        url="http://agent/lab-overlay/" + SHA,
    )
    assert len(checks) == 1
    assert checks[0]["ok"] is False
    assert "disagree" in checks[0]["evidence"]


def test_a_record_with_no_checks_is_refused() -> None:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "ci"))
    try:
        from fetch_lab_overlay_record import checks_from_record
    finally:
        sys.path.pop(0)

    checks = checks_from_record(
        {"sha": SHA, "checks": []}, sha=SHA, url="http://agent/x"
    )
    assert checks[0]["ok"] is False
    assert "nothing was verified" in checks[0]["evidence"]
