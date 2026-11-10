# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18200 lab-overlay record reader, after the k3s overlay job was retired.

The ``verify-lab-overlay-converged`` job that called the reader, and the k3s
receipt it emitted, were removed from the rebuild trigger on the operator's
2026-11-10 ruling that the dead lab Kubernetes pod step is retired (OMN-23508).
The graph assertions that pinned that job live on as the absence assertions in
``test_retired_k3s_lab_overlay_job_omn23422.py``. What stays here is what still
exists: the deploy agent's own overlay apply, the reader script, and the
distinct artifact keys of the receipt lanes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "runtime-rebuild-trigger.yml"
READER = REPO_ROOT / "scripts" / "ci" / "fetch_lab_overlay_record.py"
SHA = "a" * 40


def test_the_apply_itself_is_wired_in_the_agent() -> None:
    """The host-side half. The overlay record is written by the deploy
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
    # OMN-19501: the call now passes the settle worker's own sha, stamp and
    # capture as keywords, so count the call site, not one argument spelling.
    assert agent.count("self._apply_lab_overlay(") == 1


# --------------------------------------------------------------------------- #
# the lane value is distinct from the boot gate's                             #
# --------------------------------------------------------------------------- #
def test_the_k3s_lane_does_not_share_the_boot_gates_artifact_name() -> None:
    """The candidate smoke and persistent apply have distinct artifact keys.

    Only the persistent apply may satisfy delivery's exact-sha lab premise.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "ci"))
    try:
        from lab_pass_receipt import EnumLabLane, artifact_name
    finally:
        sys.path.pop(0)

    assert artifact_name(EnumLabLane.CANDIDATE_BOOT, SHA) != artifact_name(
        EnumLabLane.ONEX_LAB_K3S, SHA
    )

    delivery = (
        REPO_ROOT / ".github" / "workflows" / "deliver-dev-candidate-to-staging.yml"
    ).read_text(encoding="utf-8")
    assert "candidate-boot-receipt-${{ github.sha }}" in delivery
    assert "lab-pass-receipt-onex-lab-${{ github.sha }}" not in delivery
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
            "--repo",
            "OmniNode-ai/omnibase_infra",
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
            "--repo",
            "OmniNode-ai/omnibase_infra",
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
