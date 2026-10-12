# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Infra's receipt probe must receive the private census before measuring the lane."""

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/runtime-rebuild-trigger.yml"


def test_direct_receipt_fetches_private_census_before_the_probe() -> None:
    job = yaml.safe_load(WORKFLOW.read_text())["jobs"]["verify-lane-converged"]
    steps = job["steps"]
    fetches = [
        (i, step)
        for i, step in enumerate(steps)
        if step.get("uses") == "./.github/actions/fetch-census-overlay"
    ]
    assert len(fetches) == 1, "infra's lane_sync probe has no private census"
    index, fetch = fetches[0]
    probe = next(
        i for i, step in enumerate(steps) if "probe-lane" in step.get("run", "")
    )
    assert index < probe
    converge = next(i for i, step in enumerate(steps) if step.get("id") == "converge")
    claim = next(
        i
        for i, step in enumerate(steps)
        if 'echo "LAB_PASS_STARTED_AT=' in step.get("run", "")
    )
    assert claim < converge < index
    assert fetch["with"] == {
        "overlay-repo": "${{ vars.OMNI_CI_OVERLAY_REPO }}",
        "app-id": "${{ secrets.ONEXBOT_APP_ID }}",
        "private-key": "${{ secrets.ONEXBOT_APP_PRIVATE_KEY }}",
    }
    assert "continue-on-error" not in fetch
    # A failed fetch must still produce the real non-PASS evidence, including
    # lane_sync's unreadable-config check, rather than withholding a receipt.
    assert fetch["if"] == steps[probe]["if"]
    assert (
        steps[probe]["if"] == "always() && steps.converge.outputs.probe_lane != 'false'"
    )
    emit = next(
        step for step in steps if "lab_pass_receipt.py emit" in step.get("run", "")
    )
    assert "always()" in emit["if"]
    assert '--checks-json "${LAB_PASS_CHECKS}"' in emit["run"]


@pytest.mark.live_contact("tests/ci/fixtures/omn20008_direct_census_recording.json")
@pytest.mark.parametrize("variant", ["configured", "unconfigured"])
def test_recorded_private_census_is_required_for_a_passing_receipt(
    recorded_response, variant
) -> None:
    from scripts.ci import lab_pass_receipt as lab

    reading = recorded_response[variant]
    responses = iter(reading["responses"])

    def replay(argv, *, timeout, env=None):
        response = next(responses)
        return subprocess.CompletedProcess(
            list(argv), response["returncode"], response["stdout"], response["stderr"]
        )

    check = lab.check_lane_sync(runner=replay)
    assert check.to_dict() == reading["check"]
    captured = datetime.fromisoformat(recorded_response["_provenance"]["captured_utc"])
    receipt = lab.build_receipt(
        sha=recorded_response["_provenance"]["base_sha"],
        lane=lab.EnumLabLane.COMPOSE_DEV,
        started_at=captured,
        finished_at=captured,
        checks=[check],
        agent_command_id=None,
    )
    if variant == "configured":
        assert check.ok and not check.indeterminate
        assert json.loads(check.evidence)["lanes_checked"]
        assert receipt.result.value == "PASS"
    else:
        assert not check.ok and check.indeterminate
        assert "ONEX_WORKSPACE_CONFIG_ROOT" in check.evidence
        assert receipt.result.value != "PASS"
