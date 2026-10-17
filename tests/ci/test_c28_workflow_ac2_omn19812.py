# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""AC2 coverage for the scheduled C28 dev-lane request and its record."""

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/chain-canary-c28-consumer-flow.yml"
)


def test_c28_resolves_the_dev_compose_project() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    resolver = workflow["jobs"]["resolve-lane"]
    step = next(step for step in resolver["steps"] if step.get("id") == "lane")
    assert step["uses"] == "./.github/actions/resolve-lab-lane"
    assert step["with"]["lanes-json"] == "${{ vars.LAB_LANES_JSON }}"
    assert step["with"]["match"] == "compose_project=omnibase-infra"
    assert set(step["with"]["require"].split()) == {
        "projection_url",
        "docker_runs_on",
    }
    assert resolver["outputs"]["lane"] == "${{ steps.lane.outputs.name }}"
    assert resolver["outputs"]["lane_json"] == "${{ steps.lane.outputs.lane }}"
    assert (
        resolver["outputs"]["docker_runs_on"]
        == "${{ steps.lane.outputs.docker_runs_on }}"
    )


def test_c28_request_uses_the_resolved_lane_and_writes_the_uploaded_record() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    job = workflow["jobs"]["c28-consumer-flow"]
    assert job["needs"] == "resolve-lane"
    prepare = next(
        step for step in job["steps"] if "C28_BASE_URL" in step.get("env", {})
    )
    assert prepare["env"]["C28_BASE_URL"] == (
        "${{ github.event.inputs.base_url || "
        "fromJSON(needs.resolve-lane.outputs.lane_json).projection_url }}"
    )
    assert prepare["env"]["LANE_NAME"] == "${{ needs.resolve-lane.outputs.lane }}"
    assert '--arg subject_lane "$LANE_NAME"' in prepare["run"]
    assert '--arg base_url "$C28_BASE_URL"' in prepare["run"]
    assert 'record: "c28-consumer-flow.json"' in prepare["run"]
    assert "> c28-request.json" in prepare["run"]
    run = next(
        step["run"]
        for step in job["steps"]
        if "onex node node_board_probe_effect" in step.get("run", "")
    )
    assert "--input c28-request.json" in run


def test_c28_upload_requires_the_record_even_after_probe_failure() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["c28-consumer-flow"]["steps"]
    upload = next(
        step
        for step in steps
        if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert upload["if"] == "always()"
    assert upload["with"]["name"] == "c28-consumer-flow"
    assert upload["with"]["path"] == "c28-consumer-flow.json"
    assert upload["with"]["if-no-files-found"] == "error"
    record_check = next(
        step
        for step in steps
        if "scripts/ci/assert_evidence_artifact.py" in step.get("run", "")
    )
    assert record_check["if"] == "always()"
    assert "--artifact c28-consumer-flow" in record_check["run"]
    assert "--require c28-consumer-flow.json" in record_check["run"]
