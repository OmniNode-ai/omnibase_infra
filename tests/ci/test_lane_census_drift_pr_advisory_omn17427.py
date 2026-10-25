# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Replay the recorded pull-request red of the lane census drift gate (OMN-17427).

omnibase_infra#5187 failed 'Lane Census Staleness + E-5 Env Drift (OMN-13034)'
on 2026-10-24 on one nonzero_drift_count finding: a dev-lane oneshot_failed in
the private overlay census, which no pull request here can move. The replay
drives the real checker on a snapshot carrying the recorded finding: on a push
it reproduces the recorded failure, on a pull request it is warned and passes,
any other finding still fails, and the script reads the event from
GITHUB_EVENT_NAME, which the workflow step does not override.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/lane-census-staleness.yml"

sys.path.insert(0, str(ROOT / "scripts"))

from check_lane_census_drift import main

from tests.helpers.lane_census_fixtures import (
    NEUTRAL_SNAPSHOT,
    generator_consistent_manifest,
)

_FIXTURE = "tests/ci/fixtures/ci_live_contact/lane_census_drift_pr_omn17427.json"


def _recorded_snapshot(run: dict[str, Any], tmp_path: Path) -> Path:
    snapshot = json.loads(NEUTRAL_SNAPSHOT.read_text(encoding="utf-8"))
    snapshot["emitted_at"] = run["census_emitted_at"]
    snapshot["drift_count"] = 1
    snapshot["findings"] = [dict(run["census_finding"])]
    path = tmp_path / "census-snapshot.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    return path


@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/lane_census_drift_pr_omn17427.json"
)
def test_recorded_pull_request_drift_red_is_advisory_on_pull_requests_only(
    recorded_response: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = recorded_response["response"]
    assert run["event_name"] == "pull_request"
    assert run["conclusion"] == "failure"
    manifest = generator_consistent_manifest(tmp_path / "lane-manifest.yaml")
    snapshot = _recorded_snapshot(run, tmp_path)
    argv = ["--manifest", str(manifest), "--snapshot", str(snapshot)]

    assert main(argv, event_name="push") == 1
    assert run["finding_line"] in capsys.readouterr().out

    assert main(argv, event_name="workflow_dispatch") == 1
    assert run["finding_line"] in capsys.readouterr().out

    assert main(argv, event_name=run["event_name"]) == 0
    captured = capsys.readouterr()
    assert "::warning title=LANE-CENSUS-DRIFT::[nonzero_drift_count] lane=dev" in (
        captured.err
    )
    assert "OK: the committed census and the manifest agree." in captured.out

    steps = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"][
        "lane-census-staleness"
    ]["steps"]
    checker_steps = [
        s for s in steps if "scripts/check_lane_census_drift.py" in s.get("run", "")
    ]
    assert len(checker_steps) == 1
    step = checker_steps[0]
    assert "GITHUB_EVENT_NAME" not in (step.get("env") or {})


@pytest.mark.parametrize(
    ("event", "code"),
    [
        ("pull_request", 0),
        ("push", 1),
        ("workflow_dispatch", 1),
        ("pull_request_target", 1),
        ("merge_group", 1),
        ("schedule", 1),
        ("PULL_REQUEST", 1),
        (" pull_request", 1),
        ("pull_request ", 1),
        ("", 1),
    ],
)
def test_the_script_reads_the_event_from_the_actions_environment(
    event: str, code: int, tmp_path: Path
) -> None:
    manifest = generator_consistent_manifest(tmp_path / "lane-manifest.yaml")
    run = json.loads((ROOT / _FIXTURE).read_text(encoding="utf-8"))["response"]
    snapshot = _recorded_snapshot(run, tmp_path)
    env = {k: v for k, v in os.environ.items() if k != "GITHUB_EVENT_NAME"}
    if event:
        env["GITHUB_EVENT_NAME"] = event
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_lane_census_drift.py"),
            "--manifest",
            str(manifest),
            "--snapshot",
            str(snapshot),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == code, proc.stdout + proc.stderr


def test_pull_request_still_fails_every_other_finding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only nonzero_drift_count is advisory: a miscounted snapshot still fails."""
    manifest = generator_consistent_manifest(tmp_path / "lane-manifest.yaml")
    snapshot = json.loads(NEUTRAL_SNAPSHOT.read_text(encoding="utf-8"))
    snapshot["drift_count"] = 2
    snapshot["findings"] = [
        {
            "lane": "dev",
            "kind": "oneshot_failed",
            "container": "omnibase-infra-forward-migration",
            "detail": "one finding under a count of two",
        }
    ]
    path = tmp_path / "census-snapshot.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")

    argv = ["--manifest", str(manifest), "--snapshot", str(path)]
    assert main(argv, event_name="pull_request") == 1
    out = capsys.readouterr().out
    assert "[finding_count_mismatch]" in out
    assert "[nonzero_drift_count]" not in out


@pytest.mark.parametrize("event", ["pull_request", "push", "merge_group", ""])
@pytest.mark.parametrize("mismatched_count", [False, True])
def test_json_preserves_advisory_and_blocking_findings(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    event: str,
    mismatched_count: bool,
) -> None:
    manifest = generator_consistent_manifest(tmp_path / "lane-manifest.yaml")
    run = json.loads((ROOT / _FIXTURE).read_text(encoding="utf-8"))["response"]
    path = _recorded_snapshot(run, tmp_path)
    if mismatched_count:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        snapshot["drift_count"] = 2
        path.write_text(json.dumps(snapshot), encoding="utf-8")
    argv = ["--manifest", str(manifest), "--snapshot", str(path), "--json"]
    code = main(argv, event_name=event)
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    blocking = [f["kind"] for f in result["findings"]]
    advisory = [f["kind"] for f in result["advisory_findings"]]
    assert advisory == (["nonzero_drift_count"] if event == "pull_request" else [])
    assert ("nonzero_drift_count" in blocking) == (event != "pull_request")
    assert ("finding_count_mismatch" in blocking) == mismatched_count
    assert code == (1 if blocking else 0)
    assert result["snapshot_findings"] == 1
    assert result["manifest_lanes"] > 0
    assert result["manifest_containers"] > 0
    assert captured.err == ""


@pytest.mark.parametrize(("event", "code"), [("pull_request", 0), ("push", 1)])
def test_workflow_command_preserves_the_actions_event(
    tmp_path: Path, event: str, code: int
) -> None:
    """Execute the committed shell command, including its environment scopes."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    job = workflow["jobs"]["lane-census-staleness"]
    steps = [
        s
        for s in job["steps"]
        if "scripts/check_lane_census_drift.py" in s.get("run", "")
    ]
    assert len(steps) == 1
    step = steps[0]
    census_dir = tmp_path / "config" / "lab" / "lane-census"
    census_dir.mkdir(parents=True)
    generator_consistent_manifest(census_dir / "lane-manifest.yaml")
    run = json.loads((ROOT / _FIXTURE).read_text(encoding="utf-8"))["response"]
    _recorded_snapshot(run, census_dir)
    env = dict(os.environ)
    env.update(
        GITHUB_EVENT_NAME=event,
        ONEX_WORKSPACE_CONFIG_ROOT=str(tmp_path),
        PYTHONDONTWRITEBYTECODE="1",
    )
    env.pop("LANE_MANIFEST", None)
    for scope in (workflow, job, step):
        env.update(scope.get("env") or {})
    proc = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert proc.returncode == code, proc.stdout + proc.stderr
    if event == "pull_request":
        assert "::warning title=LANE-CENSUS-DRIFT::" in proc.stderr
    else:
        assert "[nonzero_drift_count]" in proc.stdout
