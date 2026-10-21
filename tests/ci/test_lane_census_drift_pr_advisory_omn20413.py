# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Replay the recorded pull-request red of the lane census drift gate (OMN-20413).

omnibase_infra#5146, #5148, #5187, #5206 and #5207 failed 'Lane Census
Staleness + E-5 Env Drift (OMN-13034)' on one nonzero_drift_count finding for
the dev lane in the private overlay census, which no pull request here can
move. The replay drives the real checker on a snapshot shaped like the recorded
one: on a push it reproduces the recorded failure, on a pull request it is
warned and passes, any other finding still fails, and the script reads the
event from GITHUB_EVENT_NAME, which the workflow step does not override.
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


def _recorded_snapshot(run: dict[str, Any], tmp_path: Path) -> Path:
    snapshot = json.loads(NEUTRAL_SNAPSHOT.read_text(encoding="utf-8"))
    snapshot["emitted_at"] = run["census_emitted_at"]
    snapshot["drift_count"] = 1
    snapshot["findings"] = [
        {
            "lane": "dev",
            "kind": "container_unhealthy",
            "container": "recorded-dev-container",
            "detail": "the one dev-lane finding the recorded census carried",
        }
    ]
    path = tmp_path / "census-snapshot.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    return path


@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/lane_census_drift_pr_omn20413.json"
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

    assert main(argv, event_name=run["event_name"]) == 0
    captured = capsys.readouterr()
    assert "::warning title=LANE-CENSUS-DRIFT::[nonzero_drift_count] lane=dev" in (
        captured.err
    )
    assert "OK: the committed census and the manifest agree." in captured.out

    steps = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"][
        "lane-census-staleness"
    ]["steps"]
    step = next(
        s for s in steps if s["name"] == "Check census-vs-manifest drift (OMN-18949)"
    )
    assert "GITHUB_EVENT_NAME" not in (step.get("env") or {})
    assert "uv run python scripts/check_lane_census_drift.py" in step["run"]


@pytest.mark.parametrize(("event", "code"), [("pull_request", 0), ("push", 1)])
def test_the_script_reads_the_event_from_the_actions_environment(
    event: str, code: int, tmp_path: Path
) -> None:
    manifest = generator_consistent_manifest(tmp_path / "lane-manifest.yaml")
    snapshot = _recorded_snapshot(
        {"census_emitted_at": "2026-10-21T06:59:13.432660+00:00"}, tmp_path
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_lane_census_drift.py"),
            "--manifest",
            str(manifest),
            "--snapshot",
            str(snapshot),
        ],
        env={**os.environ, "GITHUB_EVENT_NAME": event},
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
            "kind": "container_unhealthy",
            "container": "recorded-dev-container",
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
