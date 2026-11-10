# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-22896: security-scan.yml runs TruffleHog on a hosted runner, with a planted control.

The repository is public, so the secret-scan job must run on a hosted runner. The scanner is the
TruffleHog image by the digest the reference security-scan workflow of this organization pins
(omnimarket), spelled here because this test cannot read another repository. The TruffleHog action
only wraps a `docker run` of `:latest`, so pinning the action pins nothing that scans: no step may
use the action, and every scanner step runs the pinned image. The control detector must match a planted token and
must not match clean text or its own file.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "security-scan.yml"
CONTROL_CONFIG = REPO_ROOT / ".github" / "trufflehog-control.yml"

TRUFFLEHOG_ACTION = "trufflesecurity/trufflehog"
TRUFFLEHOG_IMAGE = (
    "ghcr.io/trufflesecurity/trufflehog@sha256:"
    "674b2f187620a19df77f44cf9a7d77efcc581a5662accef2f2739cbb46ba4a3a"
)
SCANNER_RUN = 'docker run --rm -v .:/tmp -w /tmp "$TRUFFLEHOG_IMAGE"'
HOSTED_RUNNER = "ubuntu-latest"
CONTROL_ARGS = (
    "--config=.github/trufflehog-control.yml",
    "--include-detectors=CustomRegex",
    "--results=unverified",
    "--no-verification",
)


def _workflow() -> dict[str, Any]:
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _job() -> dict[str, Any]:
    job = _workflow()["jobs"]["secret-scan"]
    assert isinstance(job, dict)
    return job


def _scanner_steps() -> list[dict[str, Any]]:
    return [s for s in _job()["steps"] if SCANNER_RUN in str(s.get("run", ""))]


def _planted_token() -> str:
    # Built from parts so this file never holds a full control token.
    return "ONEX_SCAN_" + "CONTROL_" + "abcdefghijklmnopqrstuvwxyz012345"


def _detector_regex() -> re.Pattern[str]:
    config = yaml.safe_load(CONTROL_CONFIG.read_text(encoding="utf-8"))
    (detector,) = config["detectors"]
    assert detector["name"] == "OnexScanControl"
    return re.compile(detector["regex"]["token"])


def test_job_runs_on_a_hosted_runner_only() -> None:
    runs_on = _job()["runs-on"]
    assert runs_on == HOSTED_RUNNER
    assert "self-hosted" not in str(runs_on)
    assert "vars." not in str(runs_on)


def test_scanner_steps_use_the_reference_pin() -> None:
    assert _workflow()["env"]["TRUFFLEHOG_IMAGE"] == TRUFFLEHOG_IMAGE
    steps = _scanner_steps()
    assert len(steps) == 3
    assert not [
        s
        for s in _job()["steps"]
        if str(s.get("uses", "")).startswith(TRUFFLEHOG_ACTION)
    ]
    for step in steps:
        assert "TRUFFLEHOG_IMAGE" not in step.get("env", {})
        assert "--fail" in step["run"]
        assert "--no-update" in step["run"]


def test_ordinary_scans_are_verified_only_and_the_history_is_full() -> None:
    job = _job()
    checkout = job["steps"][0]
    assert checkout["uses"].startswith("actions/checkout@")
    assert checkout["with"]["fetch-depth"] == 0
    ordinary = [s for s in _scanner_steps() if "--only-verified" in s["run"]]
    for step in ordinary:
        assert "--exclude-detectors=lob" in step["run"]
    assert {s["name"] for s in ordinary} == {
        "TruffleHog Secret Scanning (push diff)",
        "TruffleHog Secret Scanning (full scan)",
    }


def test_control_step_reads_the_declared_detector_unverified() -> None:
    (control,) = [
        s for s in _scanner_steps() if s["name"] == "TruffleHog planted control"
    ]
    args = control["run"].replace("\\\n", " ").split()
    assert set(CONTROL_ARGS) <= set(args)
    assert "--fail" in args
    assert "--only-verified" not in args
    assert "if" not in control


def test_control_detector_matches_the_planted_token() -> None:
    assert _detector_regex().search(f"x={_planted_token()}\n")


def test_control_detector_ignores_clean_text_and_its_own_file() -> None:
    pattern = _detector_regex()
    assert not pattern.search("ONEX_SCAN_CONTROL_short")
    assert not pattern.search("a clean line with no token")
    assert not pattern.search(CONTROL_CONFIG.read_text(encoding="utf-8"))
